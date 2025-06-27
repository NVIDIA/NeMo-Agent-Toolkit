# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import functools
import logging
import typing
from collections.abc import Callable
from pathlib import Path

import click
from pydantic_core import SchemaValidator

from aiq.cli.cli_utils.config_override import load_and_override_config
from aiq.cli.type_registry import GlobalTypeRegistry
from aiq.cli.type_registry import RegisteredFrontEndInfo
from aiq.data_models.config import AIQConfig
from aiq.utils.data_models.schema_validator import validate_schema
from aiq.utils.type_utils import DecomposedType

logger = logging.getLogger(__name__)


class StartCommandGroup(click.Group):

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        name: str | None = None,
        invoke_without_command: bool = False,
        no_args_is_help: bool | None = None,
        subcommand_metavar: str | None = None,
        chain: bool = False,
        result_callback: Callable[..., typing.Any] | None = None,
        **attrs: typing.Any,
    ):
        super().__init__(name=name,
                         invoke_without_command=invoke_without_command,
                         no_args_is_help=no_args_is_help,
                         subcommand_metavar=subcommand_metavar,
                         chain=chain,
                         result_callback=result_callback,
                         **attrs)

        self._commands: dict[str, click.Command] | None = None
        self._registered_front_ends: dict[str, RegisteredFrontEndInfo] = {}

    def _build_params(self, front_end: RegisteredFrontEndInfo) -> list[click.Parameter]:

        params: list[click.Parameter] = []

        # First two are always the config file and override
        params.append(
            click.Option(param_decls=["--config_file"],
                         type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
                         required=True,
                         help=("A JSON/YAML file that sets the parameters for the workflow.")))
        params.append(
            click.Option(
                param_decls=["--override"],
                type=(str, str),
                multiple=True,
                help="Override config values using dot notation (e.g., --override llms.nim_llm.temperature 0.7)"))

        fields = front_end.config_type.model_fields
        for name, field in fields.items():

            if (name in ("override", "config_file")):
                raise ValueError(
                    "Cannot have a field named 'override' or 'config_file' in the front end config. These are reserved."
                )

            # Skip init-only fields since we dont want to set them in the constructor. Must check for False explicitly
            if (field.init == False):  # noqa: E712, pylint: disable=singleton-comparison
                continue

            if (field.annotation is None):
                raise ValueError(f"Field {name} has no type annotation. Types are required for Front End Plugins.")

            # Decompose the type into its origin and arguments
            decomposed_type = DecomposedType(field.annotation)

            param_decls = [f"--{name}"]
            multiple = False

            # Remove any optional types
            while (decomposed_type.is_optional):
                decomposed_type = decomposed_type.get_optional_type()

            if (decomposed_type.is_union):
                raise ValueError(f"Invalid field '{name}'.Unions are only supported for optional parameters.")

            # Handle the types
            if (issubclass(decomposed_type.root, Path)):
                param_type = click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path)

            elif (issubclass(decomposed_type.root, (list, tuple, set))):
                if (len(decomposed_type.args) == 1):
                    param_type = decomposed_type.args[0]
                else:
                    param_type = None

                multiple = True
            else:
                param_type = decomposed_type.root

            if (field.alias is not None):
                param_decls = [f"--{field.alias}", f"{name}"]

            params.append(
                click.Option(param_decls=param_decls,
                             type=param_type,
                             required=False,
                             multiple=multiple,
                             help=field.description))

        return params

    def _load_commands(self) -> dict[str, click.Command]:

        if (self._commands is not None):
            return self._commands

        from aiq.runtime.loader import PluginTypes
        from aiq.runtime.loader import discover_and_register_plugins

        # Only load front ends here for performance. Ensures a responsive CLI
        discover_and_register_plugins(PluginTypes.FRONT_END)

        all_front_ends = GlobalTypeRegistry.get().get_registered_front_ends()

        self._commands = {}

        for front_end in all_front_ends:

            registered_front_end = GlobalTypeRegistry.get().get_front_end(config_type=front_end.config_type)

            # Build the command parameters
            params: list[click.Parameter] = self._build_params(registered_front_end)
            help_msg = f"Run an AIQ Toolkit workflow using the {registered_front_end.local_name} front end."

            cmd = click.Command(name=registered_front_end.local_name,
                                params=params,
                                help=help_msg,
                                callback=functools.partial(click.pass_context(self.invoke_subcommand),
                                                           cmd_name=front_end.local_name))

            self._registered_front_ends[front_end.local_name] = registered_front_end
            self._commands[front_end.local_name] = cmd

        return self._commands

    def invoke_subcommand(self,
                          ctx: click.Context,
                          cmd_name: str,
                          override: tuple[tuple[str, str], ...],
                          **kwargs) -> int | None:

        from aiq.runtime.loader import PluginTypes
        from aiq.runtime.loader import discover_and_register_plugins
        import os

        # Get mode-specific config files from context
        friday_config_file = ctx.obj.get('friday_config_file') if ctx.obj else None
        on_call_config_file = ctx.obj.get('on_call_config_file') if ctx.obj else None
        slack_config_file = ctx.obj.get('slack_config_file') if ctx.obj else None
        context_override = ctx.obj.get('override', ()) if ctx.obj else ()
        
        # Use context override if available, otherwise use the passed override
        if context_override:
            override = context_override

        # Validate that we have at least one mode-specific config file
        if not friday_config_file and not on_call_config_file and not slack_config_file:
            raise click.ClickException("At least one mode-specific config file (--friday_config_file, --on_call_config_file, or --slack_config_file) must be provided.")

        # Store mode-specific config files in environment variables for FastAPI frontend
        if friday_config_file:
            os.environ["AIQ_FRIDAY_CONFIG_FILE"] = str(friday_config_file.absolute())
            logger.info("FRIDAY mode config file: '%s'", friday_config_file)
        if on_call_config_file:
            os.environ["AIQ_ON_CALL_CONFIG_FILE"] = str(on_call_config_file.absolute())
            logger.info("ON_CALL mode config file: '%s'", on_call_config_file)
        if slack_config_file:
            os.environ["AIQ_SLACK_CONFIG_FILE"] = str(slack_config_file.absolute())
            logger.info("SLACK mode config file: '%s'", slack_config_file)

        # Here we need to ensure all objects are loaded before we try to create the config object
        discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

        logger.info("Starting AIQ Toolkit with mode-specific configurations")

        # Use the first available mode-specific config for frontend setup
        # This is only needed to initialize the frontend properly, actual mode handling is done in the worker
        setup_config_file = friday_config_file or on_call_config_file or slack_config_file
        config_dict = load_and_override_config(setup_config_file, override)

        # Get the front end for the command
        front_end: RegisteredFrontEndInfo = self._registered_front_ends[cmd_name]

        config = validate_schema(config_dict, AIQConfig)

        # Check that we have the right kind of front end
        if (not isinstance(config.general.front_end, front_end.config_type)):
            logger.warning(
                "The front end type in the config file (%s) does not match the command name (%s). "
                "Overwriting the config file front end.",
                config.general.front_end.type,
                cmd_name)

            # Set the front end config
            config.general.front_end = front_end.config_type()

        front_end_config = config.general.front_end

        # Iterate over the parameters and set them in the config
        for param, value in kwargs.items():

            # Skip default values so we dont overwrite the config
            if (ctx.get_parameter_source(param) == click.core.ParameterSource.DEFAULT):
                continue

            setattr(front_end_config, param, value)

        # Validate the config once more to ensure that all parameters are set correctly
        schema_validator = SchemaValidator(schema=front_end_config.__pydantic_core_schema__)
        schema_validator.validate_python(front_end_config.__dict__)

        try:

            async def run_plugin():

                # From the config, get the registered front end plugin
                front_end_info = GlobalTypeRegistry.get().get_front_end(config_type=type(front_end_config))

                # Create the front end plugin
                async with front_end_info.build_fn(front_end_config, config) as front_end_plugin:

                    # Run the front end plugin
                    await front_end_plugin.run()

            return asyncio.run(run_plugin())

        except Exception as e:
            logger.error("Failed to initialize workflow", exc_info=True)
            raise click.ClickException(str(e)) from e

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:

        return self._load_commands().get(cmd_name)

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(self._load_commands().keys())


@click.command(name=__name__,
               invoke_without_command=False,
               help="Run an AIQ Toolkit workflow using a front end configuration.",
               cls=StartCommandGroup)
@click.option("--config_file",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
              required=False,
              help="A JSON/YAML file that sets the parameters for the workflow.")
@click.option("--friday_config_file",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
              required=False,
              help="A JSON/YAML file that sets the parameters for the FRIDAY mode workflow.")
@click.option("--on_call_config_file",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
              required=False,
              help="A JSON/YAML file that sets the parameters for the ON_CALL mode workflow.")
@click.option("--slack_config_file",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
              required=False,
              help="A JSON/YAML file that sets the parameters for the SLACK mode workflow.")
@click.option("--override",
              nargs=2,
              multiple=True,
              type=str,
              help="Override config values using dot notation (e.g., --override llms.nim_llm.temperature 0.7)")
@click.pass_context
def start_command(ctx: click.Context,
                  config_file: Path | None,
                  friday_config_file: Path | None,
                  on_call_config_file: Path | None,
                  slack_config_file: Path | None,
                  override: tuple[tuple[str, str], ...],
                  **kwargs) -> None:
    """
    Start AIQ Toolkit with the specified front end.
    """
    
    # Validate that we have at least one mode-specific config file
    if not friday_config_file and not on_call_config_file and not slack_config_file:
        raise click.ClickException("At least one mode-specific config file (--friday_config_file, --on_call_config_file, or --slack_config_file) must be provided.")
    
    # Store config files in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['friday_config_file'] = friday_config_file
    ctx.obj['on_call_config_file'] = on_call_config_file
    ctx.obj['slack_config_file'] = slack_config_file
    ctx.obj['override'] = override
