# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
from contextlib import AsyncExitStack
from pathlib import Path

import click

from nat.data_models.registry_handler import RegistryHandlerBaseConfig
from nat.utils.data_models.schema_validator import validate_yaml

logger = logging.getLogger(__name__)


async def publish_artifact(registry_handler_config: RegistryHandlerBaseConfig, package_root: str) -> None:

    from nat.cli.type_registry import GlobalTypeRegistry
    from nat.registry_handlers.package_utils import build_artifact

    registry = GlobalTypeRegistry.get()

    async with AsyncExitStack() as stack:

        registry_handler_info = registry.get_registry_handler(type(registry_handler_config))
        registry_handler = await stack.enter_async_context(registry_handler_info.build_fn(registry_handler_config))
        try:
            artifact = build_artifact(package_root=package_root)
        except Exception as e:
            logger.exception("Error building artifact: %s", e, exc_info=True)
            return
        await stack.enter_async_context(registry_handler.publish(artifact=artifact))


@click.group(name=__name__,
             invoke_without_command=True,
             help=("Publish local NAT artifacts to a remote "
                   "registry from package repository."))
@click.option(
    "--config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    callback=validate_yaml,
    required=False,
    help=("A YAML file to override configured channel settings."),
)
@click.option(
    "-c",
    "--channel",
    type=str,
    required=True,
    help=("The remote registry channel to use when publishing the NAT artifact."),
)
@click.argument("package_root", type=str)
def publish(channel: str, config_file: str, package_root: str) -> None:
    """
    Publish NAT artifacts with the specified configuration
    """
    from nat.settings.global_settings import GlobalSettings

    settings = GlobalSettings().get()

    if (config_file is not None):
        settings = settings.override_settings(config_file)

    try:
        publish_channel_config = settings.channels.get(channel)

        if (publish_channel_config is None):
            logger.error("Publish channel '%s' has not been configured.", channel)
            return
    except Exception as e:
        logger.exception("Error loading user settings: %s", e, exc_info=True)
        return

    asyncio.run(publish_artifact(registry_handler_config=publish_channel_config, package_root=package_root))
