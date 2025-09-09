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


async def remove_artifact(registry_handler_config: RegistryHandlerBaseConfig, packages: list[dict[str, str]]) -> None:

    from nat.cli.type_registry import GlobalTypeRegistry
    from nat.registry_handlers.schemas.package import PackageNameVersionList

    registry = GlobalTypeRegistry.get()

    async with AsyncExitStack() as stack:

        registry_handler_info = registry.get_registry_handler(type(registry_handler_config))
        registry_handler = await stack.enter_async_context(registry_handler_info.build_fn(registry_handler_config))

        try:
            package_name_list = PackageNameVersionList(**{"packages": packages})
        except Exception as e:
            logger.exception("Invalid package format: '%s'", e)

        await stack.enter_async_context(registry_handler.remove(packages=package_name_list))


@click.group(name=__name__,
             invoke_without_command=True,
             help=("Remove NAT artifact from a remote registry by name and version."))
@click.argument("packages", type=str)
@click.option(
    "--config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    callback=validate_yaml,
    required=False,
    help=("A YAML file to override the channel settings."),
)
@click.option(
    "-c",
    "--channel",
    type=str,
    required=True,
    help=("The remote registry channel that will remove the NAT artifact."),
)
def remove(channel: str, config_file: str, packages: str) -> None:
    """
    Remove NAT artifacts from a remote registry.
    """

    from nat.settings.global_settings import GlobalSettings

    # Extract package name and version
    packages = packages.split()
    packages_versions = []
    for package in packages:
        package_dict = {}
        package_version = package.split("==")
        if (len(package_version) == 1):
            package_dict["name"] = package_version[0]
            msg = f"No package version provided for '{package_version[0]}'."
            logger.warning(msg)
        elif (len(package_version) == 2):
            package_dict["name"] = package_version[0]
            package_dict["version"] = package_version[1]
        else:
            msg = f"Invalid input: '{package}'"
            logger.error(msg)
        if (package_dict):
            packages_versions.append(package_dict)

    settings = GlobalSettings().get()

    if (config_file is not None):
        settings = settings.override_settings(config_file)

    try:
        remove_channel_config = settings.channels.get(channel)

        if (remove_channel_config is None):
            logger.error("Remove channel '%s' has not been configured.", channel)
            return
    except Exception as e:
        logger.exception("Error loading user settings: %s", e)
        return

    asyncio.run(remove_artifact(registry_handler_config=remove_channel_config, packages=packages_versions))
