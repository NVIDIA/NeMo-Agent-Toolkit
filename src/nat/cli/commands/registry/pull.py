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


async def pull_artifact(registry_handler_config: RegistryHandlerBaseConfig, packages: list[str]) -> None:

    from nat.cli.type_registry import GlobalTypeRegistry
    from nat.registry_handlers.schemas.package import PackageNameVersion
    from nat.registry_handlers.schemas.pull import PullPackageWhl
    from nat.registry_handlers.schemas.pull import PullRequestPackages

    registry = GlobalTypeRegistry.get()

    async with AsyncExitStack() as stack:

        registry_handler_info = registry.get_registry_handler(type(registry_handler_config))
        registry_handler = await stack.enter_async_context(registry_handler_info.build_fn(registry_handler_config))

        try:
            package_list = []
            for package in packages:

                package_data = {}

                assert len(package) > 0, f"Supplied invalid package '{package}'."

                if package[:-4] == ".whl":
                    package_data["whl_path"] = package
                    package_list.append(PullPackageWhl(**package_data))
                else:
                    package_split = package.split("==")

                    assert len(package_split) in (1, 2), f"Supplied invalid package '{package}'."

                    package_data["name"] = package_split[0]

                    if (package_split == 2):
                        package_data["version"] = package_split[1]

                    package_list.append(PackageNameVersion(**package_data))

            validated_packages = PullRequestPackages(packages=package_list)

        except Exception as e:
            logger.exception("Error processing package names: %s", e)
            return

        await stack.enter_async_context(registry_handler.pull(packages=validated_packages))


@click.group(name=__name__,
             invoke_without_command=True,
             help=("Pull NAT artifacts from a remote registry "
                   "by package name."))
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
    help=("The remote registry channel to use when pulling the NAT artifact."),
)
@click.argument("packages", type=str)
def pull(channel: str, config_file: str, packages: str) -> None:
    """
    Pull NAT artifacts from a remote registry channel.
    """

    from nat.settings.global_settings import GlobalSettings

    packages = packages.split()

    settings = GlobalSettings().get()

    if (config_file is not None):
        settings = settings.override_settings(config_file)

    try:
        pull_channel_config = settings.channels.get(channel)

        if (pull_channel_config is None):
            logger.error("Pull channel '%s' has not been configured.", channel)
            return
    except Exception as e:
        logger.exception("Error loading user settings: %s", e)
        return

    asyncio.run(pull_artifact(pull_channel_config, packages))
