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

import click

logger = logging.getLogger(__name__)


async def uninstall_packages(packages: list[dict[str, str]]) -> None:

    from aiq.cli.type_registry import GlobalTypeRegistry
    from aiq.registry_handlers.schemas.package import PackageNameVersionList
    from aiq.runtime.loader import PluginTypes
    from aiq.runtime.loader import discover_and_register_plugins
    from aiq.settings.global_settings import GlobalSettings

    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

    registry = GlobalTypeRegistry.get()

    config_dict = {"channels": {"uninstall_local": {"_type": "local"}}}
    registry_config = GlobalSettings.get().model_validate(config_dict)
    local_registry_config = registry_config.channels.get("uninstall_local", None)

    if (local_registry_config is None):
        logger.error("Channel runtime instance not found.")

    try:
        package_name_list = PackageNameVersionList(**{"packages": packages})
    except Exception as e:
        logger.exception("Error validating package format: %s", e, exc_info=True)
        return

    async with AsyncExitStack() as stack:
        registry_handler_info = registry.get_registry_handler(type(local_registry_config))
        registry_handler = await stack.enter_async_context(registry_handler_info.build_fn(local_registry_config))
        await stack.enter_async_context(registry_handler.remove(packages=package_name_list))


@click.group(name=__name__, invoke_without_command=True, help=("Uninstall plugin packages from the local environment."))
@click.argument("packages", type=str)
def uninstall_command(packages: str) -> None:
    """
    Uninstall plugin packages from the local environment.
    """

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

    asyncio.run(uninstall_packages(packages=packages_versions))
