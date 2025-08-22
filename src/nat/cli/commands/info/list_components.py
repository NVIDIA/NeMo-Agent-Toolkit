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

from nat.data_models.component import ComponentEnum
from nat.data_models.registry_handler import RegistryHandlerBaseConfig
from nat.registry_handlers.schemas.search import SearchFields

logger = logging.getLogger(__name__)


async def search_artifacts(registry_handler_config: RegistryHandlerBaseConfig,
                           component_types: list[ComponentEnum],
                           visualize: bool,
                           query: str,
                           num_results: int,
                           query_fields: list[SearchFields],
                           save_path: str | None) -> None:

    from nat.cli.type_registry import GlobalTypeRegistry
    from nat.registry_handlers.schemas.search import SearchQuery

    registry = GlobalTypeRegistry.get()

    async with AsyncExitStack() as stack:

        registry_handler_info = registry.get_registry_handler(type(registry_handler_config))
        registry_handler = await stack.enter_async_context(registry_handler_info.build_fn(registry_handler_config))

        if (len(component_types) == 0):
            component_types = [t.value for t in ComponentEnum]

        if (len(query_fields) == 0):
            query_fields = (SearchFields.ALL, )

        query = SearchQuery(query=query, fields=query_fields, top_k=num_results, component_types=component_types)
        search_response = await stack.enter_async_context(registry_handler.search(query=query))

        if (visualize):
            registry_handler.visualize_search_results(search_response=search_response)
        if (save_path is not None):
            registry_handler.save_search_results(search_response=search_response, save_path=save_path)


@click.group(name=__name__, invoke_without_command=True, help="List the locally registered NAT components.")
@click.option(
    "-t",
    "--types",
    "component_types",
    multiple=True,
    type=click.Choice([e.value for e in ComponentEnum], case_sensitive=False),
    required=False,
    help=("Filter the search by NAT component type."),
)
@click.option(
    "-o",
    "--output_path",
    type=str,
    required=False,
    help=("Path to save search results."),
)
@click.option(
    "-q",
    "--query",
    type=str,
    default="",
    required=False,
    help=("The query string."),
)
@click.option(
    "-n",
    "--num_results",
    type=int,
    default=-1,
    required=False,
    help=("Number of results to return."),
)
@click.option(
    "-f",
    "--fields",
    multiple=True,
    type=click.Choice([e.value for e in SearchFields], case_sensitive=False),
    required=False,
    help=("Fields used when applying query."),
)
def list_components(fields: list[SearchFields],
                    query: str,
                    num_results: int,
                    component_types: list[ComponentEnum],
                    output_path: str | None = None) -> None:

    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins
    from nat.settings.global_settings import GlobalSettings

    discover_and_register_plugins(PluginTypes.ALL)

    config_dict = {"channels": {"list_components": {"_type": "local"}}}
    registry_config = GlobalSettings.get().model_validate(config_dict)
    local_registry_config = registry_config.channels.get("list_components", None)
    if (local_registry_config is None):
        logger.error("Channel runtime instance not found.")

    asyncio.run(
        search_artifacts(local_registry_config,
                         query=query,
                         num_results=num_results,
                         query_fields=fields,
                         component_types=component_types,
                         visualize=True,
                         save_path=output_path))
