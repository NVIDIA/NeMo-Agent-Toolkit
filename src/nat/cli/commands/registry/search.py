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

from nat.data_models.component import ComponentEnum
from nat.data_models.registry_handler import RegistryHandlerBaseConfig
from nat.registry_handlers.schemas.search import SearchFields
from nat.registry_handlers.schemas.status import StatusEnum
from nat.utils.data_models.schema_validator import validate_yaml

logger = logging.getLogger(__name__)


async def search_artifacts(  # pylint: disable=R0917
        registry_handler_config: RegistryHandlerBaseConfig,
        query: str,
        search_fields: list[SearchFields],
        visualize: bool,
        component_types: list[ComponentEnum],
        save_path: str | None = None,
        n_results: int = 10) -> None:

    from nat.cli.type_registry import GlobalTypeRegistry
    from nat.registry_handlers.schemas.search import SearchQuery

    registry = GlobalTypeRegistry.get()

    async with AsyncExitStack() as stack:

        registry_handler_info = registry.get_registry_handler(type(registry_handler_config))
        registry_handler = await stack.enter_async_context(registry_handler_info.build_fn(registry_handler_config))

        if (len(component_types) == 0):
            component_types = [t.value for t in ComponentEnum]

        query = SearchQuery(query=query, fields=search_fields, top_k=n_results, component_types=component_types)

        search_response = await stack.enter_async_context(registry_handler.search(query=query))

        if (search_response.status.status == StatusEnum.SUCCESS):
            if (visualize):
                registry_handler.visualize_search_results(search_response=search_response)
            if (save_path is not None):
                registry_handler.save_search_results(search_response=search_response, save_path=save_path)


@click.group(name=__name__, invoke_without_command=True, help="Search for NAT artifacts from remote registry.")
@click.option(
    "--config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    callback=validate_yaml,
    required=False,
    help=("A JSON/YAML file that sets the parameters for the workflow."),
)
@click.option(
    "-c",
    "--channel",
    type=str,
    required=True,
    help=("The remote registry channel to use when pulling the NAT artifact."),
)
@click.option(
    "-o",
    "--output_path",
    type=str,
    required=False,
    help=("Path to save search results."),
)
@click.option(
    "-f",
    "--fields",
    multiple=True,
    type=click.Choice([e.value for e in SearchFields], case_sensitive=False),
    required=False,
    help=("The fields to include in the search."),
)
@click.option(
    "-q",
    "--query",
    type=str,
    required=True,
    help=("The query string."),
)
@click.option(
    "-n",
    "--n_results",
    type=int,
    required=False,
    default=10,
    help=("Number of search results to return."),
)
@click.option(
    "-t",
    "--types",
    "component_types",
    multiple=True,
    type=click.Choice([e.value for e in ComponentEnum], case_sensitive=False),
    required=False,
    help=("The component types to include in search."),
)
def search(  # pylint: disable=R0917
        config_file: str,
        channel: str,
        fields: list[str],
        query: str,
        component_types: list[ComponentEnum],
        n_results: int,
        output_path: str) -> None:
    """
    Search for NAT artifacts with the specified configuration.
    """

    from nat.settings.global_settings import GlobalSettings

    settings = GlobalSettings().get()

    if (config_file is not None):
        settings = settings.override_settings(config_file)

    try:
        search_channel_config = settings.channels.get(channel)

        if (search_channel_config is None):
            logger.error("Search channel '%s' has not been configured.", channel)
            return
    except Exception as e:
        logger.exception("Error loading user settings: %s", e, exc_info=True)
        return

    asyncio.run(
        search_artifacts(registry_handler_config=search_channel_config,
                         query=query,
                         component_types=component_types,
                         search_fields=fields,
                         visualize=True,
                         save_path=output_path,
                         n_results=n_results))
