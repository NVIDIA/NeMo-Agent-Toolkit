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

import logging
import os

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class SerpApiToolConfig(FunctionBaseConfig, name="serp_api_tool"):
    """
    Tool that retrieves search results from the web using SerpAPI.
    Requires a SERP_API_KEY.
    """
    api_key: str | None = Field(default=None, description="The API key for the SerpAPI service.")
    max_results: int = Field(default=5, description="The maximum number of results to return.")


@register_function(config_type=SerpApiToolConfig, framework_wrappers=[LLMFrameworkEnum.AGNO])
async def serp_api_tool(tool_config: SerpApiToolConfig, builder: Builder):
    """Create a SerpAPI search tool for use with Agno.

    This creates a search function that uses SerpAPI to search the web.

    Args:
        tool_config (SerpApiToolConfig): Configuration for the SerpAPI tool.
        builder (Builder): The AIQ Toolkit builder instance.

    Returns:
        FunctionInfo: A FunctionInfo object wrapping the SerpAPI search functionality.
    """
    import json

    from agno.tools.serpapi import SerpApiTools

    if (not tool_config.api_key):
        tool_config.api_key = os.getenv("SERP_API_KEY")

    if not tool_config.api_key:
        raise ValueError(
            "API token must be provided in the configuration or in the environment variable `SERP_API_KEY`")

    # Create the SerpAPI tools instance
    search_tool = SerpApiTools(api_key=tool_config.api_key)

    # Simple search function with a single string parameter
    async def _serp_api_search(query: str) -> str:
        """
        Search the web using SerpAPI.

        Args:
            query (str): The search query to perform. If empty, returns initialization message.

        Returns:
            str: Formatted search results or initialization message.
        """

        if not query or query.strip() == "":
            exception_msg = "Search query cannot be empty. Please provide a specific search term to continue."
            logger.warning(exception_msg)
            return exception_msg

        logger.info("Searching SerpAPI with query: '%s', max_results: %s", query, tool_config.max_results)

        try:
            # Perform the search
            raw_all_results: str = search_tool.search_google(query=query, num_results=tool_config.max_results)
            all_results: dict = json.loads(raw_all_results)
            search_results = all_results.get('search_results', [])

            logger.info("SerpAPI returned %s results", len(search_results))

            # Format the results as a string
            formatted_results = []
            for result in search_results:
                title = result.get('title', 'No Title')
                link = result.get('link', 'No Link')
                snippet = result.get('snippet', 'No Snippet')

                formatted_result = f'<Document href="{link}"/>\n'
                formatted_result += f'# {title}\n\n'
                formatted_result += f'{snippet}\n'
                formatted_result += '</Document>'
                formatted_results.append(formatted_result)

            return "\n\n---\n\n".join(formatted_results)
        except Exception as e:
            logger.exception("Error searching with SerpAPI: %s", e)
            return f"Error performing search: {str(e)}"

    fn_info = FunctionInfo.from_fn(
        _serp_api_search,
        description="""This tool searches the web using SerpAPI and returns relevant results for the given query.""")

    yield fn_info
