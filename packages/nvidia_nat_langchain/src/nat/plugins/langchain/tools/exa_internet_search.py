# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Literal

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.common import SerializableSecretStr
from nat.data_models.common import get_secret_value
from nat.data_models.function import FunctionBaseConfig


# Internet Search tool
class ExaInternetSearchToolConfig(FunctionBaseConfig, name="exa_internet_search"):
    """
    Tool that retrieves relevant contexts from web search (using Exa) for the given question.
    Requires an EXA_API_KEY.
    """
    max_results: int = Field(default=3, ge=1, description="Maximum number of search results to return.")
    api_key: SerializableSecretStr = Field(default_factory=lambda: SerializableSecretStr(""),
                                           description="The API key for the Exa service.")
    max_retries: int = Field(default=3, ge=1, description="Maximum number of retries for the search request")
    search_type: Literal["auto", "neural", "keyword"] = Field(
        default="auto",
        description="Type of search to perform - 'neural', 'keyword', or 'auto'")
    livecrawl: Literal["always", "fallback", "never"] = Field(
        default="fallback",
        description="Livecrawl behavior - 'always', 'fallback', or 'never'")


@register_function(config_type=ExaInternetSearchToolConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def exa_internet_search(tool_config: ExaInternetSearchToolConfig, builder: Builder):
    import os

    from exa_py import AsyncExa

    api_key = get_secret_value(tool_config.api_key) if tool_config.api_key else ""
    resolved_api_key = api_key or os.environ.get("EXA_API_KEY", "")
    exa_client = AsyncExa(api_key=resolved_api_key)

    async def _exa_internet_search(question: str) -> str:
        """This tool retrieves relevant contexts from web search (using Exa) for the given question.

        Args:
            question (str): The question to be answered. Will be truncated to 2000 characters if longer.

        Returns:
            str: The web search results.
        """
        if not resolved_api_key:
            return "Web search is unavailable: `EXA_API_KEY` is not configured."

        # Exa API supports longer queries than Tavily but truncate at a reasonable limit
        if len(question) > 2000:
            question = question[:1997] + "..."

        for attempt in range(tool_config.max_retries):
            try:
                search_response = await exa_client.search_and_contents(
                    question,
                    num_results=tool_config.max_results,
                    type=tool_config.search_type,
                    livecrawl=tool_config.livecrawl,
                    text={"max_characters": 3000},
                )
                if not search_response.results:
                    return f"No web search results found for: {question}"
                # Format - SearchResponse.results contains Result objects with .url and .text attrs
                web_search_results = "\n\n---\n\n".join([
                    f'<Document href="{doc.url}"/>\n{doc.text}\n</Document>'
                    for doc in search_response.results if doc.text
                ])
                return web_search_results or f"No web search results found for: {question}"
            except Exception:
                # Return a graceful message instead of raising, so the agent can
                # continue reasoning without web search rather than failing entirely.
                if attempt == tool_config.max_retries - 1:
                    return f"Web search failed after {tool_config.max_retries} attempts for: {question}"
                await asyncio.sleep(2**attempt)
        return f"Web search failed after {tool_config.max_retries} attempts for: {question}"

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _exa_internet_search,
        description=_exa_internet_search.__doc__,
    )
