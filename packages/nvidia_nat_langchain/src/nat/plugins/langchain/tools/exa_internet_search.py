# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from __future__ import annotations

import asyncio
import logging
from typing import Literal

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.common import SerializableSecretStr
from nat.data_models.common import get_secret_value
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


# Internet Search tool
class ExaInternetSearchToolConfig(FunctionBaseConfig, name="exa_internet_search"):
    """
    Tool that retrieves relevant contexts from web search (using Exa) for the given question.
    Requires an EXA_API_KEY.
    """
    max_results: int = Field(default=5, ge=1, description="Maximum number of search results to return.")
    api_key: SerializableSecretStr = Field(default_factory=lambda: SerializableSecretStr(""),
                                           description="The API key for the Exa service.")
    max_retries: int = Field(default=3, ge=1, description="Maximum number of retries for the search request")
    search_type: Literal["auto", "fast", "deep", "neural", "instant"] = Field(
        default="auto", description="Exa search type - 'auto', 'fast', 'deep', 'neural', or 'instant'")
    livecrawl: Literal["always", "fallback",
                       "never"] = Field(default="fallback",
                                        description="Livecrawl behavior - 'always', 'fallback', or 'never'")
    max_query_length: int = Field(
        default=2000,
        ge=1,
        description="Maximum query length in characters. Queries exceeding this limit will be truncated.")
    highlights: bool = Field(default=True, description="Whether to include highlights in search results.")
    max_content_length: int | None = Field(
        default=10000,
        ge=1,
        description="Maximum characters of text content per result. Set to None to disable text content.")


@register_function(config_type=ExaInternetSearchToolConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def exa_internet_search(tool_config: ExaInternetSearchToolConfig, builder: Builder):
    import os

    from langchain_exa import ExaSearchResults

    api_key = get_secret_value(tool_config.api_key) if tool_config.api_key else ""
    resolved_api_key = api_key or os.environ.get("EXA_API_KEY", "")

    async def _exa_internet_search(question: str) -> str:
        """This tool retrieves relevant contexts from web search (using Exa) for the given question.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The web search results.
        """
        if not resolved_api_key:
            return "Web search is unavailable: `EXA_API_KEY` is not configured."

        exa_search = ExaSearchResults(exa_api_key=resolved_api_key)

        # Truncate long queries to the configured limit
        max_len = tool_config.max_query_length
        if len(question) > max_len:
            logger.warning("Exa query truncated from %d to %d characters", len(question), max_len)
            question = question[:max_len - 3] + "..."

        for attempt in range(tool_config.max_retries):
            try:
                search_response = await exa_search._arun(
                    question,
                    num_results=tool_config.max_results,
                    type=tool_config.search_type,
                    livecrawl=tool_config.livecrawl,
                    text_contents_options=({
                        "max_characters": tool_config.max_content_length
                    } if tool_config.max_content_length else None),
                    highlights=tool_config.highlights or None,
                )
                # On error, ExaSearchResults may return a string error message
                if isinstance(search_response, str):
                    return f"No web search results found for: {question}"
                if not search_response.results:
                    return f"No web search results found for: {question}"
                # Format - SearchResponse.results contains Result objects with .url and .text attrs
                web_search_results = "\n\n---\n\n".join([
                    f'<Document href="{doc.url}"/>\n{doc.text}\n</Document>' for doc in search_response.results
                    if doc.text
                ])
                return web_search_results or f"No web search results found for: {question}"
            except Exception:
                # Return a graceful message instead of raising, so the agent can
                # continue reasoning without web search rather than failing entirely.
                logger.exception("Exa search attempt %d of %d failed", attempt + 1, tool_config.max_retries)
                if attempt == tool_config.max_retries - 1:
                    return f"Web search failed after {tool_config.max_retries} attempts for: {question}"
                await asyncio.sleep(2**attempt)
        return f"Web search failed after {tool_config.max_retries} attempts for: {question}"

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _exa_internet_search,
        description=_exa_internet_search.__doc__,
    )
