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

import httpx
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.common import SerializableSecretStr
from nat.data_models.common import get_secret_value
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

PERPLEXITY_SEARCH_URL = "https://api.perplexity.ai/search"


# Internet Search tool
class PerplexityInternetSearchToolConfig(FunctionBaseConfig, name="perplexity_internet_search"):
    """
    Tool that retrieves relevant contexts from web search (using Perplexity) for the given question.
    Requires a PERPLEXITY_API_KEY.
    """

    max_results: int = Field(default=5, ge=1, le=20, description="Maximum number of search results to return.")
    api_key: SerializableSecretStr = Field(
        default_factory=lambda: SerializableSecretStr(""), description="The API key for the Perplexity service."
    )
    max_retries: int = Field(default=3, ge=1, description="Maximum number of retries for the search request")
    max_query_length: int = Field(
        default=2000,
        ge=1,
        description="Maximum query length in characters. Queries exceeding this limit will be truncated.",
    )
    search_recency_filter: Literal["hour", "day", "week", "month", "year"] | None = Field(
        default=None, description="Filter search results by recency - 'hour', 'day', 'week', 'month', or 'year'."
    )
    country: str | None = Field(
        default=None, description="Country to filter search results by ISO 3166-1 alpha-2 code."
    )
    max_tokens_per_page: int = Field(
        default=4096, ge=1, description="Maximum number of tokens to retrieve per search result page."
    )


def _get_integration_header() -> str:
    from importlib import metadata

    try:
        package_version = metadata.version("nvidia-nat-langchain")
    except metadata.PackageNotFoundError:
        package_version = "unknown"
    return f"nemo-agent-toolkit/{package_version}"


@register_function(config_type=PerplexityInternetSearchToolConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def perplexity_internet_search(tool_config: PerplexityInternetSearchToolConfig, builder: Builder):
    import os

    api_key = get_secret_value(tool_config.api_key) if tool_config.api_key else ""
    resolved_api_key = api_key or os.environ.get("PERPLEXITY_API_KEY", "")

    async def _perplexity_internet_search(question: str) -> str:
        """This tool retrieves relevant contexts from web search (using Perplexity) for the given question.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The web search results.
        """
        if not resolved_api_key:
            return "Web search is unavailable: 'PERPLEXITY_API_KEY' is not configured."

        # Truncate long queries to the configured limit
        max_len = tool_config.max_query_length
        if len(question) > max_len:
            logger.warning("Perplexity query truncated from %d to %d characters", len(question), max_len)
            question = question[: max_len - 3] + "..." if max_len > 3 else question[:max_len]

        request_body = {
            "query": question,
            "max_results": tool_config.max_results,
            "max_tokens_per_page": tool_config.max_tokens_per_page,
        }
        if tool_config.search_recency_filter is not None:
            request_body["search_recency_filter"] = tool_config.search_recency_filter
        if tool_config.country is not None:
            request_body["country"] = tool_config.country

        headers = {
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
            "X-Pplx-Integration": _get_integration_header(),
        }

        async with httpx.AsyncClient() as client:
            for attempt in range(tool_config.max_retries):
                try:
                    response = await client.post(PERPLEXITY_SEARCH_URL, headers=headers, json=request_body)
                    response.raise_for_status()
                    search_response = response.json()
                    results = search_response.get("results") if isinstance(search_response, dict) else None
                    if not results:
                        return f"No web search results found for: {question}"

                    web_search_results = "\n\n---\n\n".join(
                        [
                            f'<Document href="{doc.get("url", "")}"/>\n{doc.get("snippet", "")}\n</Document>'
                            for doc in results
                        ]
                    )
                    return web_search_results or f"No web search results found for: {question}"
                except Exception:
                    # Return a graceful message instead of raising, so the agent can
                    # continue reasoning without web search rather than failing entirely.
                    logger.exception("Perplexity search attempt %d of %d failed", attempt + 1, tool_config.max_retries)
                    if attempt == tool_config.max_retries - 1:
                        return f"Web search failed after {tool_config.max_retries} attempts for: {question}"
                    await asyncio.sleep(2**attempt)
        return f"Web search failed after {tool_config.max_retries} attempts for: {question}"

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _perplexity_internet_search,
        description=_perplexity_internet_search.__doc__,
    )
