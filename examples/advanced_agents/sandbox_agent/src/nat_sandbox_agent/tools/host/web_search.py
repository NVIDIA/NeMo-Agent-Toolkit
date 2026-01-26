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

"""Host-side web search tool using Tavily API.

This tool runs on the host machine (not in the sandbox) for:
- Better security (API key not exposed to sandbox)
- Lower latency (no Docker exec overhead)
- Higher reliability (no sandbox network issues)
"""

import logging
import os
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    """Input schema for web search."""

    query: str = Field(description="Search query to execute.")
    num_results: int = Field(
        default=5,
        description="Number of search results to return.",
        ge=1,
        le=10,
    )


class HostWebSearchTool:
    """Web search tool that runs on host (not in sandbox).

    Uses Tavily API for high-quality search results optimized for AI agents.
    """

    def __init__(self, api_key: str | None = None):
        """Initialize the web search tool.

        Args:
            api_key: Tavily API key. If None, uses TAVILY_API_KEY env var.
        """
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazily initialize async Tavily client."""
        if self._client is None:
            if not self._api_key:
                raise ValueError("TAVILY_API_KEY not set")
            from tavily import AsyncTavilyClient

            self._client = AsyncTavilyClient(api_key=self._api_key)
        return self._client

    async def search(self, query: str, num_results: int = 5) -> dict[str, Any]:
        """Execute web search.

        Args:
            query: Search query.
            num_results: Number of results to return (max 10).

        Returns:
            Dict with status and search results.
        """
        logger.info(f"Web search: query_len={len(query)}")

        try:
            client = self._get_client()
            response = await client.search(
                query=query,
                max_results=min(num_results, 10),
                include_answer=True,
            )

            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                    "score": r.get("score", 0),
                }
                for r in response.get("results", [])
            ]

            logger.info(f"Web search returned {len(results)} results")

            return {
                "status": "success",
                "results": results,
                "answer": response.get("answer"),
            }

        except Exception as e:
            logger.exception("Web search error")
            return {
                "status": "error",
                "error": str(e),
            }


def create_web_search_tool(api_key: str | None = None) -> StructuredTool:
    """Create the web search tool.

    Args:
        api_key: Tavily API key. If None, uses TAVILY_API_KEY env var.

    Returns:
        LangChain StructuredTool instance.
    """
    tool = HostWebSearchTool(api_key=api_key)

    return StructuredTool.from_function(
        coroutine=tool.search,
        name="web_search",
        description=(
            "Search the web using Tavily. Returns titles, URLs, and snippets "
            "for the search results. Use this to find information, research topics, "
            "or locate relevant URLs."
        ),
        args_schema=WebSearchInput,
    )
