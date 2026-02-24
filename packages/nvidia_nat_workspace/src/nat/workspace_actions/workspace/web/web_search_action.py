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
"""Web search action implementation using SearXNG with DuckDuckGo fallback."""

from __future__ import annotations

import html
import json
import logging
import os
import re
import typing
from urllib.parse import urlencode

from nat.cli.register_workflow import register_workspace_action
from nat.data_models.workspace import ActionContext
from nat.workspace.types import TypeSchema, WorkspaceAction

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 10
MAX_RESULTS = 20
DEFAULT_SEARXNG_URL = "https://searx.be"
SEARCH_TIMEOUT_SECONDS = 15

_VALID_TIME_RANGES = {"day", "week", "month", "year", "all"}
_VALID_ENGINES = {"searxng", "duckduckgo", "auto"}


def _get_http_client() -> typing.Any:
    """Lazily import and return an httpx.AsyncClient."""
    import httpx  # noqa: F811

    return httpx.AsyncClient(
        timeout=SEARCH_TIMEOUT_SECONDS,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; NATBot/1.0)"},
    )


def _strip_html_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    stripped = re.sub(r"<[^>]*>", "", text)
    return html.unescape(stripped).strip()


async def _search_searxng(
    query: str,
    *,
    max_results: int,
    searxng_url: str,
    time_range: str,
    safe_search: bool,
) -> list[dict[str, str]]:
    """Search via a SearXNG instance (JSON API)."""
    params: dict[str, str] = {
        "q": query,
        "format": "json",
        "pageno": "1",
    }
    if time_range != "all":
        params["time_range"] = time_range
    if safe_search:
        params["safesearch"] = "1"

    search_url = f"{searxng_url}/search?{urlencode(params)}"
    client = _get_http_client()
    try:
        async with client:
            response = await client.get(search_url)
            response.raise_for_status()
            if "application/json" not in response.headers.get("content-type", ""):
                logger.warning("SearXNG returned non-JSON response for %s", search_url)
                return []
            data = response.json()
    except Exception as exc:
        logger.warning("SearXNG search failed: %s", exc)
        return []

    raw_results = data.get("results", [])
    if not isinstance(raw_results, list):
        return []

    results: list[dict[str, str]] = []
    for item in raw_results[:max_results]:
        url = item.get("url", "")
        if not url:
            continue
        results.append({
            "title": item.get("title", "Untitled"),
            "url": url,
            "snippet": item.get("content", ""),
            "source": "searxng",
        })
    return results


async def _search_duckduckgo(
    query: str,
    *,
    max_results: int,
    safe_search: bool,
) -> list[dict[str, str]]:
    """Search via DuckDuckGo Lite (HTML scraping fallback)."""
    params: dict[str, str] = {"q": query, "kl": "us-en"}
    if safe_search:
        params["kp"] = "1"

    client = _get_http_client()
    try:
        async with client:
            response = await client.get(
                f"https://lite.duckduckgo.com/lite/?{urlencode(params)}"
            )
            response.raise_for_status()
            page_html = response.text
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return []

    # Extract result links with rel="nofollow"
    link_pattern = re.compile(
        r'<a[^>]*rel=["\']?nofollow["\']?[^>]*href=["\']([^"\']+)["\'][^>]*>'
        r"([\s\S]*?)</a>",
        re.IGNORECASE,
    )
    snippet_pattern = re.compile(
        r'<td[^>]*class=["\'][^"\']*result-snippet[^"\']*["\'][^>]*>'
        r"([\s\S]*?)</td>",
        re.IGNORECASE,
    )

    urls: list[str] = []
    titles: list[str] = []
    for match in link_pattern.finditer(page_html):
        raw_url = match.group(1)
        title_html = match.group(2)
        if not raw_url or "duckduckgo.com" in raw_url or raw_url.startswith(("/", "#")):
            continue
        # Decode DDG redirect URLs
        uddg_match = re.search(r"uddg=([^&]+)", raw_url)
        if uddg_match:
            from urllib.parse import unquote

            raw_url = unquote(uddg_match.group(1))
        urls.append(raw_url)
        titles.append(_strip_html_tags(title_html) or "Untitled")

    snippets: list[str] = []
    for match in snippet_pattern.finditer(page_html):
        snippets.append(_strip_html_tags(match.group(1)))

    results: list[dict[str, str]] = []
    for i in range(min(len(urls), max_results)):
        results.append({
            "title": titles[i] if i < len(titles) else "Untitled",
            "url": urls[i],
            "snippet": snippets[i] if i < len(snippets) else "",
            "source": "duckduckgo",
        })
    return results


@register_workspace_action
class WebSearchAction(WorkspaceAction):
    """Search the web and return results with titles, URLs, and snippets."""

    name = "web_search"
    description = (
        "Search the web and return a list of results with titles, URLs, and snippets.\n"
        "\n"
        "Use this tool to:\n"
        "- Find current information not in your training data\n"
        "- Research documentation, tutorials, or guides\n"
        "- Look up error messages or technical problems\n"
        "- Find recent news or updates about topics\n"
        "\n"
        "Options:\n"
        "- max_results: Number of results (1-20, default: 10)\n"
        "- time_range: Filter by recency - 'day', 'week', 'month', 'year', or 'all' (default)\n"
        "- safe_search: Enable safe search (default: true)\n"
        "- engine: 'searxng', 'duckduckgo', or 'auto' (default, tries SearXNG first)\n"
        "\n"
        "Tips for effective searches:\n"
        "- Be specific: 'TypeScript generics tutorial' vs just 'TypeScript'\n"
        "- Include version/year for tech topics: 'Node.js 20 features'\n"
        "- Use quotes for exact phrases: '\"cannot read property\" undefined'\n"
        "\n"
        "After searching, use web_fetch to read the full content of promising results."
    )
    parameters = [
        TypeSchema(type="string", description="query: Search query string (required)."),
        TypeSchema(type="number", description="max_results: Number of results to return (1-20, default 10)."),
        TypeSchema(type="string", description="engine: Search engine - 'searxng', 'duckduckgo', or 'auto' (default)."),
        TypeSchema(type="string", description="searxng_url: Custom SearXNG instance URL."),
        TypeSchema(type="string", description="time_range: Filter results by recency - 'day', 'week', 'month', 'year', or 'all' (default)."),
        TypeSchema(type="boolean", description="safe_search: Enable safe search filtering. Default true."),
    ]
    result = TypeSchema(
        type="object",
        description="Result with results (list), total_results, query, engine, and summary.",
    )

    def __init__(self) -> None:
        pass

    async def execute(self, context: ActionContext, args: dict[str, typing.Any]) -> dict[str, typing.Any]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return {
                "is_error": True,
                "error": "A non-empty 'query' string is required.",
                "results": [],
                "total_results": 0,
                "query": str(query) if query else "",
                "engine": "none",
                "summary": "",
            }

        max_results = min(int(args.get("max_results", DEFAULT_MAX_RESULTS)), MAX_RESULTS)
        engine = str(args.get("engine", "auto")).lower()
        if engine not in _VALID_ENGINES:
            engine = "auto"

        time_range = str(args.get("time_range", "all")).lower()
        if time_range not in _VALID_TIME_RANGES:
            time_range = "all"

        safe_search = bool(args.get("safe_search", True))
        searxng_url = str(args.get("searxng_url", "")).strip() or os.environ.get(
            "SEARXNG_URL", DEFAULT_SEARXNG_URL
        )

        results: list[dict[str, str]] = []
        used_engine = "none"

        try:
            # Try SearXNG first
            if engine in ("searxng", "auto"):
                results = await _search_searxng(
                    query,
                    max_results=max_results,
                    searxng_url=searxng_url,
                    time_range=time_range,
                    safe_search=safe_search,
                )
                if results:
                    used_engine = "searxng"

            # Fallback to DuckDuckGo
            if not results and engine in ("duckduckgo", "auto"):
                results = await _search_duckduckgo(
                    query,
                    max_results=max_results,
                    safe_search=safe_search,
                )
                if results:
                    used_engine = "duckduckgo"

            # Build summary
            if not results:
                summary = (
                    f'No search results for "{query}" - all search engines failed or '
                    "returned no results. Check network connectivity or try again later."
                )
            else:
                plural = "s" if len(results) > 1 else ""
                summary = f'Found {len(results)} result{plural} for "{query}" using {used_engine}'

            return {
                "results": results,
                "total_results": len(results),
                "query": query,
                "engine": used_engine,
                "summary": summary,
            }
        except Exception as exc:
            return {
                "is_error": True,
                "error": str(exc),
                "results": [],
                "total_results": 0,
                "query": query,
                "engine": used_engine,
                "summary": "",
            }
