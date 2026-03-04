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

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nat.data_models.workspace import ActionContext
from nat.workspace_actions.workspace.web.web_search_action import WebSearchAction
from nat.workspace_actions.workspace.web.web_fetch_action import WebFetchAction


@pytest.fixture(name="fixture_action_context")
def fixture_action_context(tmp_path: Path) -> ActionContext:
    return ActionContext(session_id="test-session-web", root_path=tmp_path.resolve())


# ---------------------------------------------------------------------------
# WebSearchAction tests
# ---------------------------------------------------------------------------


@pytest.fixture(name="fixture_search_action")
def fixture_search_action() -> WebSearchAction:
    return WebSearchAction()


async def test_web_search_requires_query(
    fixture_action_context: ActionContext,
    fixture_search_action: WebSearchAction,
) -> None:
    """Empty or missing query returns an error."""
    result = await fixture_search_action.execute(fixture_action_context, {})
    assert result["is_error"] is True
    assert "query" in result["error"].lower()

    result2 = await fixture_search_action.execute(fixture_action_context, {"query": ""})
    assert result2["is_error"] is True


async def test_web_search_searxng_success(
    fixture_action_context: ActionContext,
    fixture_search_action: WebSearchAction,
) -> None:
    """SearXNG returning valid JSON results produces a successful response."""
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"content-type": "application/json"}
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "results": [
            {"title": "Result 1", "url": "https://example.com/1", "content": "Snippet 1"},
            {"title": "Result 2", "url": "https://example.com/2", "content": "Snippet 2"},
        ],
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=fake_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "nat.workspace_actions.workspace.web.web_search_action._get_http_client",
        return_value=mock_client,
    ):
        result = await fixture_search_action.execute(
            fixture_action_context,
            {"query": "test query", "engine": "searxng"},
        )

    assert "is_error" not in result
    assert result["total_results"] == 2
    assert result["engine"] == "searxng"
    assert result["query"] == "test query"
    assert len(result["results"]) == 2
    assert result["results"][0]["title"] == "Result 1"
    assert result["results"][0]["url"] == "https://example.com/1"


async def test_web_search_fallback_to_duckduckgo(
    fixture_action_context: ActionContext,
    fixture_search_action: WebSearchAction,
) -> None:
    """When SearXNG returns empty results, DuckDuckGo fallback is attempted."""
    # SearXNG returns empty
    searxng_response = MagicMock()
    searxng_response.status_code = 200
    searxng_response.headers = {"content-type": "application/json"}
    searxng_response.raise_for_status = MagicMock()
    searxng_response.json.return_value = {"results": []}

    # DDG returns HTML with results
    ddg_html = (
        '<table>'
        '<a rel="nofollow" href="https://example.com/ddg">DDG Result</a>'
        '<td class="result-snippet">DDG snippet</td>'
        '</table>'
    )
    ddg_response = MagicMock()
    ddg_response.status_code = 200
    ddg_response.headers = {"content-type": "text/html"}
    ddg_response.raise_for_status = MagicMock()
    ddg_response.text = ddg_html

    call_count = 0

    async def fake_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        if "searx" in url.lower() or "searxng" in url.lower():
            return searxng_response
        return ddg_response

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=fake_get)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "nat.workspace_actions.workspace.web.web_search_action._get_http_client",
        return_value=mock_client,
    ):
        result = await fixture_search_action.execute(
            fixture_action_context,
            {"query": "test fallback", "engine": "auto"},
        )

    assert "is_error" not in result
    assert result["engine"] == "duckduckgo"
    assert result["total_results"] >= 1
    assert result["results"][0]["url"] == "https://example.com/ddg"


async def test_web_search_no_results(
    fixture_action_context: ActionContext,
    fixture_search_action: WebSearchAction,
) -> None:
    """When all engines return empty, summary reflects it."""
    empty_response = MagicMock()
    empty_response.status_code = 200
    empty_response.headers = {"content-type": "application/json"}
    empty_response.raise_for_status = MagicMock()
    empty_response.json.return_value = {"results": []}

    ddg_response = MagicMock()
    ddg_response.status_code = 200
    ddg_response.headers = {"content-type": "text/html"}
    ddg_response.raise_for_status = MagicMock()
    ddg_response.text = "<html></html>"

    async def fake_get(url, **kwargs):
        if "duckduckgo" in url:
            return ddg_response
        return empty_response

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=fake_get)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "nat.workspace_actions.workspace.web.web_search_action._get_http_client",
        return_value=mock_client,
    ):
        result = await fixture_search_action.execute(
            fixture_action_context,
            {"query": "obscure nonexistent query xyz"},
        )

    assert result["total_results"] == 0
    assert "no search results" in result["summary"].lower()


async def test_web_search_max_results_capped(
    fixture_action_context: ActionContext,
    fixture_search_action: WebSearchAction,
) -> None:
    """max_results is capped at 20."""
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"content-type": "application/json"}
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "results": [
            {"title": f"R{i}", "url": f"https://example.com/{i}", "content": f"S{i}"}
            for i in range(30)
        ],
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=fake_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "nat.workspace_actions.workspace.web.web_search_action._get_http_client",
        return_value=mock_client,
    ):
        result = await fixture_search_action.execute(
            fixture_action_context,
            {"query": "many results", "max_results": 50, "engine": "searxng"},
        )

    assert "is_error" not in result
    assert result["total_results"] <= 20


# ---------------------------------------------------------------------------
# WebFetchAction tests
# ---------------------------------------------------------------------------


@pytest.fixture(name="fixture_fetch_action")
def fixture_fetch_action() -> WebFetchAction:
    return WebFetchAction()


async def test_web_fetch_requires_url(
    fixture_action_context: ActionContext,
    fixture_fetch_action: WebFetchAction,
) -> None:
    """Empty or missing url returns an error."""
    result = await fixture_fetch_action.execute(fixture_action_context, {})
    assert result["is_error"] is True
    assert "url" in result["error"].lower()


async def test_web_fetch_rejects_private_hosts(
    fixture_action_context: ActionContext,
    fixture_fetch_action: WebFetchAction,
) -> None:
    """Localhost and private IPs are blocked."""
    for blocked_url in [
        "http://localhost/path",
        "http://127.0.0.1/path",
        "http://192.168.1.1/path",
        "http://10.0.0.1/path",
    ]:
        result = await fixture_fetch_action.execute(
            fixture_action_context, {"url": blocked_url}
        )
        assert result["is_error"] is True
        assert "private" in result["error"].lower() or "localhost" in result["error"].lower()


async def test_web_fetch_rejects_unsupported_protocol(
    fixture_action_context: ActionContext,
    fixture_fetch_action: WebFetchAction,
) -> None:
    """Non-http(s) protocols are rejected."""
    result = await fixture_fetch_action.execute(
        fixture_action_context, {"url": "ftp://example.com/file"}
    )
    assert result["is_error"] is True
    assert "unsupported protocol" in result["error"].lower()


async def test_web_fetch_html_conversion(
    fixture_action_context: ActionContext,
    fixture_fetch_action: WebFetchAction,
) -> None:
    """HTML content is converted to markdown-like text."""
    html_content = (
        "<html><head><title>Test Page</title>"
        '<meta name="description" content="A test page">'
        "</head><body>"
        "<h1>Hello World</h1>"
        "<p>This is a <strong>test</strong> paragraph.</p>"
        '<a href="https://example.com">Link</a>'
        "</body></html>"
    )

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"content-type": "text/html; charset=utf-8"}
    fake_response.url = "https://example.com/page"
    fake_response.text = html_content
    fake_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=fake_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("nat.workspace_actions.workspace.web.web_fetch_action.httpx", create=True) as mock_httpx:
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        # Patch at module level since httpx is imported inside execute
        with patch(
            "nat.workspace_actions.workspace.web.web_fetch_action.WebFetchAction.execute",
            wraps=fixture_fetch_action.execute,
        ):
            # We need to mock httpx import inside execute
            pass

    # Use a direct approach - mock the httpx module
    import types
    mock_httpx_module = types.ModuleType("httpx")
    mock_httpx_module.AsyncClient = MagicMock(return_value=mock_client)

    with patch.dict("sys.modules", {"httpx": mock_httpx_module}):
        result = await fixture_fetch_action.execute(
            fixture_action_context,
            {"url": "https://example.com/page"},
        )

    assert "is_error" not in result
    assert result["title"] == "Test Page"
    assert result["description"] == "A test page"
    assert "Hello World" in result["content"]
    # The lightweight HTML→markdown converter wraps <strong> content in **
    assert "test**" in result["content"]
    assert "[Link](https://example.com)" in result["content"]
    assert result["content_type"] == "text/html; charset=utf-8"
    assert isinstance(result["length"], int)


async def test_web_fetch_raw_mode(
    fixture_action_context: ActionContext,
    fixture_fetch_action: WebFetchAction,
) -> None:
    """Raw mode returns unprocessed content."""
    raw_html = "<html><body><p>raw</p></body></html>"

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"content-type": "text/html"}
    fake_response.url = "https://example.com/raw"
    fake_response.text = raw_html
    fake_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=fake_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    import types
    mock_httpx_module = types.ModuleType("httpx")
    mock_httpx_module.AsyncClient = MagicMock(return_value=mock_client)

    with patch.dict("sys.modules", {"httpx": mock_httpx_module}):
        result = await fixture_fetch_action.execute(
            fixture_action_context,
            {"url": "https://example.com/raw", "raw": True},
        )

    assert "is_error" not in result
    # Raw mode should preserve HTML tags
    assert "<p>raw</p>" in result["content"]


async def test_web_fetch_truncation(
    fixture_action_context: ActionContext,
    fixture_fetch_action: WebFetchAction,
) -> None:
    """Content exceeding max_length is truncated."""
    long_content = "A" * 1000

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"content-type": "text/plain"}
    fake_response.url = "https://example.com/long"
    fake_response.text = long_content
    fake_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=fake_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    import types
    mock_httpx_module = types.ModuleType("httpx")
    mock_httpx_module.AsyncClient = MagicMock(return_value=mock_client)

    with patch.dict("sys.modules", {"httpx": mock_httpx_module}):
        result = await fixture_fetch_action.execute(
            fixture_action_context,
            {"url": "https://example.com/long", "max_length": 200},
        )

    assert "is_error" not in result
    assert "[Content truncated]" in result["content"]
    assert result["length"] <= 200 + len("\n\n[Content truncated]")
