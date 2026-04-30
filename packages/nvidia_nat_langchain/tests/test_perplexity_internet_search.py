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

import os
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import SecretStr
from pydantic import ValidationError

# -- Config validation tests --


@pytest.mark.parametrize(
    "constructor_args",
    [{}, {"api_key": ""}, {"api_key": "my_api_key"}],
    ids=["default", "empty_api_key", "provided_api_key"],
)
def test_api_key_is_secret_str(constructor_args: dict):
    from nat.plugins.langchain.tools.perplexity_internet_search import PerplexityInternetSearchToolConfig

    expected_api_key = constructor_args.get("api_key", "")

    config = PerplexityInternetSearchToolConfig(**constructor_args)
    assert isinstance(config.api_key, SecretStr)

    api_key = config.api_key.get_secret_value()
    assert api_key == expected_api_key


def test_default_api_key_is_unique_instance():
    from nat.plugins.langchain.tools.perplexity_internet_search import PerplexityInternetSearchToolConfig

    config1 = PerplexityInternetSearchToolConfig()
    config2 = PerplexityInternetSearchToolConfig()

    assert config1.api_key is not config2.api_key


def test_max_retries_rejects_zero():
    from nat.plugins.langchain.tools.perplexity_internet_search import PerplexityInternetSearchToolConfig

    with pytest.raises(ValidationError):
        PerplexityInternetSearchToolConfig(max_retries=0)


def test_max_results_rejects_zero():
    from nat.plugins.langchain.tools.perplexity_internet_search import PerplexityInternetSearchToolConfig

    with pytest.raises(ValidationError):
        PerplexityInternetSearchToolConfig(max_results=0)


def test_max_results_rejects_above_20():
    from nat.plugins.langchain.tools.perplexity_internet_search import PerplexityInternetSearchToolConfig

    with pytest.raises(ValidationError):
        PerplexityInternetSearchToolConfig(max_results=21)


def test_invalid_search_recency_filter_rejected():
    from nat.plugins.langchain.tools.perplexity_internet_search import PerplexityInternetSearchToolConfig

    with pytest.raises(ValidationError):
        PerplexityInternetSearchToolConfig(search_recency_filter="invalid")


# -- Tool behavior tests --


@pytest.fixture
def tool_config():
    from nat.plugins.langchain.tools.perplexity_internet_search import PerplexityInternetSearchToolConfig

    return PerplexityInternetSearchToolConfig(api_key="test-key", max_retries=2, max_query_length=50)


def _mock_response(results: list[dict] | None):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"results": results}
    return response


def _mock_async_client(post_mock: AsyncMock):
    mock_client = MagicMock()
    mock_client.post = post_mock
    mock_context_manager = MagicMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)
    return mock_context_manager, mock_client


async def test_empty_key_returns_unavailable():
    from nat.plugins.langchain.tools.perplexity_internet_search import PerplexityInternetSearchToolConfig
    from nat.plugins.langchain.tools.perplexity_internet_search import perplexity_internet_search

    config = PerplexityInternetSearchToolConfig(api_key="")
    with patch.dict(os.environ, {"PERPLEXITY_API_KEY": ""}):
        async with perplexity_internet_search(config, None) as func_info:
            result = await func_info.single_fn("test query")
            assert "unavailable" in result.lower()
            assert "PERPLEXITY_API_KEY" in result


async def test_query_truncation(tool_config):
    from nat.plugins.langchain.tools.perplexity_internet_search import perplexity_internet_search

    long_query = "a" * 100  # exceeds max_query_length=50
    post_mock = AsyncMock(return_value=_mock_response([]))
    mock_context_manager, mock_client = _mock_async_client(post_mock)

    with patch(
        "nat.plugins.langchain.tools.perplexity_internet_search.httpx.AsyncClient", return_value=mock_context_manager
    ):
        async with perplexity_internet_search(tool_config, None) as func_info:
            await func_info.single_fn(long_query)

            # Verify the query was truncated
            call_args = mock_client.post.call_args
            truncated_query = call_args.kwargs["json"]["query"]
            assert len(truncated_query) <= 50
            assert truncated_query.endswith("...")


async def test_empty_results(tool_config):
    from nat.plugins.langchain.tools.perplexity_internet_search import perplexity_internet_search

    post_mock = AsyncMock(return_value=_mock_response([]))
    mock_context_manager, _ = _mock_async_client(post_mock)

    with patch(
        "nat.plugins.langchain.tools.perplexity_internet_search.httpx.AsyncClient", return_value=mock_context_manager
    ):
        async with perplexity_internet_search(tool_config, None) as func_info:
            result = await func_info.single_fn("test query")
            assert "No web search results found" in result


async def test_retries_on_exception(tool_config):
    from nat.plugins.langchain.tools.perplexity_internet_search import perplexity_internet_search

    post_mock = AsyncMock(side_effect=Exception("API error"))
    mock_context_manager, mock_client = _mock_async_client(post_mock)

    with (
        patch(
            "nat.plugins.langchain.tools.perplexity_internet_search.httpx.AsyncClient",
            return_value=mock_context_manager,
        ),
        patch("nat.plugins.langchain.tools.perplexity_internet_search.asyncio.sleep", new_callable=AsyncMock),
    ):
        async with perplexity_internet_search(tool_config, None) as func_info:
            result = await func_info.single_fn("test query")

            # Should have retried max_retries times (2)
            assert mock_client.post.call_count == 2
            assert "Web search failed after 2 attempts" in result


async def test_attribution_header_sent(tool_config):
    from nat.plugins.langchain.tools.perplexity_internet_search import perplexity_internet_search

    post_mock = AsyncMock(return_value=_mock_response([]))
    mock_context_manager, mock_client = _mock_async_client(post_mock)

    with patch(
        "nat.plugins.langchain.tools.perplexity_internet_search.httpx.AsyncClient", return_value=mock_context_manager
    ):
        async with perplexity_internet_search(tool_config, None) as func_info:
            await func_info.single_fn("test query")

            call_args = mock_client.post.call_args
            assert call_args.kwargs["headers"]["X-Pplx-Integration"].startswith("nemo-agent-toolkit/")


async def test_results_formatted_as_documents(tool_config):
    from nat.plugins.langchain.tools.perplexity_internet_search import perplexity_internet_search

    post_mock = AsyncMock(
        return_value=_mock_response(
            [
                {
                    "url": "https://example.com/one",
                    "snippet": "First result.",
                },
                {
                    "url": "https://example.com/two",
                    "snippet": "Second result.",
                },
            ]
        )
    )
    mock_context_manager, _ = _mock_async_client(post_mock)

    with patch(
        "nat.plugins.langchain.tools.perplexity_internet_search.httpx.AsyncClient", return_value=mock_context_manager
    ):
        async with perplexity_internet_search(tool_config, None) as func_info:
            result = await func_info.single_fn("test query")

            assert '<Document href="https://example.com/one"/>' in result
            assert '<Document href="https://example.com/two"/>' in result
            assert "\n\n---\n\n" in result
