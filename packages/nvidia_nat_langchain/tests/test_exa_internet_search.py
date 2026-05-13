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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import SecretStr
from pydantic import ValidationError

# -- Config validation tests --


@pytest.mark.parametrize("constructor_args", [{}, {
    "api_key": ""
}, {
    "api_key": "my_api_key"
}],
                         ids=["default", "empty_api_key", "provided_api_key"])
def test_api_key_is_secret_str(constructor_args: dict):
    from nat.plugins.langchain.tools.exa_internet_search import ExaInternetSearchToolConfig
    expected_api_key = constructor_args.get("api_key", "")

    config = ExaInternetSearchToolConfig(**constructor_args)
    assert isinstance(config.api_key, SecretStr)

    api_key = config.api_key.get_secret_value()
    assert api_key == expected_api_key


def test_default_api_key_is_unique_instance():
    from nat.plugins.langchain.tools.exa_internet_search import ExaInternetSearchToolConfig

    config1 = ExaInternetSearchToolConfig()
    config2 = ExaInternetSearchToolConfig()

    assert config1.api_key is not config2.api_key


def test_max_retries_rejects_zero():
    from nat.plugins.langchain.tools.exa_internet_search import ExaInternetSearchToolConfig

    with pytest.raises(ValidationError):
        ExaInternetSearchToolConfig(max_retries=0)


def test_max_results_rejects_zero():
    from nat.plugins.langchain.tools.exa_internet_search import ExaInternetSearchToolConfig

    with pytest.raises(ValidationError):
        ExaInternetSearchToolConfig(max_results=0)


def test_invalid_search_type_rejected():
    from nat.plugins.langchain.tools.exa_internet_search import ExaInternetSearchToolConfig

    with pytest.raises(ValidationError):
        ExaInternetSearchToolConfig(search_type="invalid")


def test_invalid_livecrawl_rejected():
    from nat.plugins.langchain.tools.exa_internet_search import ExaInternetSearchToolConfig

    with pytest.raises(ValidationError):
        ExaInternetSearchToolConfig(livecrawl="invalid")


# -- Tool behavior tests --


@pytest.fixture
def tool_config():
    from nat.plugins.langchain.tools.exa_internet_search import ExaInternetSearchToolConfig
    return ExaInternetSearchToolConfig(api_key="test-key", max_retries=2, max_query_length=50)


async def test_empty_key_returns_unavailable(tool_config):
    from nat.plugins.langchain.tools.exa_internet_search import ExaInternetSearchToolConfig
    from nat.plugins.langchain.tools.exa_internet_search import exa_internet_search

    config = ExaInternetSearchToolConfig(api_key="")
    async with exa_internet_search(config, None) as func_info:
        result = await func_info.single_fn("test query")
        assert "unavailable" in result.lower()
        assert "EXA_API_KEY" in result


async def test_query_truncation(tool_config):
    from nat.plugins.langchain.tools.exa_internet_search import exa_internet_search

    long_query = "a" * 100  # exceeds max_query_length=50

    mock_result = MagicMock()
    mock_result.results = []

    with patch("langchain_exa.ExaSearchResults") as mock_exa_cls:
        mock_instance = MagicMock()
        mock_instance._arun = AsyncMock(return_value=mock_result)
        mock_exa_cls.return_value = mock_instance

        async with exa_internet_search(tool_config, None) as func_info:
            await func_info.single_fn(long_query)

            # Verify the query was truncated
            call_args = mock_instance._arun.call_args
            truncated_query = call_args[0][0]
            assert len(truncated_query) <= 50
            assert truncated_query.endswith("...")


async def test_empty_results(tool_config):
    from nat.plugins.langchain.tools.exa_internet_search import exa_internet_search

    mock_result = MagicMock()
    mock_result.results = []

    with patch("langchain_exa.ExaSearchResults") as mock_exa_cls:
        mock_instance = MagicMock()
        mock_instance._arun = AsyncMock(return_value=mock_result)
        mock_exa_cls.return_value = mock_instance

        async with exa_internet_search(tool_config, None) as func_info:
            result = await func_info.single_fn("test query")
            assert "No web search results found" in result


async def test_retries_on_exception(tool_config):
    from nat.plugins.langchain.tools.exa_internet_search import exa_internet_search

    with patch("langchain_exa.ExaSearchResults") as mock_exa_cls:
        mock_instance = MagicMock()
        mock_instance._arun = AsyncMock(side_effect=Exception("API error"))
        mock_exa_cls.return_value = mock_instance

        async with exa_internet_search(tool_config, None) as func_info:
            result = await func_info.single_fn("test query")

            # Should have retried max_retries times (2)
            assert mock_instance._arun.call_count == 2
            assert "Web search failed after 2 attempts" in result
