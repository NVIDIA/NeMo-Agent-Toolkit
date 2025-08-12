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

# pylint: disable=not-async-context-manager

import json
import os
import sys
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.builder.builder import Builder
from nat.builder.function import LambdaFunction
from nat.builder.function_info import FunctionInfo
from nat.plugins.agno.tools.serp_api_tool import SerpApiToolConfig
from nat.plugins.agno.tools.serp_api_tool import serp_api_tool


# Mock the agno.tools.serpapi module and SerpApiTools class
class MockSerpApiTools:

    def __init__(self, api_key):
        self.api_key = api_key

    async def search_google(self, query, num_results):
        return []


# Create a patch for imports
mock_modules = {'agno.tools': MagicMock(), 'agno.tools.serpapi': MagicMock(), 'google-search-results': MagicMock()}
mock_modules['agno.tools'].serpapi = mock_modules['agno.tools.serpapi']


class TestSerpApiTool:
    """Tests for the serp_api_tool function."""

    @pytest.fixture
    def mock_builder(self):
        """Create a mock Builder object."""
        return MagicMock(spec=Builder)

    @pytest.fixture
    def tool_config(self):
        """Create a valid SerpApiToolConfig object."""
        return SerpApiToolConfig(api_key="test_api_key", max_results=3)

    @pytest.fixture
    def mock_serpapi_tools(self):
        """Create a mock SerpApiTools object."""
        mock = MagicMock()
        mock.search_google = AsyncMock()
        return mock

    @pytest.fixture
    def mock_search_results(self):
        """Create mock search results as a JSON string."""
        return json.dumps({
            "search_results": [{
                "title": "Test Result 1",
                "link": "https://example.com/1",
                "snippet": "This is the first test result snippet."
            },
                               {
                                   "title": "Test Result 2",
                                   "link": "https://example.com/2",
                                   "snippet": "This is the second test result snippet."
                               }]
        })

    @pytest.fixture
    def mock_incomplete_search_results(self):
        """Create mock search results as a JSON string."""
        return json.dumps({
            "search_results": [
                {
                    "title": "Complete Result",
                    "link": "https://example.com/complete",
                    "snippet": "This result has all fields."
                },
                {
                    # Missing title and snippet
                    "link": "https://example.com/incomplete"
                }
            ]
        })

    @pytest.mark.asyncio
    @patch.dict("sys.modules", {**sys.modules, **mock_modules})
    async def test_serp_api_tool_creation(self, tool_config, mock_builder):
        """Test that serp_api_tool correctly creates a FunctionInfo object."""
        # Set up the mock
        mock_tools = MagicMock()
        mock_serpapi_module = MagicMock()
        mock_serpapi_module.SerpApiTools = mock_tools
        sys.modules['agno.tools.serpapi'] = mock_serpapi_module

        # Call the function under test - handle as context manager
        async with serp_api_tool(tool_config, mock_builder) as fn_info:
            # Verify the result is a FunctionInfo instance
            assert isinstance(fn_info, FunctionInfo)

            # Verify SerpApiTools was created with the correct API key
            mock_tools.assert_called_once_with(api_key="test_api_key")

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"SERP_API_KEY": "env_api_key"})
    @patch.dict("sys.modules", {**sys.modules, **mock_modules})
    async def test_serp_api_tool_env_api_key(self, mock_builder):
        """Test that serp_api_tool correctly uses API key from environment."""
        # Create config without API key
        config = SerpApiToolConfig(max_results=3)

        # Set up the mock
        mock_tools = MagicMock()
        mock_serpapi_module = MagicMock()
        mock_serpapi_module.SerpApiTools = mock_tools
        sys.modules['agno.tools.serpapi'] = mock_serpapi_module

        # Call the function under test
        async with serp_api_tool(config, mock_builder) as fn_info:
            # Verify the result is a FunctionInfo instance
            assert isinstance(fn_info, FunctionInfo)

            # Verify SerpApiTools was created with the API key from environment
            mock_tools.assert_called_once_with(api_key="env_api_key")

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)  # Clear environment variables
    @patch.dict("sys.modules", {**sys.modules, **mock_modules})
    async def test_serp_api_tool_missing_api_key(self, mock_builder):
        """Test that serp_api_tool raises an error when API key is missing."""
        # Create config without API key
        config = SerpApiToolConfig(max_results=3)

        # Call the function under test and expect ValueError
        with pytest.raises(ValueError, match="API token must be provided"):
            async with serp_api_tool(config, mock_builder):
                pass

    @pytest.mark.asyncio
    @patch.dict("sys.modules", {**sys.modules, **mock_modules})
    async def test_serp_api_search_with_query(self, tool_config, mock_builder, mock_search_results):
        """Test that _serp_api_search correctly searches with a non-empty query."""
        # Set up the mocks
        mock_tool = MagicMock()
        mock_tool.search_google = MagicMock(return_value=mock_search_results)
        mock_tools = MagicMock(return_value=mock_tool)
        mock_serpapi_module = MagicMock()
        mock_serpapi_module.SerpApiTools = mock_tools
        sys.modules['agno.tools.serpapi'] = mock_serpapi_module

        # Get the function info
        async with serp_api_tool(tool_config, mock_builder) as fn_info:
            # Call the search function with a valid query
            serp_tool_instance = LambdaFunction.from_info(
                config=tool_config,
                info=fn_info,  # type: ignore
                instance_name="test_serp_tool")
            result = await serp_tool_instance.acall_invoke(query="test query")

            # Verify search was called with correct parameters
            mock_tool.search_google.assert_called_once_with(query="test query", num_results=3)

            # Verify the result contains formatted search results
            assert "Test Result 1" in result
            assert "https://example.com/1" in result
            assert "Test Result 2" in result
            assert "https://example.com/2" in result

    @pytest.mark.asyncio
    @patch.dict("sys.modules", {**sys.modules, **mock_modules})
    async def test_serp_api_search_exception_handling(self, tool_config, mock_builder):
        """Test that _serp_api_search correctly handles exceptions from the search API."""
        # Set up the mocks to raise an exception
        mock_tool = MagicMock()
        mock_tool.search_google = MagicMock(return_value="")
        mock_tools = MagicMock(return_value=mock_tool)
        mock_serpapi_module = MagicMock()
        mock_serpapi_module.SerpApiTools = mock_tools
        sys.modules['agno.tools.serpapi'] = mock_serpapi_module

        # Get the function info
        async with serp_api_tool(tool_config, mock_builder) as fn_info:
            # Call the search function
            serp_tool_instance = LambdaFunction.from_info(
                config=tool_config,
                info=fn_info,  # type: ignore
                instance_name="test_serp_tool")
            result = await serp_tool_instance.acall_invoke(query="test query")
            # Verify search was called
            mock_tool.search_google.assert_called_once()

            # Verify the result contains error information
            assert "Error performing search" in result

    @pytest.mark.asyncio
    @patch.dict("sys.modules", {**sys.modules, **mock_modules})
    async def test_serp_api_search_result_formatting(self, tool_config, mock_builder, mock_incomplete_search_results):
        """Test that _serp_api_search correctly formats search results."""
        # Setup the mocks
        mock_tool = MagicMock()
        mock_tool.search_google = MagicMock(return_value=mock_incomplete_search_results)
        mock_tools = MagicMock(return_value=mock_tool)
        mock_serpapi_module = MagicMock()
        mock_serpapi_module.SerpApiTools = mock_tools
        sys.modules['agno.tools.serpapi'] = mock_serpapi_module

        # Get the function info
        async with serp_api_tool(tool_config, mock_builder) as fn_info:
            # Call the search function
            serp_tool_instance = LambdaFunction.from_info(
                config=tool_config,
                info=fn_info,  # type: ignore
                instance_name="test_serp_tool")
            result = await serp_tool_instance.acall_invoke(query="test query")

            # Verify the result contains properly formatted search results
            assert "Complete Result" in result
            assert "https://example.com/complete" in result
            assert "This result has all fields" in result

            # Verify the result handles missing fields gracefully
            assert "No Title" in result
            assert "https://example.com/incomplete" in result
            assert "No Snippet" in result

            # Verify results are separated by the proper delimiter
            assert "---" in result

    @pytest.mark.asyncio
    @patch.dict("sys.modules", {**sys.modules, **mock_modules})
    async def test_serp_api_search_empty_results(self, tool_config, mock_builder):
        """Test that _serp_api_search correctly handles empty results from the search API."""
        # Set up the mocks to return empty results
        mock_tool = MagicMock()
        mock_tool.search_google = MagicMock(return_value=json.dumps({"search_results": []}))
        mock_tools = MagicMock(return_value=mock_tool)
        mock_serpapi_module = MagicMock()
        mock_serpapi_module.SerpApiTools = mock_tools
        sys.modules['agno.tools.serpapi'] = mock_serpapi_module

        # Get the function info
        async with serp_api_tool(tool_config, mock_builder) as fn_info:
            # Call the search function
            serp_tool_instance = LambdaFunction.from_info(
                config=tool_config,
                info=fn_info,  # type: ignore
                instance_name="test_serp_tool")
            result = await serp_tool_instance.acall_invoke(query="test query")

            # Verify search was called
            mock_tool.search_google.assert_called_once()

            # Verify the result is an empty string (no results to format)
            assert result == ""

    @pytest.mark.asyncio
    @patch.dict("sys.modules", {**sys.modules, **mock_modules})
    async def test_serp_api_tool_max_results(self, mock_builder, mock_search_results):
        """Test that serp_api_tool respects the max_results configuration."""
        # Create config with custom max_results
        tool_config = SerpApiToolConfig(api_key="test_api_key", max_results=10)

        # Set up the mocks
        mock_tool = MagicMock()
        mock_tool.search_google = MagicMock(return_value=mock_search_results)
        mock_tools = MagicMock(return_value=mock_tool)
        mock_serpapi_module = MagicMock()
        mock_serpapi_module.SerpApiTools = mock_tools
        sys.modules['agno.tools.serpapi'] = mock_serpapi_module

        # Get the function info
        async with serp_api_tool(tool_config, mock_builder) as fn_info:
            # Call the search function
            serp_tool_instance = LambdaFunction.from_info(
                config=tool_config,
                info=fn_info,  # type: ignore
                instance_name="test_serp_tool")
            await serp_tool_instance.acall_invoke(query="test query")

            # Verify search was called with the configured max_results
            mock_tool.search_google.assert_called_once_with(query="test query", num_results=10)
