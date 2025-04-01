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

import os
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.plugins.agno.tools.serp_api_tool import SerpApiToolConfig
from aiq.plugins.agno.tools.serp_api_tool import serp_api_tool


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
        """Create mock search results."""
        return [{
            "title": "Test Result 1",
            "link": "https://example.com/1",
            "snippet": "This is the first test result snippet."
        },
                {
                    "title": "Test Result 2",
                    "link": "https://example.com/2",
                    "snippet": "This is the second test result snippet."
                }]

    @pytest.mark.asyncio
    @patch("aiq.plugins.agno.tools.serp_api_tool.SerpApiTools")
    async def test_serp_api_tool_creation(self, mock_serpapi_tools_class, tool_config, mock_builder):
        """Test that serp_api_tool correctly creates a FunctionInfo object."""
        # Set up the mock
        mock_serpapi_tools_class.return_value = MagicMock()

        # Call the function under test
        fn_info_generator = serp_api_tool(tool_config, mock_builder)
        fn_info = await anext(fn_info_generator)

        # Verify the result is a FunctionInfo instance
        assert isinstance(fn_info, FunctionInfo)

        # Verify SerpApiTools was created with the correct API key
        mock_serpapi_tools_class.assert_called_once_with(api_key="test_api_key")

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"SERP_API_KEY": "env_api_key"})
    @patch("aiq.plugins.agno.tools.serp_api_tool.SerpApiTools")
    async def test_serp_api_tool_env_api_key(self, mock_serpapi_tools_class, mock_builder):
        """Test that serp_api_tool correctly uses API key from environment."""
        # Create config without API key
        config = SerpApiToolConfig(max_results=3)

        # Set up the mock
        mock_serpapi_tools_class.return_value = MagicMock()

        # Call the function under test
        fn_info_generator = serp_api_tool(config, mock_builder)
        fn_info = await anext(fn_info_generator)

        # Verify the result is a FunctionInfo instance
        assert isinstance(fn_info, FunctionInfo)

        # Verify SerpApiTools was created with the API key from environment
        mock_serpapi_tools_class.assert_called_once_with(api_key="env_api_key")

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)  # Clear environment variables
    async def test_serp_api_tool_missing_api_key(self, mock_builder):
        """Test that serp_api_tool raises an error when API key is missing."""
        # Create config without API key
        config = SerpApiToolConfig(max_results=3)

        # Call the function under test and expect ValueError
        with pytest.raises(ValueError, match="API token must be provided"):
            fn_info_generator = serp_api_tool(config, mock_builder)
            await anext(fn_info_generator)

    @pytest.mark.asyncio
    @patch("aiq.plugins.agno.tools.serp_api_tool.SerpApiTools")
    async def test_serp_api_search_empty_query_first_time(self, mock_serpapi_tools_class, tool_config, mock_builder):
        """Test that _serp_api_search handles empty queries correctly (first time)."""
        # Set up the mocks
        mock_tool = MagicMock()
        mock_serpapi_tools_class.return_value = mock_tool

        # Reset module-level variable
        global _empty_query_handled
        _empty_query_handled = False

        # Get the function info
        fn_info_generator = serp_api_tool(tool_config, mock_builder)
        fn_info = await anext(fn_info_generator)

        # Call the search function with empty query
        result = await fn_info.fn("")

        # Verify the result
        assert "Tool is initialized" in result
        assert "Please provide a search query" in result

        # Verify search was not called
        mock_tool.search_google.assert_not_called()

        # Verify global flag was set
        assert _empty_query_handled is True

    @pytest.mark.asyncio
    @patch("aiq.plugins.agno.tools.serp_api_tool.SerpApiTools")
    async def test_serp_api_search_empty_query_subsequent(self, mock_serpapi_tools_class, tool_config, mock_builder):
        """Test that _serp_api_search handles empty queries correctly (subsequent times)."""
        # Set up the mocks
        mock_tool = MagicMock()
        mock_serpapi_tools_class.return_value = mock_tool

        # Set module-level variable to simulate previous empty query
        global _empty_query_handled
        _empty_query_handled = True

        # Get the function info
        fn_info_generator = serp_api_tool(tool_config, mock_builder)
        fn_info = await anext(fn_info_generator)

        # Call the search function with empty query
        result = await fn_info.fn("")

        # Verify the result contains error message
        assert "ERROR" in result
        assert "Search query cannot be empty" in result

        # Verify search was not called
        mock_tool.search_google.assert_not_called()

    @pytest.mark.asyncio
    @patch("aiq.plugins.agno.tools.serp_api_tool.SerpApiTools")
    async def test_serp_api_search_with_query(self,
                                              mock_serpapi_tools_class,
                                              tool_config,
                                              mock_builder,
                                              mock_search_results):
        """Test that _serp_api_search correctly searches with a non-empty query."""
        # Set up the mocks
        mock_tool = MagicMock()
        mock_tool.search_google = AsyncMock(return_value=mock_search_results)
        mock_serpapi_tools_class.return_value = mock_tool

        # Set module-level variable (should be reset after successful search)
        global _empty_query_handled
        _empty_query_handled = True

        # Get the function info
        fn_info_generator = serp_api_tool(tool_config, mock_builder)
        fn_info = await anext(fn_info_generator)

        # Call the search function with a valid query
        result = await fn_info.fn("test query")

        # Verify search was called with correct parameters
        mock_tool.search_google.assert_called_once_with(query="test query", num_results=3)

        # Verify the result contains formatted search results
        assert "Test Result 1" in result
        assert "https://example.com/1" in result
        assert "Test Result 2" in result
        assert "https://example.com/2" in result

        # Verify global flag was reset
        assert _empty_query_handled is False

    @pytest.mark.asyncio
    @patch("aiq.plugins.agno.tools.serp_api_tool.SerpApiTools")
    async def test_serp_api_search_exception_handling(self, mock_serpapi_tools_class, tool_config, mock_builder):
        """Test that _serp_api_search correctly handles exceptions from the search API."""
        # Set up the mocks to raise an exception
        mock_tool = MagicMock()
        mock_tool.search_google = AsyncMock(side_effect=Exception("API error"))
        mock_serpapi_tools_class.return_value = mock_tool

        # Get the function info
        fn_info_generator = serp_api_tool(tool_config, mock_builder)
        fn_info = await anext(fn_info_generator)

        # Call the search function
        result = await fn_info.fn("test query")

        # Verify search was called
        mock_tool.search_google.assert_called_once()

        # Verify the result contains error information
        assert "Error performing search" in result
        assert "API error" in result

    @pytest.mark.asyncio
    @patch("aiq.plugins.agno.tools.serp_api_tool.SerpApiTools")
    async def test_serp_api_search_result_formatting(self, mock_serpapi_tools_class, tool_config, mock_builder):
        """Test that _serp_api_search correctly formats search results."""
        # Create a search result with missing fields
        incomplete_results = [
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

        # Set up the mocks
        mock_tool = MagicMock()
        mock_tool.search_google = AsyncMock(return_value=incomplete_results)
        mock_serpapi_tools_class.return_value = mock_tool

        # Get the function info
        fn_info_generator = serp_api_tool(tool_config, mock_builder)
        fn_info = await anext(fn_info_generator)

        # Call the search function
        result = await fn_info.fn("test query")

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
    @patch("aiq.plugins.agno.tools.serp_api_tool.SerpApiTools")
    async def test_serp_api_search_empty_results(self, mock_serpapi_tools_class, tool_config, mock_builder):
        """Test that _serp_api_search correctly handles empty results from the search API."""
        # Set up the mocks to return empty results
        mock_tool = MagicMock()
        mock_tool.search_google = AsyncMock(return_value=[])
        mock_serpapi_tools_class.return_value = mock_tool

        # Get the function info
        fn_info_generator = serp_api_tool(tool_config, mock_builder)
        fn_info = await anext(fn_info_generator)

        # Call the search function
        result = await fn_info.fn("test query")

        # Verify search was called
        mock_tool.search_google.assert_called_once()

        # Verify the result is an empty string (no results to format)
        assert result == ""

    @pytest.mark.asyncio
    @patch("aiq.plugins.agno.tools.serp_api_tool.SerpApiTools")
    async def test_serp_api_tool_max_results(self, mock_serpapi_tools_class, mock_builder):
        """Test that serp_api_tool respects the max_results configuration."""
        # Create config with custom max_results
        config = SerpApiToolConfig(api_key="test_api_key", max_results=10)

        # Set up the mocks
        mock_tool = MagicMock()
        mock_tool.search_google = AsyncMock(return_value=[{
            "title": "Test", "link": "https://example.com", "snippet": "Test"
        }])
        mock_serpapi_tools_class.return_value = mock_tool

        # Get the function info
        fn_info_generator = serp_api_tool(config, mock_builder)
        fn_info = await anext(fn_info_generator)

        # Call the search function
        await fn_info.fn("test query")

        # Verify search was called with the configured max_results
        mock_tool.search_google.assert_called_once_with(query="test query", num_results=10)
