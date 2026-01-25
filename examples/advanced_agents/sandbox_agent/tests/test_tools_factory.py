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
"""Tests for tool factory functions."""

from nat_sandbox_agent.tools import create_all_tools
from nat_sandbox_agent.tools import create_host_tools
from nat_sandbox_agent.tools import create_sandbox_tools
from nat_sandbox_agent.tools import get_tool_descriptions


class TestCreateAllTools:
    """Tests for create_all_tools factory function."""

    def test_returns_correct_number_of_tools(self, mock_sandbox):
        """Test that create_all_tools returns all 7 tools."""
        tools = create_all_tools(sandbox=mock_sandbox)

        assert len(tools) == 7

    def test_returns_all_expected_tool_names(self, mock_sandbox):
        """Test that all expected tools are present."""
        tools = create_all_tools(sandbox=mock_sandbox)
        tool_names = {t.name for t in tools}

        expected_tools = {
            "shell",
            "python",
            "file_read",
            "file_write",
            "web_browse",
            "web_search",
            "youtube_transcript",
        }
        assert tool_names == expected_tools

    def test_include_tools_filters_correctly(self, mock_sandbox):
        """Test that include_tools parameter filters the tools."""
        tools = create_all_tools(
            sandbox=mock_sandbox,
            include_tools=["shell", "python", "web_search"],
        )

        tool_names = {t.name for t in tools}
        assert tool_names == {"shell", "python", "web_search"}
        assert len(tools) == 3

    def test_include_tools_ignores_unknown_names(self, mock_sandbox):
        """Test that unknown tool names are silently ignored."""
        tools = create_all_tools(
            sandbox=mock_sandbox,
            include_tools=["shell", "unknown_tool", "python"],
        )

        tool_names = {t.name for t in tools}
        assert tool_names == {"shell", "python"}
        assert len(tools) == 2

    def test_accepts_custom_tavily_api_key(self, mock_sandbox):
        """Test that tavily_api_key parameter is accepted."""
        # Should not raise
        tools = create_all_tools(
            sandbox=mock_sandbox,
            tavily_api_key="test-api-key",
        )
        assert len(tools) == 7

    def test_accepts_custom_max_output_chars(self, mock_sandbox):
        """Test that max_output_chars parameter is accepted."""
        tools = create_all_tools(
            sandbox=mock_sandbox,
            max_output_chars=5000,
        )
        assert len(tools) == 7


class TestCreateSandboxTools:
    """Tests for create_sandbox_tools function."""

    def test_returns_five_sandbox_tools(self, mock_sandbox):
        """Test that create_sandbox_tools returns exactly 5 tools."""
        tools = create_sandbox_tools(sandbox=mock_sandbox)

        assert len(tools) == 5

    def test_returns_expected_sandbox_tools(self, mock_sandbox):
        """Test that all sandbox tools are present."""
        tools = create_sandbox_tools(sandbox=mock_sandbox)
        tool_names = {t.name for t in tools}

        expected = {"shell", "python", "file_read", "file_write", "web_browse"}
        assert tool_names == expected

    def test_include_tools_filters_sandbox_tools(self, mock_sandbox):
        """Test filtering sandbox tools with include_tools."""
        tools = create_sandbox_tools(
            sandbox=mock_sandbox,
            include_tools=["shell", "file_read"],
        )

        tool_names = {t.name for t in tools}
        assert tool_names == {"shell", "file_read"}


class TestCreateHostTools:
    """Tests for create_host_tools function."""

    def test_returns_two_host_tools(self):
        """Test that create_host_tools returns exactly 2 tools."""
        tools = create_host_tools()

        assert len(tools) == 2

    def test_returns_expected_host_tools(self):
        """Test that all host tools are present."""
        tools = create_host_tools()
        tool_names = {t.name for t in tools}

        expected = {"web_search", "youtube_transcript"}
        assert tool_names == expected


class TestGetToolDescriptions:
    """Tests for get_tool_descriptions function."""

    def test_returns_string(self):
        """Test that get_tool_descriptions returns a string."""
        descriptions = get_tool_descriptions()

        assert isinstance(descriptions, str)

    def test_contains_all_tool_names(self):
        """Test that all tool names are mentioned in descriptions."""
        descriptions = get_tool_descriptions()

        expected_tools = [
            "shell",
            "python",
            "file_read",
            "file_write",
            "web_browse",
            "web_search",
            "youtube_transcript",
        ]
        for tool_name in expected_tools:
            assert tool_name in descriptions

    def test_starts_with_header(self):
        """Test that descriptions start with proper header."""
        descriptions = get_tool_descriptions()

        assert descriptions.startswith("Available tools:")

    def test_each_tool_has_description(self):
        """Test that each tool line has a description."""
        descriptions = get_tool_descriptions()
        lines = descriptions.split("\n")

        # Skip header line
        tool_lines = [line for line in lines[1:] if line.strip()]

        for line in tool_lines:
            assert ":" in line, f"Tool line missing description separator: {line}"
            assert line.strip().startswith("- "), f"Tool line missing bullet: {line}"
