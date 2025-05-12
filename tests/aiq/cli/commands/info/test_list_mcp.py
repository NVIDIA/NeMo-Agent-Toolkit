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

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from click.testing import CliRunner

# Replace this with the correct filename for your CLI script
from aiq.cli.commands.info.list_mcp import list_mcp


@pytest.fixture
def mock_tools():
    return [
        {
            "name": "tool_a",
            "description": "Tool A description",
            "input_schema": None,
        },
        {
            "name": "tool_b",
            "description": "Tool B description",
            "input_schema": '{"type": "object", "properties": {"x": {"type": "number"}}}',
        },
    ]


@patch("aiq.cli.commands.info.list_mcp.list_tools_and_schemas", new_callable=AsyncMock)
def test_list_tool_names(mock_fetcher, mock_tools):
    mock_fetcher.return_value = mock_tools
    runner = CliRunner()
    result = runner.invoke(list_mcp, [])
    assert result.exit_code == 0
    assert "tool_a" in result.output
    assert "tool_b" in result.output


@patch("aiq.cli.commands.info.list_mcp.list_tools_and_schemas", new_callable=AsyncMock)
def test_list_tool_details(mock_fetcher, mock_tools):
    mock_fetcher.return_value = mock_tools
    runner = CliRunner()
    result = runner.invoke(list_mcp, ["--detail"])
    assert result.exit_code == 0
    assert "Description: Tool A description" in result.output
    assert "Input Schema:" in result.output


@patch("aiq.cli.commands.info.list_mcp.list_tools_and_schemas", new_callable=AsyncMock)
def test_list_json_output(mock_fetcher, mock_tools):
    mock_fetcher.return_value = mock_tools
    runner = CliRunner()
    result = runner.invoke(list_mcp, ["--json-output"])
    assert result.exit_code == 0
    assert '"name": "tool_a"' in result.output
    assert result.output.strip().startswith("[")


@patch("aiq.cli.commands.info.list_mcp.list_tools_and_schemas", new_callable=AsyncMock)
def test_list_specific_tool(mock_fetcher, mock_tools):
    mock_fetcher.return_value = [mock_tools[1]]  # return only one tool
    runner = CliRunner()
    result = runner.invoke(list_mcp, ["--tool", "tool_b"])
    assert result.exit_code == 0
    assert "Tool: tool_b" in result.output
    assert "Description: Tool B description" in result.output
