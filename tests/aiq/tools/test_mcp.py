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

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from mcp.types import TextContent
from pytest_httpserver import HTTPServer

from aiq.tool.mcp.mcp_client import MCPBuilder
from aiq.tool.mcp.mcp_client import MCPSSEClient
from aiq.tool.mcp.mcp_client import MCPStdioClient


@pytest.fixture(name="test_mcp_server")
def _get_test_mcp_server(httpserver: HTTPServer):
    httpserver.expect_request("/sse", )


@pytest.fixture(name="sample_schema")
def _get_sample_schema():
    return {
        'description': 'Test Tool',
        'properties': {
            'required_string_field': {
                'description': 'Required field that needs to be a string',
                'minLength': 1,
                'title': 'RequiredString',
                'type': 'string'
            },
            'optional_string_field': {
                'default': 'default_string',
                'description': 'Optional field that needs to be a string',
                'minLength': 1,
                'title': 'OptionalString',
                'type': 'string'
            },
            'required_int_field': {
                'description': 'Required int field.',
                'exclusiveMaximum': 1000000,
                'exclusiveMinimum': 0,
                'title': 'Required Int',
                'type': 'integer'
            },
            'optional_int_field': {
                'default': 5000,
                'description': 'Optional Integer field.',
                'exclusiveMaximum': 1000000,
                'exclusiveMinimum': 0,
                'title': 'Optional Int',
                'type': 'integer'
            },
            'required_float_field': {
                'description': 'Optional Float Field.', 'title': 'Optional Float', 'type': 'number'
            },
            'optional_float_field': {
                'default': 5.0, 'description': 'Optional Float Field.', 'title': 'Optional Float', 'type': 'number'
            },
            'optional_bool_field': {
                'default': False, 'description': 'Optional Boolean Field.', 'title': 'Raw', 'type': 'boolean'
            }
        },
        'required': [
            'required_string_field',
            'required_int_field',
            'required_float_field',
        ],
        'title': 'Fetch',
        'type': 'object'
    }


def test_schema_generation(sample_schema):
    from aiq.tool.mcp.mcp_client import model_from_mcp_schema
    _model = model_from_mcp_schema("test_model", sample_schema)

    for k, _ in sample_schema["properties"].items():
        assert k in _model.model_fields.keys()

    test_input = {
        "required_string_field": "This is a string",
        "optional_string_field": "This is another string",
        "required_int_field": 4,
        "optional_int_field": 1,
        "required_float_field": 5.5,
        "optional_float_field": 3.2,
        "optional_bool_field": True,
    }

    m = _model.model_validate(test_input)
    assert isinstance(m, _model)

    test_input = {
        "required_string_field": "This is a string",
        "required_int_field": 4,
        "required_float_field": 5.5,
    }

    m = _model.model_validate(test_input)
    assert isinstance(m, _model)


@pytest.fixture
def mock_tool():
    tool = MagicMock()
    tool.name = "test_tool"
    tool.description = "Test tool description"
    tool.inputSchema = {}
    return tool


@pytest.fixture
def mock_session(mock_tool):
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[mock_tool]))
    session.call_tool = AsyncMock(return_value=[TextContent(type="text", text="test response")])
    return session


@pytest.fixture
def tool_input():
    return {"param": "value"}


@pytest.mark.asyncio
async def test_sse_client_usage(mock_session, tool_input):
    with patch('aiq.tool.mcp.mcp_client.sse_client') as mock_sse_client, \
     patch('aiq.tool.mcp.mcp_client.ClientSession') as mock_client_session:
        mock_sse_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())
        mock_client_session.return_value.__aenter__.return_value = mock_session

        client = MCPSSEClient("http://localhost:8080/sse")
        async with client.connect_to_server() as session:
            await session.initialize()
            result = await session.call_tool("test_tool", tool_input)
            assert result[0].text == "test response"

        mock_sse_client.assert_called_once()


@pytest.mark.asyncio
async def test_stdio_client_usage(mock_session, tool_input):
    with patch('aiq.tool.mcp.mcp_client.stdio_client') as mock_stdio_client, \
     patch('aiq.tool.mcp.mcp_client.ClientSession') as mock_client_session:
        mock_stdio_client.return_value.__aenter__.return_value = (mock_session, AsyncMock())
        mock_client_session.return_value.__aenter__.return_value = mock_session

        client = MCPStdioClient("python", ["-m", "test_server"], {"TEST_ENV": "test_value"})
        async with client.connect_to_server() as session:
            await session.initialize()
            result = await session.call_tool("test_tool", tool_input)
            assert result[0].text == "test response"

        mock_stdio_client.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_builder_with_stdio(mock_session, mock_tool, tool_input):
    with patch('aiq.tool.mcp.mcp_client.stdio_client') as mock_stdio_client, \
     patch('aiq.tool.mcp.mcp_client.ClientSession') as mock_client_session:
        mock_stdio_client.return_value.__aenter__.return_value = (mock_session, AsyncMock())
        mock_client_session.return_value.__aenter__.return_value = mock_session

        builder = MCPBuilder("python", client_type="stdio", args=["-m", "test_server"], env={"TEST_ENV": "test_value"})

        tools = await builder.get_tools()
        assert "test_tool" in tools

        tool = await builder.get_tool("test_tool")
        assert tool.name == "test_tool"

        result = await builder.call_tool("test_tool", tool_input)
        assert result[0].text == "test response"

        mock_stdio_client.assert_called()
        assert mock_client_session.call_count == 3


@pytest.mark.asyncio
async def test_mcp_builder_with_sse(mock_session, mock_tool, tool_input):
    with patch('aiq.tool.mcp.mcp_client.sse_client') as mock_sse_client, \
     patch('aiq.tool.mcp.mcp_client.ClientSession') as mock_client_session:
        mock_sse_client.return_value.__aenter__.return_value = (mock_session, AsyncMock())
        mock_client_session.return_value.__aenter__.return_value = mock_session

        builder = MCPBuilder("http://localhost:8080/sse", client_type="sse")

        tools = await builder.get_tools()
        assert "test_tool" in tools

        tool = await builder.get_tool("test_tool")
        assert tool.name == "test_tool"

        result = await builder.call_tool("test_tool", tool_input)
        assert result[0].text == "test response"

        mock_sse_client.assert_called()
        assert mock_client_session.call_count == 3
