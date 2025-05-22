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
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pytest_httpserver import HTTPServer


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


@pytest.mark.asyncio
async def test_sse_client_usage():
    '''
    Create a mock tool and verify that the client can call the tool and get a response.
    Use the SSE client to connect to the server.
    '''
    from mcp import ClientSession
    from mcp.types import TextContent

    from aiq.tool.mcp.mcp_client import MCPSSEClient

    # Mock the SSE client and session
    mock_session = AsyncMock(spec=ClientSession)
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=[TextContent(type="text", text="test response")])

    with patch('aiq.tool.mcp.mcp_client.sse_client') as mock_sse_client:
        mock_sse_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())

        client = MCPSSEClient("http://localhost:8080/sse")
        async with client.connect_to_server() as session:
            assert session == mock_session
            await session.initialize()
            result = await session.call_tool("test_tool", {"param": "value"})
            assert result[0].text == "test response"


@pytest.mark.asyncio
async def test_stdio_client_usage():
    '''
    Create a mock tool and verify that the client can call the tool and get a response.
    Use the stdio client to connect to the server.
    '''
    from mcp import ClientSession
    from mcp.types import TextContent

    from aiq.tool.mcp.mcp_client import MCPStdioClient

    # Mock the stdio client and session
    mock_session = AsyncMock(spec=ClientSession)
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=[TextContent(type="text", text="test response")])

    with patch('aiq.tool.mcp.mcp_client.stdio_client') as mock_stdio_client:
        mock_stdio_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())

        client = MCPStdioClient("python", ["-m", "test_server"], {"TEST_ENV": "test_value"})
        async with client.connect_to_server() as session:
            assert session == mock_session
            await session.initialize()
            result = await session.call_tool("test_tool", {"param": "value"})
            assert result[0].text == "test response"


@pytest.mark.asyncio
async def test_mcp_builder_with_stdio():
    '''
    Test MCPBuilder methods with stdio client.
    '''
    from mcp.types import TextContent

    from aiq.tool.mcp.mcp_client import MCPBuilder

    # Mock the tool response
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "Test tool description"
    mock_tool.inputSchema = {}

    # Mock the session
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[mock_tool]))
    mock_session.call_tool = AsyncMock(return_value=[TextContent(type="text", text="test response")])

    with patch('aiq.tool.mcp.mcp_client.stdio_client') as mock_stdio_client:
        mock_stdio_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())

        builder = MCPBuilder("python", client_type="stdio", args=["-m", "test_server"], env={"TEST_ENV": "test_value"})

        # Test getting tools
        tools = await builder.get_tools()
        assert "test_tool" in tools

        # Test getting a specific tool
        tool = await builder.get_tool("test_tool")
        assert tool.name == "test_tool"

        # Test calling a tool
        result = await builder.call_tool("test_tool", {"param": "value"})
        assert result[0].text == "test response"


@pytest.mark.asyncio
async def test_mcp_builder_with_sse():
    '''
    Test MCPBuilder methods with SSE client.
    '''
    from mcp.types import TextContent

    from aiq.tool.mcp.mcp_client import MCPBuilder

    # Mock the tool response
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "Test tool description"
    mock_tool.inputSchema = {}

    # Mock the session
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[mock_tool]))
    mock_session.call_tool = AsyncMock(return_value=[TextContent(type="text", text="test response")])

    with patch('aiq.tool.mcp.mcp_client.sse_client') as mock_sse_client:
        mock_sse_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())

        builder = MCPBuilder("http://localhost:8080/sse", client_type="sse")

        # Test getting tools
        tools = await builder.get_tools()
        assert "test_tool" in tools

        # Test getting a specific tool
        tool = await builder.get_tool("test_tool")
        assert tool.name == "test_tool"

        # Test calling a tool
        result = await builder.call_tool("test_tool", {"param": "value"})
        assert result[0].text == "test response"
