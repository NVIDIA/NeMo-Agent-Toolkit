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
import argparse
import asyncio
import os

import pytest
import uvicorn
from mcp.server.fastmcp.server import FastMCP
from mcp.types import TextContent
from pydantic.networks import HttpUrl
from pytest_httpserver import HTTPServer

from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.tool.mcp.mcp_client_base import MCPBaseClient
from aiq.tool.mcp.mcp_client_base import MCPSSEClient
from aiq.tool.mcp.mcp_client_base import MCPStdioClient
from aiq.tool.mcp.mcp_client_base import MCPStreamableHTTPClient
from aiq.tool.mcp.mcp_client_base import model_from_mcp_schema
from aiq.tool.mcp.mcp_tool import MCPToolConfig


def _create_test_mcp_server(port: int):
    s = FastMCP(name="Test Server", description="Test Server", port=port)

    @s.tool()
    async def return_42(param: str):
        return f"{param} 42 {os.environ['TEST']}"

    @s.tool()
    async def throw_error(param: str):
        raise RuntimeError(f"Error message: {param}")

    return s


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
            },
            'optional_array_field': {
                'default': ['item'],
                'description': 'Optional Array Field.',
                'title': 'Array',
                'type': 'array',
                'items': {
                    'type': 'string'
                }
            },
            'optional_array_object_field': {
                'default': [{
                    'key': 'value'
                }],
                'description': 'Optional Array Field.',
                'title': 'Array',
                'type': 'array',
                'items': {
                    'type': 'object', 'properties': {
                        'key': {
                            'type': 'string'
                        }
                    }
                }
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
        "optional_array_field": ["item1"],
        "optional_array_object_field": [{
            'key': 'value1'
        }],
    }

    m = _model.model_validate(test_input)
    assert isinstance(m, _model)


@pytest.fixture(name="mcp_client", params=["stdio", "sse", "streamable-http"])
async def mcp_client_fixture(request: pytest.FixtureRequest):

    async with asyncio.TaskGroup() as tg:

        server: uvicorn.Server | None = None

        os.environ["TEST"] = "env value"

        if request.param == "stdio":

            client = MCPStdioClient(command="python",
                                    args=[__file__, "--transport", request.param],
                                    env={"TEST": os.environ["TEST"]})

        elif request.param == "sse":

            mcp_server = _create_test_mcp_server(port=8123)

            config = uvicorn.Config(
                mcp_server.sse_app(),
                host=mcp_server.settings.host,
                port=mcp_server.settings.port,
                log_level=mcp_server.settings.log_level.lower(),
            )
            server = uvicorn.Server(config)

            tg.create_task(server.serve())

            client = MCPSSEClient(url="http://localhost:8123/sse")

        elif request.param == "streamable-http":

            mcp_server = _create_test_mcp_server(port=8124)

            config = uvicorn.Config(
                mcp_server.streamable_http_app(),
                host=mcp_server.settings.host,
                port=mcp_server.settings.port,
                log_level=mcp_server.settings.log_level.lower(),
            )
            server = uvicorn.Server(config)

            tg.create_task(server.serve())

            client = MCPStreamableHTTPClient(url="http://localhost:8124/mcp/")

        else:
            raise ValueError(f"Invalid transport: {request.param}")

        yield client

        await asyncio.sleep(1)

        if server:
            # Signal the server to exit. Task group will wait for the server to exit.
            server.should_exit = True

    await asyncio.sleep(1)


async def test_mcp_client(mcp_client: MCPBaseClient):

    async with mcp_client:

        # Test get_tools
        tools = await mcp_client.get_tools()
        assert len(tools) == 2
        assert "return_42" in tools

        # Test get_tool
        tool = await mcp_client.get_tool("return_42")
        assert tool.name == "return_42"

        # Test call_tool
        result = await mcp_client.call_tool("return_42", {"param": "value"})

        value = result.content[0]

        assert isinstance(value, TextContent)
        assert value.text == f"value 42 {os.environ['TEST']}"


async def test_error_handling(mcp_client: MCPBaseClient):
    async with mcp_client:

        tool = await mcp_client.get_tool("throw_error")

        with pytest.raises(RuntimeError) as e:
            await tool.acall({"param": "value"})

        assert "Error message: value" in str(e.value)


async def test_function(mcp_client: MCPBaseClient):
    async with WorkflowBuilder() as builder:

        if isinstance(mcp_client, MCPSSEClient):
            fn_obj = await builder.add_function(name="test_function",
                                                config=MCPToolConfig(url=HttpUrl(mcp_client.url),
                                                                     mcp_tool_name="return_42",
                                                                     client_type=mcp_client.client_type))
        elif isinstance(mcp_client, MCPStdioClient):
            fn_obj = await builder.add_function(name="test_function",
                                                config=MCPToolConfig(mcp_tool_name="return_42",
                                                                     client_type=mcp_client.client_type,
                                                                     command=mcp_client.command,
                                                                     args=mcp_client.args,
                                                                     env=mcp_client.env))
        elif isinstance(mcp_client, MCPStreamableHTTPClient):
            fn_obj = await builder.add_function(name="test_function",
                                                config=MCPToolConfig(url=HttpUrl(mcp_client.url),
                                                                     mcp_tool_name="return_42",
                                                                     client_type=mcp_client.client_type))
        else:
            raise ValueError(f"Invalid client type: {type(mcp_client)}")

        assert fn_obj.has_single_output

        result = await fn_obj.acall_invoke(param="value")

        assert result == "value 42 env value"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MCP Server")
    parser.add_argument("--transport", type=str, default="stdio", help="Transport to use for the server")

    args = parser.parse_args()

    _create_test_mcp_server(port=8122).run(transport=args.transport)
