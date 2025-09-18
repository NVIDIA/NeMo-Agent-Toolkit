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

import pytest
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from starlette.testclient import TestClient

from nat.builder.function_base import FunctionBase
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.front_ends.mcp.mcp_front_end_config import MCPFrontEndConfig
from nat.front_ends.mcp.mcp_front_end_plugin_worker import MCPFrontEndPluginWorker


# Test fixtures and mock classes
class MockTestSchema(BaseModel):
    """Test schema for regular functions."""
    text: str | None = None
    number: int = 42


class ChatRequestSchema(BaseModel):
    """Mock ChatRequest schema for testing special handling."""
    messages: list = []
    model: str | None = None


class WorkflowMock:
    """Mock workflow class."""

    def __init__(self):
        self.config = type("Cfg", (), {"workflow": type("W", (), {"type": "test_workflow"})})()
        self.functions = {}
        self.function_groups = {}

    def run(self, *_args, **_kwargs):
        """Mock run method to identify as workflow."""
        return "workflow_result"


class RegularFunction(FunctionBase[str, str, str]):
    """Regular function with test schema."""
    description = "Regular function description"

    def __init__(self):
        super().__init__(input_schema=MockTestSchema)

    async def _ainvoke(self, value: str) -> str:
        return value

    async def _astream(self, value: str):
        yield value


class ChatRequestFunction(FunctionBase[str, str, str]):
    """Function with ChatRequest schema for testing special handling."""
    description = "Chat request function description"

    def __init__(self):
        super().__init__(input_schema=ChatRequestSchema)

    async def _ainvoke(self, value: str) -> str:
        return value

    async def _astream(self, value: str):
        yield value


class NoSchemaFunction(FunctionBase[str, str, str]):
    """Function without input schema."""
    description = "Function without schema"
    input_schema = None

    def __init__(self):
        super().__init__(input_schema=None)

    async def _ainvoke(self, value: str) -> str:
        return value

    async def _astream(self, value: str):
        yield value


@pytest.fixture
def mcp_config():
    """Fixture providing MCP configuration."""
    return Config(general=GeneralConfig(front_end=MCPFrontEndConfig(name="Test MCP")))


@pytest.fixture
def worker(mcp_config):
    """Fixture providing MCP worker instance."""
    return MCPFrontEndPluginWorker(mcp_config)


@pytest.fixture
def mcp_server():
    """Fixture providing FastMCP server instance."""
    return FastMCP("Test Server")


@pytest.fixture
def test_functions():
    """Fixture providing a comprehensive set of test functions."""
    return {
        "regular_tool": RegularFunction(),
        "chat_tool": ChatRequestFunction(),
        "no_schema_tool": NoSchemaFunction(),
        "workflow_tool": WorkflowMock(),
    }


@pytest.fixture
def setup_debug_endpoints(worker, mcp_server, test_functions):
    """Fixture that sets up debug endpoints with test functions."""
    worker._setup_debug_endpoints(mcp_server, test_functions)
    return mcp_server


# Streamlined test cases focusing on essential functionality
class TestDebugRouteBasics:
    """Test basic functionality of the debug route."""

    @pytest.mark.asyncio
    async def test_route_exists_and_structure(self, setup_debug_endpoints, test_functions):
        """Test that the debug route is accessible and returns expected structure."""
        with TestClient(setup_debug_endpoints.streamable_http_app()) as client:
            resp = client.get("/debug/tools/list")
            assert resp.status_code == 200
            data = resp.json()

            # Check response structure
            assert "count" in data
            assert "tools" in data
            assert "server_name" in data
            assert data["server_name"] == "Test Server"
            assert data["count"] == len(test_functions)

            # Check all tools are listed
            tool_names = {tool["name"] for tool in data["tools"]}
            expected_names = set(test_functions.keys())
            assert tool_names == expected_names


class TestDetailParameter:
    """Test the detail parameter behavior."""

    @pytest.mark.parametrize(
        "detail_param,expected_schema",
        [
            ("true", True),
            ("false", False),
            ("invalid", True)  # Invalid defaults to True when names specified
        ])
    @pytest.mark.asyncio
    async def test_detail_with_names(self, setup_debug_endpoints, detail_param, expected_schema):
        """Test detail parameter when tool names are specified."""
        with TestClient(setup_debug_endpoints.streamable_http_app()) as client:
            resp = client.get(f"/debug/tools/list?name=regular_tool&detail={detail_param}")
            data = resp.json()

            if expected_schema:
                assert "schema" in data["tools"][0]
                assert data["tools"][0]["schema"] is not None
            else:
                assert "schema" not in data["tools"][0]

    @pytest.mark.parametrize(
        "detail_param,expected_schema",
        [
            ("true", True),
            ("false", False),
            ("invalid", False)  # Invalid defaults to False when no names
        ])
    @pytest.mark.asyncio
    async def test_detail_without_names(self, setup_debug_endpoints, detail_param, expected_schema):
        """Test detail parameter when no tool names are specified."""
        with TestClient(setup_debug_endpoints.streamable_http_app()) as client:
            resp = client.get(f"/debug/tools/list?detail={detail_param}")
            data = resp.json()

            for tool in data["tools"]:
                assert "is_workflow" in tool  # Always present
                if expected_schema:
                    assert "schema" in tool
                else:
                    assert "schema" not in tool

    @pytest.mark.asyncio
    async def test_defaults(self, setup_debug_endpoints):
        """Test default behavior for detail parameter."""
        with TestClient(setup_debug_endpoints.streamable_http_app()) as client:
            # No names, no detail -> simplified (no schema)
            resp1 = client.get("/debug/tools/list")
            data1 = resp1.json()
            assert "schema" not in data1["tools"][0]
            assert "is_workflow" in data1["tools"][0]

            # With names, no detail -> detailed (with schema)
            resp2 = client.get("/debug/tools/list?name=regular_tool")
            data2 = resp2.json()
            assert "schema" in data2["tools"][0]
            assert data2["tools"][0]["schema"] is not None


class TestNameParameter:
    """Test the name parameter behavior."""

    @pytest.mark.parametrize("name_param,expected_count",
                             [
                                 ("regular_tool", 1),
                                 ("regular_tool,chat_tool", 2),
                                 ("regular_tool,chat_tool,no_schema_tool,workflow_tool", 4),
                             ])
    @pytest.mark.asyncio
    async def test_name_parameter_formats(self, setup_debug_endpoints, name_param, expected_count):
        """Test various name parameter formats."""
        with TestClient(setup_debug_endpoints.streamable_http_app()) as client:
            resp = client.get(f"/debug/tools/list?name={name_param}")
            data = resp.json()

            assert "count" in data
            assert data["count"] == expected_count
            assert len(data["tools"]) == expected_count

    @pytest.mark.asyncio
    async def test_repeated_name_parameters(self, setup_debug_endpoints):
        """Test multiple name parameters (repeated query params)."""
        with TestClient(setup_debug_endpoints.streamable_http_app()) as client:
            resp = client.get("/debug/tools/list?name=regular_tool&name=chat_tool")
            data = resp.json()

            assert data["count"] == 2
            returned_names = {tool["name"] for tool in data["tools"]}
            assert returned_names == {"regular_tool", "chat_tool"}

    @pytest.mark.asyncio
    async def test_invalid_tool_names(self, setup_debug_endpoints):
        """Test that invalid tool names return 404."""
        with TestClient(setup_debug_endpoints.streamable_http_app()) as client:
            resp = client.get("/debug/tools/list?name=nonexistent_tool")
            assert resp.status_code == 404
            # HTTPException returns plain text, not JSON
            assert "not found" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_edge_cases(self, setup_debug_endpoints):
        """Test edge cases for name parameter."""
        with TestClient(setup_debug_endpoints.streamable_http_app()) as client:
            # Empty name -> returns all tools
            resp1 = client.get("/debug/tools/list?name=")
            data1 = resp1.json()
            assert data1["count"] == 4

            # Duplicate names -> deduplicated
            resp2 = client.get("/debug/tools/list?name=regular_tool,regular_tool")
            data2 = resp2.json()
            assert data2["count"] == 1
            assert data2["tools"][0]["name"] == "regular_tool"


class TestSchemaHandling:
    """Test schema generation and handling."""

    @pytest.mark.asyncio
    async def test_regular_schema_generation(self, setup_debug_endpoints):
        """Test that regular Pydantic schemas are generated correctly."""
        with TestClient(setup_debug_endpoints.streamable_http_app()) as client:
            resp = client.get("/debug/tools/list?name=regular_tool&detail=true")
            data = resp.json()

            schema = data["tools"][0]["schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "text" in schema["properties"]
            assert "number" in schema["properties"]

    @pytest.mark.asyncio
    async def test_chat_request_schema_simplification(self, setup_debug_endpoints):
        """Test that ChatRequest schemas are simplified."""
        with TestClient(setup_debug_endpoints.streamable_http_app()) as client:
            resp = client.get("/debug/tools/list?name=chat_tool&detail=true")
            data = resp.json()

            schema = data["tools"][0]["schema"]
            assert schema["type"] == "object"
            assert schema["title"] == "ChatRequestQuery"
            assert "query" in schema["properties"]
            assert schema["properties"]["query"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_no_schema_and_workflow_identification(self, setup_debug_endpoints):
        """Test handling of functions without schemas and workflow identification."""
        with TestClient(setup_debug_endpoints.streamable_http_app()) as client:
            # No schema - check that schema field is not present when detail=false
            resp1 = client.get("/debug/tools/list?name=no_schema_tool&detail=false")
            data1 = resp1.json()
            assert "schema" not in data1["tools"][0]

            # With detail=true, schema should be present (even if None)
            resp2 = client.get("/debug/tools/list?name=no_schema_tool&detail=true")
            data2 = resp2.json()
            assert "schema" in data2["tools"][0]

            # Workflow identification
            resp2 = client.get("/debug/tools/list?name=workflow_tool&detail=true")
            data2 = resp2.json()
            assert data2["tools"][0]["is_workflow"] is True
