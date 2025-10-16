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
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nat.data_models.config import Config
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker


class _ToolStub:

    def __init__(self, description: str):
        self.description = description


class _ClientStub:

    def __init__(self, server_name: str, tools: dict[str, _ToolStub], raise_on_get: bool = False):
        self.server_name = server_name
        self._tools = tools
        self._raise = raise_on_get

    async def get_tools(self) -> dict[str, _ToolStub]:
        if self._raise:
            raise RuntimeError("Failed to get tools")
        return self._tools


class _FnStub:

    def __init__(self, description: str):
        self.description = description


class _GroupInstanceStub:

    def __init__(self, client: _ClientStub, functions_map: dict[str, _FnStub]):
        # Reuse the pre-established client session on the group, like runtime
        self.mcp_client = client
        self._functions_map = functions_map

    async def get_accessible_functions(self, filter_fn=None) -> dict[str, _FnStub]:
        return self._functions_map


class _ConfiguredGroupStub:

    def __init__(self, config, instance):
        self.config = config
        self.instance = instance


class _BuilderStub:

    def __init__(self, groups: dict[str, _ConfiguredGroupStub]):
        # FastAPI worker inspects this internal mapping
        self._function_groups = groups


@pytest_asyncio.fixture
async def app_worker():
    cfg = Config()
    worker = FastApiFrontEndPluginWorker(cfg)
    app = FastAPI()
    worker.set_cors_config(app)
    return app, worker


@pytest.mark.asyncio
async def test_mcp_client_tool_list_success_with_alias(app_worker):
    app, worker = app_worker

    # Build MCP client config with alias override
    from nat.plugins.mcp.client_config import MCPClientConfig
    from nat.plugins.mcp.client_config import MCPServerConfig
    from nat.plugins.mcp.client_config import MCPToolOverrideConfig

    server_cfg = MCPServerConfig(transport="streamable-http", url="http://localhost:9901/mcp")
    cfg = MCPClientConfig(
        server=server_cfg,
        tool_overrides={"orig_tool": MCPToolOverrideConfig(alias="alias_tool", description="Overridden desc")})

    # Server exposes the original tool name
    client = _ClientStub("streamable-http:http://localhost:9901/mcp", {"orig_tool": _ToolStub("Server Desc")})

    # Workflow configured function uses the alias name
    group_name = "mcp_group"
    group_instance = _GroupInstanceStub(client, {f"{group_name}.alias_tool": _FnStub("Overridden desc")})
    configured_group = _ConfiguredGroupStub(cfg, group_instance)
    builder = _BuilderStub({group_name: configured_group})

    await worker.add_mcp_client_tool_list_route(app, builder)  # register endpoint

    with TestClient(app) as client_http:
        resp = client_http.get("/mcp/client/tool/list")
        assert resp.status_code == 200
        data = resp.json()
        assert "mcp_clients" in data
        assert len(data["mcp_clients"]) == 1
        group = data["mcp_clients"][0]
        assert group["function_group"] == group_name
        assert group["server"].startswith("streamable-http:")
        assert group["session_healthy"] is True
        assert group["total_tools"] == 1
        assert group["available_tools"] == 1
        assert len(group["tools"]) == 1
        tool = group["tools"][0]
        assert tool["name"] == "alias_tool"
        assert tool["available"] is True
        assert tool["server"].startswith("streamable-http:")
        # Prefer workflow/override description
        assert tool["description"] == "Overridden desc"


@pytest.mark.asyncio
async def test_mcp_client_tool_list_unhealthy_marks_unavailable(app_worker):
    app, worker = app_worker

    from nat.plugins.mcp.client_config import MCPClientConfig
    from nat.plugins.mcp.client_config import MCPServerConfig

    server_cfg = MCPServerConfig(transport="streamable-http", url="http://localhost:9901/mcp")
    cfg = MCPClientConfig(server=server_cfg)

    # Simulate connection failure
    client = _ClientStub("streamable-http:http://localhost:9901/mcp", {}, raise_on_get=True)

    group_name = "mcp_math"
    group_instance = _GroupInstanceStub(client,
                                        {
                                            f"{group_name}.calculator_add": _FnStub("Add"),
                                            f"{group_name}.calculator_subtract": _FnStub("Subtract"),
                                        })
    configured_group = _ConfiguredGroupStub(cfg, group_instance)
    builder = _BuilderStub({group_name: configured_group})

    await worker.add_mcp_client_tool_list_route(app, builder)

    with TestClient(app) as client_http:
        resp = client_http.get("/mcp/client/tool/list")
        assert resp.status_code == 200
        data = resp.json()
        group = data["mcp_clients"][0]
        assert group["function_group"] == group_name
        assert group["session_healthy"] is False
        assert group["total_tools"] == 2
        assert group["available_tools"] == 0
        assert len(group["tools"]) == 2
        assert all(t["available"] is False for t in group["tools"])
