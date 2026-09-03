# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.data_models.function import FunctionBaseConfig
from nat.plugins.mcp.server.front_end_config import MCPFrontEndConfig
from nat.plugins.mcp.server.front_end_plugin import MCPFrontEndPlugin
from nat.plugins.mcp.server.front_end_plugin_worker import MCPFrontEndPluginWorker
from nat.runtime.session import SessionManager


class _Input(BaseModel):
    message: str


class _Output(BaseModel):
    result: str


class PerUserMCPWorkflowConfig(FunctionBaseConfig, name="per_user_mcp_test_workflow"):
    """Per-user workflow config for MCP front-end tests."""


class SharedMCPWorkflowConfig(FunctionBaseConfig, name="shared_mcp_test_workflow"):
    """Shared workflow config for MCP front-end tests."""


@pytest.fixture(name="registered_workflows", scope="module")
def fixture_registered_workflows():
    """Register test workflows in a pushed registry so they do not leak."""

    @register_per_user_function(config_type=PerUserMCPWorkflowConfig, input_type=_Input, single_output_type=_Output)
    async def _build_per_user(_config: PerUserMCPWorkflowConfig, _builder: Builder):

        async def _impl(inp: _Input) -> _Output:
            return _Output(result=f"per-user: {inp.message}")

        yield FunctionInfo.from_fn(_impl)

    @register_function(config_type=SharedMCPWorkflowConfig)
    async def _build_shared(_config: SharedMCPWorkflowConfig, _builder: Builder):

        async def _impl(inp: _Input) -> _Output:
            return _Output(result=f"shared: {inp.message}")

        yield FunctionInfo.from_fn(_impl)


def _config(workflow) -> Config:
    return Config(
        general=GeneralConfig(front_end=MCPFrontEndConfig(
            name="Test MCP Server",
            host="localhost",
            port=9902,
            debug=False,
            log_level="INFO",
        )),
        workflow=workflow,
    )


@pytest.fixture(name="per_user_config")
def fixture_per_user_config(registered_workflows) -> Config:
    return _config(PerUserMCPWorkflowConfig())


@pytest.fixture(name="shared_config")
def fixture_shared_config(registered_workflows) -> Config:
    return _config(SharedMCPWorkflowConfig())


class TestPerUserWorkflowStartup:
    """The MCP front end must serve per-user workflows, not die building them."""

    async def test_per_user_workflow_server_starts(self, per_user_config, monkeypatch):
        """Startup used to raise "Must set a workflow before building"."""

        async def _no_serve(_self):
            return None

        monkeypatch.setattr("mcp.server.fastmcp.server.FastMCP.run_streamable_http_async", _no_serve)

        await MCPFrontEndPlugin(full_config=per_user_config).run()

    async def test_shared_workflow_still_builds(self, shared_config, monkeypatch):
        captured = {}

        original_create = SessionManager.create

        async def _capture_create(*args, **kwargs):
            session_manager = await original_create(*args, **kwargs)
            if not session_manager.is_workflow_per_user:
                captured["workflow"] = session_manager.workflow
            return session_manager

        async def _no_serve(_self):
            return None

        monkeypatch.setattr(SessionManager, "create", _capture_create)
        monkeypatch.setattr("mcp.server.fastmcp.server.FastMCP.run_streamable_http_async", _no_serve)

        await MCPFrontEndPlugin(full_config=shared_config).run()

        assert captured["workflow"] is not None

    async def test_per_user_session_manager_reaps_and_shuts_down(self, per_user_config):
        worker = MCPFrontEndPluginWorker(per_user_config)

        async with WorkflowBuilder.from_config(config=per_user_config) as builder:
            mcp = await worker.create_mcp_server()
            await worker._default_add_routes(mcp, builder)

            assert len(worker._session_managers) == 1
            session_manager = worker._session_managers[0]
            assert session_manager.is_workflow_per_user
            assert session_manager._per_user_builders_cleanup_task is not None

            cleanup_task = session_manager._per_user_builders_cleanup_task
            await worker.cleanup()
            assert cleanup_task.done()

    async def test_register_function_skips_shared_workflow_lookup(self, per_user_config, monkeypatch):
        from nat.plugins.mcp.server import tool_converter

        worker = MCPFrontEndPluginWorker(per_user_config)

        async with WorkflowBuilder.from_config(config=per_user_config) as builder:
            session_manager = await SessionManager.create(config=per_user_config, shared_builder=builder)
            mcp = await worker.create_mcp_server()

            get_schema = MagicMock(return_value=_Input)
            monkeypatch.setattr(session_manager, "get_workflow_input_schema", get_schema)

            tool_converter.register_function_with_mcp(mcp, "per_user_mcp_test_workflow", session_manager)

            get_schema.assert_called_once()


class TestWorkerCleanup:
    """Worker cleanup must shut down every manager and clear tracking."""

    async def test_cleanup_shuts_down_all_managers_after_shutdown_failure(self, per_user_config):
        worker = MCPFrontEndPluginWorker(per_user_config)
        failing = MagicMock()
        failing.shutdown = AsyncMock(side_effect=RuntimeError("boom"))
        succeeding = MagicMock()
        succeeding.shutdown = AsyncMock()
        worker._session_managers = [failing, succeeding]

        with pytest.raises(RuntimeError, match="boom"):
            await worker.cleanup()

        failing.shutdown.assert_awaited_once()
        succeeding.shutdown.assert_awaited_once()
        assert worker._session_managers == []


class TestPerUserRequestIdentity:
    """Per-user tool calls must reach session() with a resolved user id."""

    async def test_run_through_session_manager_uses_context_user_id(self, monkeypatch):
        from nat.plugins.mcp.server.tool_converter import _run_through_session_manager

        session_manager = MagicMock()
        session_manager.is_workflow_per_user = True
        session_manager.session = MagicMock()
        session = MagicMock()
        runner = MagicMock()
        runner.result = AsyncMock(return_value="ok")
        session.run.return_value.__aenter__ = AsyncMock(return_value=runner)
        session.run.return_value.__aexit__ = AsyncMock(return_value=False)
        session_manager.session.return_value.__aenter__ = AsyncMock(return_value=session)
        session_manager.session.return_value.__aexit__ = AsyncMock(return_value=False)

        context = SimpleNamespace(user_id="alice")
        monkeypatch.setattr("nat.builder.context.Context.get", lambda: context)

        payload = _Input(message="hello")
        result = await _run_through_session_manager(session_manager, payload)

        assert result == "ok"
        session_manager.session.assert_called_once_with(user_id="alice", http_connection=None)

    async def test_run_through_session_manager_resolves_user_from_mcp_request(self, monkeypatch):
        from nat.plugins.mcp.server.tool_converter import _run_through_session_manager

        session_manager = MagicMock()
        session_manager.is_workflow_per_user = True
        session_manager.session = MagicMock()
        session = MagicMock()
        runner = MagicMock()
        runner.result = AsyncMock(return_value="ok")
        session.run.return_value.__aenter__ = AsyncMock(return_value=runner)
        session.run.return_value.__aexit__ = AsyncMock(return_value=False)
        session_manager.session.return_value.__aenter__ = AsyncMock(return_value=session)
        session_manager.session.return_value.__aexit__ = AsyncMock(return_value=False)

        context = SimpleNamespace(user_id=None)
        monkeypatch.setattr("nat.builder.context.Context.get", lambda: context)

        request = MagicMock()
        ctx = SimpleNamespace(request_context=SimpleNamespace(request=request))
        user_info = MagicMock()
        user_info.get_user_id.return_value = "bob"
        monkeypatch.setattr(
            "nat.runtime.user_manager.UserManager.extract_user_from_connection",
            MagicMock(return_value=user_info),
        )

        payload = _Input(message="hello")
        result = await _run_through_session_manager(session_manager, payload, ctx=ctx)

        assert result == "ok"
        session_manager.session.assert_called_once_with(user_id="bob", http_connection=request)
