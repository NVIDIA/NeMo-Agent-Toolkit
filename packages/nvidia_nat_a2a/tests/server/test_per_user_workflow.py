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

import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
import uvicorn
from pydantic import BaseModel
from starlette.requests import Request

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.data_models.function import FunctionBaseConfig
from nat.plugins.a2a.server.agent_executor_adapter import NATWorkflowAgentExecutor
from nat.plugins.a2a.server.call_context import NATCallContextBuilder
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig
from nat.plugins.a2a.server.front_end_plugin import A2AFrontEndPlugin
from nat.plugins.a2a.server.front_end_plugin_worker import A2AFrontEndPluginWorker
from nat.runtime.user_manager import UserManager

SUBJECT = "alice@example.com"


class _Input(BaseModel):
    message: str


class _Output(BaseModel):
    result: str


class PerUserWorkflowConfig(FunctionBaseConfig, name="per_user_a2a_test_workflow"):
    pass


class SharedWorkflowConfig(FunctionBaseConfig, name="shared_a2a_test_workflow"):
    pass


def _jwt(subject: str) -> str:
    """Build an unsigned JWT carrying `subject`, which is all identity resolution reads."""

    def segment(payload: dict) -> str:
        raw = json.dumps(payload).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{segment({'alg': 'none', 'typ': 'JWT'})}.{segment({'sub': subject})}.c2lnbmF0dXJl"


def _request(subject: str | None) -> Request:
    """Build a request shaped like one OAuth2ValidationMiddleware has passed."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "query_string": b"",
        "headers": [(b"authorization", f"Bearer {_jwt(SUBJECT)}".encode())],
        "state": {},
    }
    request = Request(scope)
    if subject is not None:
        request.state.oauth_user = subject
    return request


@pytest.fixture(name="registered_workflows", scope="module")
def fixture_registered_workflows():
    """Register the test workflows in a pushed registry so they do not leak."""

    @register_per_user_function(config_type=PerUserWorkflowConfig, input_type=_Input, single_output_type=_Output)
    async def _build_per_user(_config: PerUserWorkflowConfig, _builder: Builder):

        async def _impl(inp: _Input) -> _Output:
            return _Output(result=f"per-user: {inp.message}")

        yield FunctionInfo.from_fn(_impl)

    @register_function(config_type=SharedWorkflowConfig)
    async def _build_shared(_config: SharedWorkflowConfig, _builder: Builder):

        async def _impl(inp: _Input) -> _Output:
            return _Output(result=f"shared: {inp.message}")

        yield FunctionInfo.from_fn(_impl)


def _config(workflow) -> Config:
    return Config(general=GeneralConfig(front_end=A2AFrontEndConfig(
        name="Test Agent", description="Test agent", host="localhost", port=10001, version="1.0.0")),
                  workflow=workflow)


@pytest.fixture(name="per_user_config")
def fixture_per_user_config(registered_workflows) -> Config:
    """A2A server config whose workflow is per-user."""
    return _config(PerUserWorkflowConfig())


@pytest.fixture(name="shared_config")
def fixture_shared_config(registered_workflows) -> Config:
    """A2A server config whose workflow is shared."""
    return _config(SharedWorkflowConfig())


class TestPerUserWorkflowStartup:
    """The A2A front end must serve per-user workflows, not die building them."""

    async def test_per_user_workflow_server_starts(self, per_user_config, monkeypatch):
        """Startup used to raise "Must set a workflow before building".

        The shared builder leaves a per-user workflow unset on purpose, but the front
        end built it anyway. MCP authentication prescribes exactly this configuration.
        """

        async def _no_serve(_self):
            return None

        monkeypatch.setattr(uvicorn.Server, "serve", _no_serve)

        await A2AFrontEndPlugin(full_config=per_user_config).run()

    async def test_shared_workflow_still_advertises_its_skills(self, shared_config, monkeypatch):
        """Guards the opposite regression: never skipping the build for shared workflows."""
        captured = {}

        async def _no_serve(_self):
            return None

        original = A2AFrontEndPluginWorker.create_agent_card

        async def _capture(self, workflow):
            captured["workflow"] = workflow
            return await original(self, workflow)

        monkeypatch.setattr(uvicorn.Server, "serve", _no_serve)
        monkeypatch.setattr(A2AFrontEndPluginWorker, "create_agent_card", _capture)

        await A2AFrontEndPlugin(full_config=shared_config).run()

        assert captured["workflow"] is not None

    async def test_per_user_server_wires_in_the_nat_call_context_builder(self, per_user_config, monkeypatch):
        """Without this builder the context is unauthenticated and per-user has no user."""
        captured = {}

        async def _no_serve(_self):
            return None

        original = A2AFrontEndPluginWorker.create_a2a_server

        def _capture(self, agent_card, agent_executor):
            server = original(self, agent_card, agent_executor)
            captured["server"] = server
            return server

        monkeypatch.setattr(uvicorn.Server, "serve", _no_serve)
        monkeypatch.setattr(A2AFrontEndPluginWorker, "create_a2a_server", _capture)

        await A2AFrontEndPlugin(full_config=per_user_config).run()

        assert isinstance(captured["server"]._context_builder, NATCallContextBuilder)

    async def test_per_user_session_manager_reaps_and_shuts_down(self, per_user_config):
        """A per-user SessionManager must start its reaper, or builders accumulate forever."""
        from nat.builder.workflow_builder import WorkflowBuilder

        worker = A2AFrontEndPluginWorker(per_user_config)

        async with WorkflowBuilder.from_config(config=per_user_config) as builder:
            session_manager = await worker.create_session_manager(builder)

            assert session_manager.is_workflow_per_user
            assert session_manager._per_user_builders_cleanup_task is not None

            cleanup_task = session_manager._per_user_builders_cleanup_task

            await worker.cleanup()

            assert cleanup_task.done()

    async def test_agent_card_without_shared_workflow_advertises_no_skills(self, per_user_config):
        """A per-user workflow has no shared instance to introspect for skills."""
        agent_card = await A2AFrontEndPluginWorker(per_user_config).create_agent_card(None)

        assert agent_card.skills == []
        assert agent_card.name == "Test Agent"


class TestPerUserRequestIdentity:
    """Each request must resolve to the user its per-user workflow is built for."""

    def test_executor_init_does_not_touch_shared_workflow(self, per_user_config):
        """`SessionManager.workflow` raises for per-user, so init must not read it."""
        session_manager = MagicMock()
        session_manager.config = per_user_config
        type(session_manager).workflow = property(lambda _self:
                                                  (_ for _ in ()).throw(ValueError("Workflow is per-user.")))

        NATWorkflowAgentExecutor(session_manager)

    async def test_executor_passes_resolved_user_to_the_session(self, per_user_config):
        """The resolved user must reach `session()`, which is what builds per user."""
        session = MagicMock()
        runner = MagicMock()
        runner.result = AsyncMock(return_value="done")
        session.run.return_value.__aenter__ = AsyncMock(return_value=runner)
        session.run.return_value.__aexit__ = AsyncMock(return_value=False)

        session_manager = MagicMock()
        session_manager.config = per_user_config
        session_manager.session.return_value.__aenter__ = AsyncMock(return_value=session)
        session_manager.session.return_value.__aexit__ = AsyncMock(return_value=False)

        context = MagicMock()
        context.get_user_input.return_value = "hello"
        context.context_id = "ctx-1"
        context.task_id = "task-1"
        context.call_context = SimpleNamespace(user=SimpleNamespace(is_authenticated=True, user_name="alice"))

        event_queue = MagicMock()
        event_queue.enqueue_event = AsyncMock()

        await NATWorkflowAgentExecutor(session_manager).execute(context, event_queue)

        session_manager.session.assert_called_once_with(user_id="alice")

    @pytest.mark.parametrize(
        "call_context, expected",
        [
            (None, None),
            (SimpleNamespace(user=None), None),
            (SimpleNamespace(user=SimpleNamespace(is_authenticated=False, user_name="")), None),
            (SimpleNamespace(user=SimpleNamespace(is_authenticated=True, user_name="alice")), "alice"),
        ],
        ids=["no-call-context", "no-user", "unauthenticated", "authenticated"],
    )
    def test_resolve_user_id(self, call_context, expected):
        """Only an authenticated call context yields a user id."""
        assert NATWorkflowAgentExecutor._resolve_user_id(SimpleNamespace(call_context=call_context)) == expected

    def test_call_context_builder_uses_nat_user_id_not_the_raw_subject(self):
        """A2A must agree with the other front ends on what a user id is.

        `user_id` keys stored OAuth tokens, so returning the raw subject here would
        make the same person miss their own cached credentials across front ends.
        """
        context = NATCallContextBuilder().build(_request(SUBJECT))

        expected = UserManager.extract_user_from_connection(_request(SUBJECT)).get_user_id()

        assert context.user.is_authenticated
        assert context.user.user_name == expected
        assert context.user.user_name != SUBJECT

    def test_call_context_builder_leaves_unverified_requests_alone(self):
        """A request the middleware has not verified stays unauthenticated.

        The bearer token is present either way, so this pins that the builder trusts
        `request.state.oauth_user` rather than the raw header.
        """
        context = NATCallContextBuilder().build(_request(None))

        assert not context.user.is_authenticated
