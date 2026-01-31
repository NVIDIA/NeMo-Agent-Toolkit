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
"""
E2E tests for FastAPI integration with per-user workflows.

Tests the following:
1. SessionManager.create() integration
2. Session passing to response helpers
3. Per-user workflow isolation via HTTP endpoints
4. Cleanup of session managers
"""

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport
from httpx import AsyncClient
from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.data_models.function import FunctionBaseConfig
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker


# ============= Test Schemas =============
class SimpleInput(BaseModel):
    message: str = Field(description="Input message")


class SimpleOutput(BaseModel):
    response: str = Field(description="Output response")


class CounterInput(BaseModel):
    action: str = Field(description="Either 'increment' or 'get'")


class CounterOutput(BaseModel):
    count: int = Field(description="Current count value")
    user_id: str = Field(default="", description="User ID that owns this counter")


# ============= Test Configs =============
class SharedWorkflowConfig(FunctionBaseConfig, name="shared_workflow_fastapi_test"):
    """A shared workflow config for FastAPI testing."""
    pass


class PerUserCounterWorkflowConfig(FunctionBaseConfig, name="per_user_counter_workflow_fastapi"):
    """A per-user counter workflow config for FastAPI testing."""
    initial_value: int = 0


# ============= Register Test Components =============
@pytest.fixture(scope="module", autouse=True)
def _register_components():
    """Register all test components."""

    # Shared workflow - simple echo
    @register_function(config_type=SharedWorkflowConfig)
    async def shared_workflow(config: SharedWorkflowConfig, builder: Builder):

        async def _impl(inp: SimpleInput) -> SimpleOutput:
            return SimpleOutput(response=f"echo: {inp.message}")

        yield FunctionInfo.from_fn(_impl)

    # Per-user counter workflow - maintains state per user
    @register_per_user_function(config_type=PerUserCounterWorkflowConfig,
                                input_type=CounterInput,
                                single_output_type=CounterOutput)
    async def per_user_counter_workflow(config: PerUserCounterWorkflowConfig, builder: Builder):
        from nat.builder.context import Context

        # This state is unique per user!
        counter_state = {"count": config.initial_value}

        async def _counter(inp: CounterInput) -> CounterOutput:
            if inp.action == "increment":
                counter_state["count"] += 1

            # Try to get user_id from context
            try:
                ctx = Context.get()
                user_id = ""
                if ctx.metadata and hasattr(ctx.metadata, '_request') and ctx.metadata._request.cookies:
                    user_id = ctx.metadata._request.cookies.get("nat-session", "")
            except Exception:
                user_id = ""

            return CounterOutput(count=counter_state["count"], user_id=user_id)

        yield FunctionInfo.from_fn(_counter)


# ============= Test Fixtures =============
def create_shared_workflow_config() -> Config:
    """Create a config with shared workflow."""
    front_end = FastApiFrontEndConfig(root_path="",
                                      workflow=FastApiFrontEndConfig.EndpointBase(path="/generate",
                                                                                  method="POST",
                                                                                  description="Test endpoint"))
    return Config(general=GeneralConfig(front_end=front_end), workflow=SharedWorkflowConfig())


def create_per_user_workflow_config() -> Config:
    """Create a config with per-user workflow."""
    front_end = FastApiFrontEndConfig(root_path="",
                                      workflow=FastApiFrontEndConfig.EndpointBase(
                                          path="/counter", method="POST", description="Per-user counter endpoint"))
    return Config(general=GeneralConfig(front_end=front_end), workflow=PerUserCounterWorkflowConfig(initial_value=0))


# ============= Tests =============
class TestSessionManagerCreate:
    """Tests for SessionManager.create() in FastAPI context."""

    async def test_create_session_manager_shared_workflow(self):
        """Test _create_session_manager with shared workflow."""
        config = create_shared_workflow_config()
        worker = FastApiFrontEndPluginWorker(config)

        async with WorkflowBuilder.from_config(config) as builder:
            sm = await worker._create_session_manager(builder)

            assert sm is not None
            assert sm in worker._session_managers
            assert sm.is_workflow_per_user is False
            assert sm._shared_workflow is not None

            # Cleanup
            await worker.cleanup_session_managers()

    async def test_create_session_manager_per_user_workflow(self):
        """Test _create_session_manager with per-user workflow."""
        config = create_per_user_workflow_config()
        worker = FastApiFrontEndPluginWorker(config)

        async with WorkflowBuilder.from_config(config) as builder:
            sm = await worker._create_session_manager(builder)

            assert sm is not None
            assert sm in worker._session_managers
            assert sm.is_workflow_per_user is True
            assert sm._shared_workflow is None

            # Cleanup
            await worker.cleanup_session_managers()

    async def test_create_multiple_session_managers(self):
        """Test creating multiple session managers."""
        config = create_shared_workflow_config()
        worker = FastApiFrontEndPluginWorker(config)

        async with WorkflowBuilder.from_config(config) as builder:
            sm1 = await worker._create_session_manager(builder)
            sm2 = await worker._create_session_manager(builder, entry_function=None)

            assert len(worker._session_managers) == 2
            assert sm1 in worker._session_managers
            assert sm2 in worker._session_managers

            # Cleanup
            await worker.cleanup_session_managers()


class TestSessionManagerCleanup:
    """Tests for SessionManager cleanup."""

    async def test_cleanup_session_managers(self):
        """Test cleanup_session_managers clears all managers."""
        config = create_shared_workflow_config()
        worker = FastApiFrontEndPluginWorker(config)

        async with WorkflowBuilder.from_config(config) as builder:
            await worker._create_session_manager(builder)
            await worker._create_session_manager(builder)

            assert len(worker._session_managers) == 2

            await worker.cleanup_session_managers()

            assert len(worker._session_managers) == 0

    async def test_cleanup_per_user_session_managers(self):
        """Test cleanup of per-user session managers."""
        config = create_per_user_workflow_config()
        worker = FastApiFrontEndPluginWorker(config)

        async with WorkflowBuilder.from_config(config) as builder:
            sm = await worker._create_session_manager(builder)

            # Create a per-user session to populate the cache
            async with sm.session(user_id="test_user"):
                pass

            assert "test_user" in sm._per_user_builders

            await worker.cleanup_session_managers()

            assert len(worker._session_managers) == 0


class TestSharedWorkflowEndpoint:
    """Tests for HTTP endpoints with shared workflow."""

    async def test_post_endpoint_shared_workflow(self):
        """Test POST endpoint with shared workflow."""
        config = create_shared_workflow_config()
        worker = FastApiFrontEndPluginWorker(config)
        app = worker.build_app()

        # Use LifespanManager to properly trigger lifespan events (route registration)
        async with LifespanManager(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/generate", json={"message": "hello"})

                assert response.status_code == 200
                data = response.json()
                assert data["response"] == "echo: hello"

    async def test_multiple_requests_shared_workflow(self):
        """Test multiple requests share the same workflow."""
        config = create_shared_workflow_config()
        worker = FastApiFrontEndPluginWorker(config)
        app = worker.build_app()

        async with LifespanManager(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response1 = await client.post("/generate", json={"message": "first"})
                response2 = await client.post("/generate", json={"message": "second"})

                assert response1.status_code == 200
                assert response2.status_code == 200
                assert response1.json()["response"] == "echo: first"
                assert response2.json()["response"] == "echo: second"


class TestPerUserWorkflowEndpoint:
    """Tests for HTTP endpoints with per-user workflow."""

    async def test_post_endpoint_per_user_workflow(self):
        """Test POST endpoint with per-user workflow."""
        config = create_per_user_workflow_config()
        worker = FastApiFrontEndPluginWorker(config)
        app = worker.build_app()

        async with LifespanManager(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                # Set session cookie on client
                client.cookies.set("nat-session", "user123")
                response = await client.post("/counter", json={"action": "get"})

                assert response.status_code == 200
                data = response.json()
                assert data["count"] == 0

    async def test_per_user_isolation(self):
        """Test that different users have isolated state."""
        config = create_per_user_workflow_config()
        worker = FastApiFrontEndPluginWorker(config)
        app = worker.build_app()

        async with LifespanManager(app):
            # Use separate clients for different users to properly isolate cookies
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as alice_client:
                alice_client.cookies.set("nat-session", "alice")

                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as bob_client:
                    bob_client.cookies.set("nat-session", "bob")

                    # User 1 increments counter twice
                    await alice_client.post("/counter", json={"action": "increment"})
                    response1 = await alice_client.post("/counter", json={"action": "increment"})
                    assert response1.json()["count"] == 2

                    # User 2 should have fresh counter at 0
                    response2 = await bob_client.post("/counter", json={"action": "get"})
                    assert response2.json()["count"] == 0

                    # User 2 increments once
                    response3 = await bob_client.post("/counter", json={"action": "increment"})
                    assert response3.json()["count"] == 1

                    # User 1 counter should still be at 2
                    response4 = await alice_client.post("/counter", json={"action": "get"})
                    assert response4.json()["count"] == 2

    async def test_per_user_state_persists_across_requests(self):
        """Test that per-user state persists across multiple requests."""
        config = create_per_user_workflow_config()
        worker = FastApiFrontEndPluginWorker(config)
        app = worker.build_app()

        async with LifespanManager(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                client.cookies.set("nat-session", "persistent_user")

                # Increment 5 times
                for i in range(5):
                    response = await client.post("/counter", json={"action": "increment"})
                    assert response.json()["count"] == i + 1

                # Final get should show 5
                response = await client.post("/counter", json={"action": "get"})
                assert response.json()["count"] == 5


class TestSessionManagerSchemas:
    """Tests for schema access in add_route."""

    async def test_shared_workflow_schema_access(self):
        """Test schema access for shared workflow in add_route."""
        config = create_shared_workflow_config()
        worker = FastApiFrontEndPluginWorker(config)

        async with WorkflowBuilder.from_config(config) as builder:
            sm = await worker._create_session_manager(builder)

            # Should access workflow directly for shared
            assert sm.is_workflow_per_user is False
            workflow = sm.workflow
            assert workflow.input_schema == SimpleInput
            assert workflow.single_output_schema == SimpleOutput

            # Cleanup
            await worker.cleanup_session_managers()

    async def test_per_user_workflow_schema_access(self):
        """Test schema access for per-user workflow in add_route."""
        config = create_per_user_workflow_config()
        worker = FastApiFrontEndPluginWorker(config)

        async with WorkflowBuilder.from_config(config) as builder:
            sm = await worker._create_session_manager(builder)

            # Should use accessor methods for per-user
            assert sm.is_workflow_per_user is True
            assert sm.get_workflow_input_schema() == CounterInput
            assert sm.get_workflow_single_output_schema() == CounterOutput

            # Direct workflow access should raise
            with pytest.raises(ValueError, match="Workflow is per-user"):
                _ = sm.workflow

            # Cleanup
            await worker.cleanup_session_managers()


def create_per_user_workflow_config_with_monitoring() -> Config:
    """Create a config with per-user workflow and monitoring enabled."""
    front_end = FastApiFrontEndConfig(root_path="",
                                      workflow=FastApiFrontEndConfig.EndpointBase(
                                          path="/counter", method="POST", description="Per-user counter endpoint"))
    return Config(general=GeneralConfig(front_end=front_end, enable_per_user_monitoring=True),
                  workflow=PerUserCounterWorkflowConfig(initial_value=0))


class TestPerUserMonitoringEndpoint:
    """Tests for the /monitor/users endpoint."""

    async def test_monitor_endpoint_disabled_by_default(self):
        """Test that monitoring endpoint is not available when disabled."""
        config = create_per_user_workflow_config()  # monitoring disabled by default
        worker = FastApiFrontEndPluginWorker(config)
        app = worker.build_app()

        async with LifespanManager(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/monitor/users")
                # Endpoint should not exist
                assert response.status_code == 404

    async def test_monitor_endpoint_enabled(self):
        """Test that monitoring endpoint is available when enabled."""
        config = create_per_user_workflow_config_with_monitoring()
        worker = FastApiFrontEndPluginWorker(config)
        app = worker.build_app()

        async with LifespanManager(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/monitor/users")
                assert response.status_code == 200
                data = response.json()
                assert "timestamp" in data
                assert "total_active_users" in data
                assert "users" in data
                assert data["total_active_users"] == 0
                assert data["users"] == []

    async def test_monitor_endpoint_shows_active_users(self):
        """Test that monitoring endpoint shows metrics for active users."""
        config = create_per_user_workflow_config_with_monitoring()
        worker = FastApiFrontEndPluginWorker(config)
        app = worker.build_app()

        async with LifespanManager(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                # Create some user activity first
                client.cookies.set("nat-session", "monitor_test_user")
                await client.post("/counter", json={"action": "increment"})
                await client.post("/counter", json={"action": "increment"})

                # Now check monitoring endpoint
                response = await client.get("/monitor/users")
                assert response.status_code == 200
                data = response.json()

                assert data["total_active_users"] == 1
                assert len(data["users"]) == 1

                user_metrics = data["users"][0]
                assert user_metrics["user_id"] == "monitor_test_user"

                # Check session metrics
                assert "session" in user_metrics
                assert user_metrics["session"]["ref_count"] >= 0

                # Check request metrics
                assert "requests" in user_metrics
                assert user_metrics["requests"]["total_requests"] == 2

                # Check memory metrics
                assert "memory" in user_metrics
                assert "per_user_functions_count" in user_metrics["memory"]

    async def test_monitor_endpoint_filter_by_user_id(self):
        """Test that monitoring endpoint can filter by user_id."""
        config = create_per_user_workflow_config_with_monitoring()
        worker = FastApiFrontEndPluginWorker(config)
        app = worker.build_app()

        async with LifespanManager(app):
            # Create activity for two users
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as alice_client:
                alice_client.cookies.set("nat-session", "alice")
                await alice_client.post("/counter", json={"action": "increment"})

                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as bob_client:
                    bob_client.cookies.set("nat-session", "bob")
                    await bob_client.post("/counter", json={"action": "increment"})

                    # Filter for alice only
                    response = await alice_client.get("/monitor/users", params={"user_id": "alice"})
                    assert response.status_code == 200
                    data = response.json()

                    assert data["total_active_users"] == 1
                    assert len(data["users"]) == 1
                    assert data["users"][0]["user_id"] == "alice"

                    # Filter for non-existent user
                    response = await alice_client.get("/monitor/users", params={"user_id": "nonexistent"})
                    assert response.status_code == 200
                    data = response.json()
                    assert data["total_active_users"] == 0
                    assert data["users"] == []

    async def test_monitor_endpoint_tracks_errors(self):
        """Test that monitoring endpoint tracks error counts."""
        # This test would require a workflow that can produce errors
        # For now, we just verify the error_count field exists
        config = create_per_user_workflow_config_with_monitoring()
        worker = FastApiFrontEndPluginWorker(config)
        app = worker.build_app()

        async with LifespanManager(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                client.cookies.set("nat-session", "error_test_user")
                await client.post("/counter", json={"action": "get"})

                response = await client.get("/monitor/users")
                data = response.json()

                assert len(data["users"]) == 1
                assert "error_count" in data["users"][0]["requests"]
                # No errors in this simple case
                assert data["users"][0]["requests"]["error_count"] == 0
