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

import asyncio
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.builder.context import ContextState
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.data_models.runtime_enum import RuntimeTypeEnum
from nat.runtime.session import PerUserBuilderInfo
from nat.runtime.session import Session
from nat.runtime.session import SessionManager

# Rebuild model to resolve forward references (PerUserWorkflowBuilder, Workflow)
PerUserBuilderInfo.model_rebuild()


class MockInputSchema(BaseModel):
    message: str


class MockOutputSchema(BaseModel):
    response: str


class MockWorkflow:
    """Mock workflow for testing."""

    def __init__(self):
        self.config = MagicMock(spec=Config)
        self.input_schema = MockInputSchema
        self.single_output_schema = MockOutputSchema
        self.streaming_output_schema = MockOutputSchema

    def run(self, message, runtime_type=RuntimeTypeEnum.RUN_OR_SERVE):
        """Return an async context manager for run."""
        runner = MagicMock()
        runner.result = AsyncMock(return_value=MockOutputSchema(response="test"))

        class MockContext:

            async def __aenter__(self):
                return runner

            async def __aexit__(self, *args):
                pass

        return MockContext()


class MockWorkflowBuilder:
    """Mock workflow builder for testing."""

    def __init__(self):
        self._functions = {}
        self._function_groups = {}
        self._llm_providers = {}

    def get_function(self, name):
        return self._functions.get(name)

    def get_function_group(self, name):
        return self._function_groups.get(name)

    def get_llm_provider(self, name):
        return self._llm_providers.get(name)


class MockPerUserWorkflowBuilder:
    """Mock per-user workflow builder for testing."""

    def __init__(self, user_id, shared_builder):
        self.user_id = user_id
        self._shared_builder = shared_builder
        self._entered = False
        self._exited = False

    async def __aenter__(self):
        self._entered = True
        return self

    async def __aexit__(self, *args):
        self._exited = True

    async def populate_builder(self, config):
        pass

    async def build(self, entry_function: str | None = None):
        """Build workflow with optional entry function."""
        return MockWorkflow()


def create_mock_config(is_per_user: bool = False) -> Config:
    """Create a mock config for testing."""
    config = MagicMock(spec=Config)
    config.general = MagicMock(spec=GeneralConfig)
    config.general.per_user_workflow_timeout = timedelta(minutes=30)
    config.general.per_user_workflow_cleanup_interval = timedelta(minutes=5)
    config.workflow = MagicMock()
    return config


def create_mock_function_registration(is_per_user: bool = False):
    """Create a mock function registration info."""
    registration = MagicMock()
    registration.is_per_user = is_per_user
    registration.per_user_function_input_schema = MockInputSchema if is_per_user else None
    registration.per_user_function_single_output_schema = MockOutputSchema if is_per_user else None
    registration.per_user_function_streaming_output_schema = MockOutputSchema if is_per_user else None
    return registration


class TestPerUserBuilderInfo:
    """Tests for PerUserBuilderInfo Pydantic model."""

    def test_per_user_builder_info_creation(self):
        """Test PerUserBuilderInfo can be created with required fields."""
        builder = MockPerUserWorkflowBuilder("user1", MockWorkflowBuilder())
        workflow = MockWorkflow()
        semaphore = asyncio.Semaphore(8)

        info = PerUserBuilderInfo(builder=builder, workflow=workflow, semaphore=semaphore)

        assert info.builder == builder
        assert info.workflow == workflow
        assert info.semaphore == semaphore
        assert info.ref_count == 0
        assert isinstance(info.last_activity, datetime)
        assert isinstance(info.lock, asyncio.Lock)

    def test_per_user_builder_info_ref_count_default(self):
        """Test ref_count defaults to 0."""
        info = PerUserBuilderInfo(builder=MockPerUserWorkflowBuilder("user1", MockWorkflowBuilder()),
                                  workflow=MockWorkflow(),
                                  semaphore=asyncio.Semaphore(8))
        assert info.ref_count == 0

    def test_per_user_builder_info_ref_count_validation(self):
        """Test ref_count cannot be negative."""
        with pytest.raises(ValueError):
            PerUserBuilderInfo(builder=MockPerUserWorkflowBuilder("user1", MockWorkflowBuilder()),
                               workflow=MockWorkflow(),
                               semaphore=asyncio.Semaphore(8),
                               ref_count=-1)


class TestSession:
    """Tests for Session class."""

    def test_session_properties(self):
        """Test Session exposes correct properties."""
        mock_workflow = MockWorkflow()
        mock_session_manager = MagicMock(spec=SessionManager)
        semaphore = asyncio.Semaphore(8)

        session = Session(session_manager=mock_session_manager,
                          workflow=mock_workflow,
                          semaphore=semaphore,
                          user_id="user123")

        assert session.user_id == "user123"
        assert session.workflow == mock_workflow
        assert session.session_manager == mock_session_manager
        assert session._semaphore == semaphore

    def test_session_without_user_id(self):
        """Test Session works without user_id (shared workflow)."""
        session = Session(session_manager=MagicMock(),
                          workflow=MockWorkflow(),
                          semaphore=asyncio.Semaphore(8),
                          user_id=None)

        assert session.user_id is None

    def test_session_with_different_semaphores(self):
        """Test different sessions can have different semaphores for concurrency isolation."""
        semaphore1 = asyncio.Semaphore(4)
        semaphore2 = asyncio.Semaphore(8)

        session1 = Session(session_manager=MagicMock(), workflow=MockWorkflow(), semaphore=semaphore1, user_id="user1")
        session2 = Session(session_manager=MagicMock(), workflow=MockWorkflow(), semaphore=semaphore2, user_id="user2")

        assert session1._semaphore is not session2._semaphore
        assert session1._semaphore == semaphore1
        assert session2._semaphore == semaphore2


class TestSessionManagerInit:
    """Tests for SessionManager initialization."""

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    def test_init_with_shared_workflow(self, mock_registry):
        """Test SessionManager initialization with shared workflow."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=False)

        config = create_mock_config()
        shared_builder = MockWorkflowBuilder()
        shared_workflow = MockWorkflow()

        sm = SessionManager(config=config,
                            shared_builder=shared_builder,
                            entry_function=None,
                            shared_workflow=shared_workflow,
                            max_concurrency=8)

        assert sm.config == config
        assert sm.shared_builder == shared_builder
        assert sm.is_workflow_per_user is False
        assert sm._shared_workflow == shared_workflow
        assert sm._per_user_builders == {}
        assert sm._entry_function is None

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    def test_init_with_per_user_workflow(self, mock_registry):
        """Test SessionManager initialization with per-user workflow."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        config = create_mock_config(is_per_user=True)
        shared_builder = MockWorkflowBuilder()

        sm = SessionManager(
            config=config,
            shared_builder=shared_builder,
            entry_function=None,
            shared_workflow=None,  # No shared workflow for per-user
            max_concurrency=8)

        assert sm.is_workflow_per_user is True
        assert sm._shared_workflow is None
        assert sm._per_user_workflow_input_schema == MockInputSchema
        assert sm._per_user_workflow_single_output_schema == MockOutputSchema
        assert sm._entry_function is None

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    def test_workflow_property_raises_for_per_user(self, mock_registry):
        """Test workflow property raises error for per-user workflows."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=None)

        with pytest.raises(ValueError, match="Workflow is per-user"):
            _ = sm.workflow

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    def test_zero_concurrency_uses_nullcontext(self, mock_registry):
        """Test max_concurrency=0 uses nullcontext instead of Semaphore."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=False)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=MockWorkflow(),
                            max_concurrency=0)

        # Should not be a Semaphore
        assert not isinstance(sm._semaphore, asyncio.Semaphore)


class TestSessionManagerSchemas:
    """Tests for SessionManager schema access methods."""

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    def test_get_workflow_input_schema_shared(self, mock_registry):
        """Test get_workflow_input_schema for shared workflow."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=False)

        workflow = MockWorkflow()
        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=workflow)

        assert sm.get_workflow_input_schema() == MockInputSchema

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    def test_get_workflow_input_schema_per_user(self, mock_registry):
        """Test get_workflow_input_schema for per-user workflow."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=None)

        assert sm.get_workflow_input_schema() == MockInputSchema


class TestSessionManagerRun:
    """Tests for SessionManager.run() method."""

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_run_raises_for_per_user_workflow(self, mock_registry):
        """Test run() raises error for per-user workflows."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=None)

        with pytest.raises(ValueError, match=r"Cannot use SessionManager.run\(\) with per-user workflows"):
            async with sm.run("test message"):
                pass


class TestSessionManagerSession:
    """Tests for SessionManager.session() context manager."""

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_session_shared_workflow(self, mock_registry):
        """Test session() with shared workflow returns Session with shared workflow."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=False)

        shared_workflow = MockWorkflow()
        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=shared_workflow)

        async with sm.session() as session:
            assert isinstance(session, Session)
            assert session.workflow == shared_workflow
            assert session.user_id is None
            # Shared workflow uses SessionManager's semaphore
            assert session._semaphore is sm._semaphore

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_session_per_user_requires_user_id(self, mock_registry):
        """Test session() with per-user workflow requires user_id."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=None)

        with pytest.raises(ValueError, match="user_id is required for per-user workflow but could not be determined"):
            async with sm.session():
                pass

    @patch('nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder', MockPerUserWorkflowBuilder)
    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_session_per_user_with_explicit_user_id(self, mock_registry):
        """Test session() with per-user workflow and explicit user_id."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=None)

        async with sm.session(user_id="user123") as session:
            assert isinstance(session, Session)
            assert session.user_id == "user123"
            # Builder should be cached
            assert "user123" in sm._per_user_builders

    @patch('nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder', MockPerUserWorkflowBuilder)
    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_session_per_user_increments_ref_count(self, mock_registry):
        """Test session() increments ref_count on entry and decrements on exit."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=None)

        async with sm.session(user_id="user123"):
            builder_info = sm._per_user_builders["user123"]
            assert builder_info.ref_count == 1

        # After exit, ref_count should be decremented
        assert sm._per_user_builders["user123"].ref_count == 0

    @patch('nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder', MockPerUserWorkflowBuilder)
    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_session_per_user_reuses_cached_builder(self, mock_registry):
        """Test session() reuses cached per-user builder."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=None)

        # First session creates builder
        async with sm.session(user_id="user123"):
            first_builder = sm._per_user_builders["user123"].builder

        # Second session should reuse same builder
        async with sm.session(user_id="user123"):
            second_builder = sm._per_user_builders["user123"].builder
            assert first_builder is second_builder

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_session_sets_context_vars(self, mock_registry):
        """Test session() properly sets and resets context vars."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=False)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=MockWorkflow())

        ctx_state = ContextState.get()
        original_callback = ctx_state.user_input_callback.get()

        test_callback = AsyncMock()

        async with sm.session(user_input_callback=test_callback):
            assert ctx_state.user_input_callback.get() == test_callback

        # After exit, should be reset
        assert ctx_state.user_input_callback.get() == original_callback


class TestSessionManagerCleanup:
    """Tests for SessionManager per-user builder cleanup."""

    @patch('nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder', MockPerUserWorkflowBuilder)
    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_cleanup_inactive_builders(self, mock_registry):
        """Test _cleanup_inactive_per_user_builders removes old builders."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        config = create_mock_config()
        config.general.per_user_workflow_timeout = timedelta(seconds=1)

        sm = SessionManager(config=config,
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=None)

        # Create a builder
        async with sm.session(user_id="user123"):
            pass

        # Manually set last_activity to past
        sm._per_user_builders["user123"].last_activity = datetime.now() - timedelta(seconds=10)

        # Run cleanup
        cleaned = await sm._cleanup_inactive_per_user_builders()

        assert cleaned == 1
        assert "user123" not in sm._per_user_builders

    @patch('nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder', MockPerUserWorkflowBuilder)
    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_cleanup_skips_active_builders(self, mock_registry):
        """Test cleanup doesn't remove builders with active sessions (ref_count > 0)."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        config = create_mock_config()
        config.general.per_user_workflow_timeout = timedelta(seconds=0)  # Immediate timeout

        sm = SessionManager(config=config,
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=None)

        async with sm.session(user_id="user123"):
            # While session is active, cleanup should skip this builder
            cleaned = await sm._cleanup_inactive_per_user_builders()
            assert cleaned == 0
            assert "user123" in sm._per_user_builders


class TestSessionManagerContextExtraction:
    """Tests for _get_user_id_from_context method."""

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    def test_get_user_id_from_cookie(self, mock_registry):
        """Test user_id extraction from nat-session cookie."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=False)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            shared_workflow=MockWorkflow())

        # Set user_id in context state (this is what set_metadata_from_http_request does
        # when it extracts the nat-session cookie)
        ctx_state = ContextState.get()
        token = ctx_state.user_id.set("session-123")

        try:
            user_id = sm._get_user_id_from_context()
            assert user_id == "session-123"
        finally:
            ctx_state.user_id.reset(token)

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    def test_get_user_id_returns_none_when_no_cookie(self, mock_registry):
        """Test user_id extraction returns None when no cookie."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=False)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            shared_workflow=MockWorkflow())

        # With default empty context
        user_id = sm._get_user_id_from_context()
        # Should return None (or user_manager fallback if set)
        # The exact behavior depends on default metadata state
        assert user_id is None


class TestPerUserWorkflowIntegration:
    """Integration tests for complete per-user workflow flow."""

    @patch('nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder', MockPerUserWorkflowBuilder)
    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_multiple_users_isolated_builders(self, mock_registry):
        """Test multiple users get isolated builders."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=None)

        async with sm.session(user_id="user1"):
            async with sm.session(user_id="user2"):
                # Both should have their own builders
                assert "user1" in sm._per_user_builders
                assert "user2" in sm._per_user_builders

                # Builders should be different
                builder1 = sm._per_user_builders["user1"].builder
                builder2 = sm._per_user_builders["user2"].builder
                assert builder1 is not builder2

                # Both should have ref_count of 1
                assert sm._per_user_builders["user1"].ref_count == 1
                assert sm._per_user_builders["user2"].ref_count == 1

        # After both exit, ref_counts should be 0
        assert sm._per_user_builders["user1"].ref_count == 0
        assert sm._per_user_builders["user2"].ref_count == 0

    @patch('nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder', MockPerUserWorkflowBuilder)
    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_multiple_users_isolated_semaphores(self, mock_registry):
        """Test multiple users get isolated semaphores for concurrency control."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=None,
                            max_concurrency=4)

        async with sm.session(user_id="user1") as session1:
            async with sm.session(user_id="user2") as session2:
                # Each user should have their own semaphore
                semaphore1 = sm._per_user_builders["user1"].semaphore
                semaphore2 = sm._per_user_builders["user2"].semaphore

                assert semaphore1 is not semaphore2
                assert isinstance(semaphore1, asyncio.Semaphore)
                assert isinstance(semaphore2, asyncio.Semaphore)

                # Sessions should use the per-user semaphores
                assert session1._semaphore is semaphore1
                assert session2._semaphore is semaphore2

    @patch('nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder', MockPerUserWorkflowBuilder)
    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_concurrent_sessions_same_user(self, mock_registry):
        """Test concurrent sessions for same user share builder and track ref_count."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=None)

        async with sm.session(user_id="user1") as session1:
            assert sm._per_user_builders["user1"].ref_count == 1

            async with sm.session(user_id="user1"):
                # Same builder, ref_count = 2
                assert sm._per_user_builders["user1"].ref_count == 2
                assert session1.workflow is not None  # Both have access to workflow

            # After inner exits, ref_count = 1
            assert sm._per_user_builders["user1"].ref_count == 1

        # After outer exits, ref_count = 0
        assert sm._per_user_builders["user1"].ref_count == 0


class TestSessionManagerEntryFunction:
    """Tests for SessionManager entry_function support."""

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    def test_init_with_entry_function(self, mock_registry):
        """Test SessionManager stores entry_function."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=False)

        config = create_mock_config()
        shared_builder = MockWorkflowBuilder()
        shared_workflow = MockWorkflow()

        sm = SessionManager(config=config,
                            shared_builder=shared_builder,
                            entry_function="custom_func",
                            shared_workflow=shared_workflow,
                            max_concurrency=8)

        assert sm._entry_function == "custom_func"

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    def test_init_without_entry_function(self, mock_registry):
        """Test SessionManager defaults entry_function to None."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=False)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function=None,
                            shared_workflow=MockWorkflow())

        assert sm._entry_function is None

    @patch('nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder', MockPerUserWorkflowBuilder)
    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_per_user_builder_uses_entry_function(self, mock_registry):
        """Test per-user builder is created with correct entry_function."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function="custom_entry",
                            shared_workflow=None)

        # Mock the build method to capture the entry_function argument
        build_called_with = []
        original_build = MockPerUserWorkflowBuilder.build

        async def mock_build_with_capture(self, entry_function=None):
            build_called_with.append(entry_function)
            return await original_build(self, entry_function)

        MockPerUserWorkflowBuilder.build = mock_build_with_capture

        try:
            async with sm.session(user_id="user123"):
                pass

            # Verify build was called with the correct entry_function
            assert "custom_entry" in build_called_with
        finally:
            MockPerUserWorkflowBuilder.build = original_build

    @patch('nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder', MockPerUserWorkflowBuilder)
    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_different_entry_functions_create_separate_caches(self, mock_registry):
        """
        Test that different SessionManagers with different entry_functions
        create separate per-user builder caches (route isolation).
        """
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        shared_builder = MockWorkflowBuilder()
        config = create_mock_config()

        # Create two SessionManagers with different entry functions
        sm1 = SessionManager(
            config=config,
            shared_builder=shared_builder,
            entry_function=None,  # Default route
            shared_workflow=None)

        sm2 = SessionManager(
            config=config,
            shared_builder=shared_builder,
            entry_function="custom_func",  # Custom route
            shared_workflow=None)

        # Same user accessing different routes
        async with sm1.session(user_id="alice"):
            assert "alice" in sm1._per_user_builders

        async with sm2.session(user_id="alice"):
            assert "alice" in sm2._per_user_builders

        # Verify they have separate caches
        assert sm1._per_user_builders is not sm2._per_user_builders


class TestSessionManagerCreate:
    """Tests for SessionManager.create() factory method."""

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_create_shared_workflow(self, mock_registry):
        """Test create() builds shared workflow for non-per-user."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=False)

        shared_builder = MockWorkflowBuilder()
        build_called_with = []

        async def mock_build(entry_function=None):
            build_called_with.append(entry_function)
            return MockWorkflow()

        shared_builder.build = mock_build

        config = create_mock_config()

        sm = await SessionManager.create(config=config, shared_builder=shared_builder, entry_function="my_entry")

        assert sm._entry_function == "my_entry"
        assert "my_entry" in build_called_with
        assert sm._shared_workflow is not None
        assert sm._is_workflow_per_user is False

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_create_per_user_workflow(self, mock_registry):
        """Test create() does NOT build workflow for per-user."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        shared_builder = MockWorkflowBuilder()
        build_called = []

        async def mock_build(entry_function=None):
            build_called.append(entry_function)
            return MockWorkflow()

        shared_builder.build = mock_build

        config = create_mock_config()

        sm = await SessionManager.create(config=config, shared_builder=shared_builder, entry_function="my_entry")

        # Should NOT have built shared workflow
        assert len(build_called) == 0
        assert sm._shared_workflow is None
        assert sm._is_workflow_per_user is True

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_create_starts_cleanup_task_for_per_user(self, mock_registry):
        """Test create() starts cleanup task for per-user workflow."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        config = create_mock_config()

        sm = await SessionManager.create(config=config, shared_builder=MockWorkflowBuilder())

        assert sm._per_user_builders_cleanup_task is not None
        assert not sm._per_user_builders_cleanup_task.done()

        # Cleanup
        await sm.shutdown()

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_create_does_not_start_cleanup_for_shared(self, mock_registry):
        """Test create() does NOT start cleanup task for shared workflow."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=False)

        shared_builder = MockWorkflowBuilder()
        shared_builder.build = AsyncMock(return_value=MockWorkflow())

        sm = await SessionManager.create(config=create_mock_config(), shared_builder=shared_builder)

        assert sm._per_user_builders_cleanup_task is None


class TestSessionManagerShutdown:
    """Tests for SessionManager.shutdown() method."""

    @patch('nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder', MockPerUserWorkflowBuilder)
    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_shutdown_stops_cleanup_task(self, mock_registry):
        """Test shutdown() stops the cleanup task."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        sm = await SessionManager.create(config=create_mock_config(), shared_builder=MockWorkflowBuilder())

        cleanup_task = sm._per_user_builders_cleanup_task
        assert cleanup_task is not None
        assert not cleanup_task.done()

        await sm.shutdown()

        # Give the task time to finish
        await asyncio.sleep(0.1)
        assert cleanup_task.done() or cleanup_task.cancelled()

    @patch('nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder', MockPerUserWorkflowBuilder)
    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_shutdown_cleans_up_all_per_user_builders(self, mock_registry):
        """Test shutdown() cleans up all per-user builders."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        sm = await SessionManager.create(config=create_mock_config(), shared_builder=MockWorkflowBuilder())

        # Create some per-user builders
        async with sm.session(user_id="user1"):
            pass
        async with sm.session(user_id="user2"):
            pass

        assert len(sm._per_user_builders) == 2

        # Get references to builders to check __aexit__ was called
        builder1 = sm._per_user_builders["user1"].builder
        builder2 = sm._per_user_builders["user2"].builder

        await sm.shutdown()

        # Builders should be cleared
        assert len(sm._per_user_builders) == 0

        # Builders should have been exited
        assert builder1._exited is True
        assert builder2._exited is True

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_shutdown_is_safe_for_shared_workflow(self, mock_registry):
        """Test shutdown() is safe to call on shared workflow SessionManager."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=False)

        shared_builder = MockWorkflowBuilder()
        shared_builder.build = AsyncMock(return_value=MockWorkflow())

        sm = await SessionManager.create(config=create_mock_config(), shared_builder=shared_builder)

        # Should not raise
        await sm.shutdown()


class TestMultipleSessionManagersSharedBuilder:
    """Tests for multiple SessionManagers sharing a WorkflowBuilder."""

    @patch('nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder', MockPerUserWorkflowBuilder)
    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_multiple_session_managers_share_builder(self, mock_registry):
        """Test multiple SessionManagers can share the same WorkflowBuilder."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        shared_builder = MockWorkflowBuilder()
        config = create_mock_config()

        # Create multiple SessionManagers
        sm1 = SessionManager(config=config, shared_builder=shared_builder, entry_function=None)

        sm2 = SessionManager(config=config, shared_builder=shared_builder, entry_function="custom")

        # Verify they share the same builder
        assert sm1.shared_builder is sm2.shared_builder

        # But have independent per-user caches
        async with sm1.session(user_id="user1"):
            pass

        async with sm2.session(user_id="user1"):
            pass

        assert "user1" in sm1._per_user_builders
        assert "user1" in sm2._per_user_builders
        # Different PerUserWorkflowBuilder instances
        assert sm1._per_user_builders["user1"].builder is not sm2._per_user_builders["user1"].builder

    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_route_isolation_for_shared_workflows(self, mock_registry):
        """Test route isolation works for shared (non-per-user) workflows."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=False)

        shared_builder = MockWorkflowBuilder()
        config = create_mock_config()

        # Track workflows built
        workflows_built = []

        async def mock_build(entry_function=None):
            workflow = MockWorkflow()
            workflows_built.append((entry_function, workflow))
            return workflow

        shared_builder.build = mock_build

        sm_default = await SessionManager.create(config=config, shared_builder=shared_builder, entry_function=None)

        sm_eval = await SessionManager.create(config=config, shared_builder=shared_builder, entry_function="eval")

        # Should have built separate workflows
        assert len(workflows_built) == 2
        assert workflows_built[0][0] is None  # default
        assert workflows_built[1][0] == "eval"

        # Different workflow instances
        assert sm_default._shared_workflow is not sm_eval._shared_workflow

    @patch('nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder', MockPerUserWorkflowBuilder)
    @patch('nat.cli.type_registry.GlobalTypeRegistry')
    @pytest.mark.asyncio
    async def test_per_user_with_custom_entry_function(self, mock_registry):
        """Test per-user workflow with custom entry function."""
        mock_registry.get.return_value.get_function.return_value = create_mock_function_registration(is_per_user=True)

        sm = SessionManager(config=create_mock_config(),
                            shared_builder=MockWorkflowBuilder(),
                            entry_function="my_custom_entry",
                            shared_workflow=None)

        async with sm.session(user_id="user1") as session:
            assert session.user_id == "user1"
            assert session.workflow is not None
            # The workflow was built with the custom entry function
            assert sm._entry_function == "my_custom_entry"
