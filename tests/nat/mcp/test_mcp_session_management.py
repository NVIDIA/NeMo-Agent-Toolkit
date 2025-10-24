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

import asyncio
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.plugins.mcp.client_config import MCPClientConfig
from nat.plugins.mcp.client_config import MCPServerConfig
from nat.plugins.mcp.client_impl import MCPFunctionGroup
from nat.plugins.mcp.client_impl import SessionData


class TestMCPSessionManagement:
    """Test the per-session client management functionality in MCPFunctionGroup."""

    async def cleanup_sessions(self, function_group):
        """Helper method to clean up all sessions."""
        for session_data in function_group._sessions.values():
            if hasattr(session_data, 'stop_event') and session_data.stop_event:
                session_data.stop_event.set()
            if hasattr(session_data,
                       'lifetime_task') and session_data.lifetime_task and not session_data.lifetime_task.done():
                try:
                    await asyncio.wait_for(session_data.lifetime_task, timeout=1.0)
                except (TimeoutError, asyncio.CancelledError):
                    session_data.lifetime_task.cancel()
                    try:
                        await session_data.lifetime_task
                    except asyncio.CancelledError:
                        pass
        function_group._sessions.clear()

    @pytest.fixture
    def mock_config(self):
        """Create a mock MCPClientConfig for testing."""
        config = MagicMock(spec=MCPClientConfig)
        config.type = "mcp_client"  # Required by FunctionGroup constructor
        config.max_sessions = 5
        config.session_idle_timeout = timedelta(minutes=30)

        # Mock server config
        config.server = MagicMock(spec=MCPServerConfig)
        config.server.transport = "streamable-http"
        config.server.url = "http://localhost:8080/mcp"

        # Mock timeouts
        config.tool_call_timeout = timedelta(seconds=60)
        config.auth_flow_timeout = timedelta(seconds=300)
        config.reconnect_enabled = True
        config.reconnect_max_attempts = 2
        config.reconnect_initial_backoff = 0.5
        config.reconnect_max_backoff = 50.0

        return config

    @pytest.fixture
    def mock_auth_provider(self):
        """Create a mock auth provider for testing."""
        auth_provider = MagicMock()
        auth_provider.config = MagicMock()
        auth_provider.config.default_user_id = "default-user-123"
        return auth_provider

    @pytest.fixture
    def mock_base_client(self):
        """Create a mock base MCP client for testing."""
        client = AsyncMock()
        client.server_name = "test-server"
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        return client

    @pytest.fixture
    def function_group(self, mock_config, mock_auth_provider, mock_base_client):
        """Create an MCPFunctionGroup instance for testing."""
        group = MCPFunctionGroup(config=mock_config)
        group._shared_auth_provider = mock_auth_provider
        group._client_config = mock_config
        group.mcp_client = mock_base_client
        return group

    async def test_get_session_client_returns_base_client_for_default_user(self, function_group):
        """Test that the base client is returned for the default user ID."""
        session_id = "default-user-123"  # Same as default_user_id

        client = await function_group._get_session_client(session_id)

        assert client == function_group.mcp_client
        assert len(function_group._sessions) == 0

    async def test_get_session_client_creates_new_session_client(self, function_group):
        """Test that a new session client is created for non-default session IDs."""
        session_id = "session-123"

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_client_class.return_value = mock_session_client

            client = await function_group._get_session_client(session_id)

            assert client == mock_session_client
            assert session_id in function_group._sessions
            assert function_group._sessions[session_id].client == mock_session_client
            mock_client_class.assert_called_once()
            mock_session_client.__aenter__.assert_called_once()

            # Clean up session
            await self.cleanup_sessions(function_group)

    async def test_get_session_client_reuses_existing_session_client(self, function_group):
        """Test that existing session clients are reused."""
        session_id = "session-123"

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_client_class.return_value = mock_session_client

            # Create first client
            client1 = await function_group._get_session_client(session_id)

            # Get the same client again
            client2 = await function_group._get_session_client(session_id)

            assert client1 == client2
            assert mock_client_class.call_count == 1  # Only created once

            # Clean up session
            await self.cleanup_sessions(function_group)

    async def test_get_session_client_updates_last_activity(self, function_group):
        """Test that last activity is updated when accessing existing sessions."""
        session_id = "session-123"

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_client_class.return_value = mock_session_client

            # Create session client
            await function_group._get_session_client(session_id)

            # Record initial activity time
            initial_time = function_group._sessions[session_id].last_activity

            # Wait a small amount and access again
            await asyncio.sleep(0.01)
            await function_group._get_session_client(session_id)

            # Activity time should be updated
            updated_time = function_group._sessions[session_id].last_activity
            assert updated_time > initial_time

            # Clean up session
            await self.cleanup_sessions(function_group)

    async def test_get_session_client_enforces_max_sessions_limit(self, function_group):
        """Test that the maximum session limit is enforced."""
        # Create clients up to the limit
        for i in range(function_group._client_config.max_sessions):
            session_id = f"session-{i}"

            with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
                mock_session_client = AsyncMock()
                mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
                mock_client_class.return_value = mock_session_client

                await function_group._get_session_client(session_id)

        # Try to create one more session - should raise RuntimeError
        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_client_class.return_value = mock_session_client

            with pytest.raises(RuntimeError, match="Maximum concurrent.*sessions.*exceeded"):
                await function_group._get_session_client("session-overflow")

        # Clean up all sessions
        await self.cleanup_sessions(function_group)

    async def test_cleanup_inactive_sessions_removes_old_sessions(self, function_group):
        """Test that inactive sessions are cleaned up."""
        session_id = "session-123"

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_session_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_session_client

            # Create session client
            await function_group._get_session_client(session_id)

            # Manually set last activity to be old
            old_time = datetime.now() - timedelta(hours=1)
            function_group._sessions[session_id].last_activity = old_time

            # Cleanup inactive sessions
            await function_group._cleanup_inactive_sessions(timedelta(minutes=30))

            # Session should be removed
            assert session_id not in function_group._sessions
            mock_session_client.__aexit__.assert_called_once()

    async def test_cleanup_inactive_sessions_preserves_active_sessions(self, function_group):
        """Test that sessions with active references are not cleaned up."""
        session_id = "session-123"

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_client_class.return_value = mock_session_client

            # Create session client
            await function_group._get_session_client(session_id)

            # Set reference count to indicate active usage
            function_group._sessions[session_id].ref_count = 1

            # Manually set last activity to be old
            old_time = datetime.now() - timedelta(hours=1)
            function_group._sessions[session_id].last_activity = old_time

            # Cleanup inactive sessions
            await function_group._cleanup_inactive_sessions(timedelta(minutes=30))

            # Session should be preserved due to active reference
            assert session_id in function_group._sessions

            # Clean up session
            await self.cleanup_sessions(function_group)

    async def test_session_usage_context_manager(self, function_group):
        """Test the session usage context manager for reference counting."""
        session_id = "session-123"

        # Create a session first
        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_client_class.return_value = mock_session_client

            await function_group._get_session_client(session_id)

        # Initially reference count should be 0
        assert function_group._sessions[session_id].ref_count == 0

        # Use context manager
        async with function_group._session_usage_context(session_id):
            # Reference count should be incremented
            assert function_group._sessions[session_id].ref_count == 1

            # Nested usage
            async with function_group._session_usage_context(session_id):
                assert function_group._sessions[session_id].ref_count == 2

        # Reference count should be decremented back to 0
        assert function_group._sessions[session_id].ref_count == 0

        # Clean up session
        await self.cleanup_sessions(function_group)

    async def test_session_usage_context_manager_multiple_sessions(self, function_group):
        """Test the session usage context manager with multiple sessions."""
        session1 = "session-1"
        session2 = "session-2"

        # Create sessions first
        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_client_class.return_value = mock_session_client

            await function_group._get_session_client(session1)
            await function_group._get_session_client(session2)

        # Use context managers for different sessions
        async with function_group._session_usage_context(session1):
            async with function_group._session_usage_context(session2):
                assert function_group._sessions[session1].ref_count == 1
                assert function_group._sessions[session2].ref_count == 1

        # Both should be back to 0
        assert function_group._sessions[session1].ref_count == 0
        assert function_group._sessions[session2].ref_count == 0

        # Clean up sessions
        await self.cleanup_sessions(function_group)

    async def test_create_session_client_unsupported_transport(self, function_group):
        """Test that creating session clients fails for unsupported transports."""
        # Change transport to unsupported type
        function_group._client_config.server.transport = "stdio"

        with pytest.raises(ValueError, match="Unsupported transport"):
            await function_group._create_session_client("session-123")

    async def test_cleanup_inactive_sessions_with_custom_max_age(self, function_group):
        """Test cleanup with custom max_age parameter."""
        session_id = "session-123"

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_session_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_session_client

            # Create session client
            await function_group._get_session_client(session_id)

            # Set last activity to be 10 minutes old
            old_time = datetime.now() - timedelta(minutes=10)
            function_group._sessions[session_id].last_activity = old_time

            # Cleanup with 5 minute max_age (should remove session)
            await function_group._cleanup_inactive_sessions(timedelta(minutes=5))

            # Session should be removed
            assert session_id not in function_group._sessions

    async def test_cleanup_inactive_sessions_with_longer_max_age(self, function_group):
        """Test cleanup with longer max_age parameter that doesn't remove sessions."""
        session_id = "session-123"

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_client_class.return_value = mock_session_client

            # Create session client
            await function_group._get_session_client(session_id)

            # Set last activity to be 10 minutes old
            old_time = datetime.now() - timedelta(minutes=10)
            function_group._sessions[session_id].last_activity = old_time

            # Cleanup with 20 minute max_age (should not remove session)
            await function_group._cleanup_inactive_sessions(timedelta(minutes=20))

            # Session should be preserved
            assert session_id in function_group._sessions

            # Clean up session
            await self.cleanup_sessions(function_group)

    async def test_cleanup_handles_client_close_errors(self, function_group):
        """Test that cleanup handles errors when closing client connections."""
        session_id = "session-123"

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_session_client.__aexit__ = AsyncMock(side_effect=Exception("Close error"))
            mock_client_class.return_value = mock_session_client

            # Create session client
            await function_group._get_session_client(session_id)

            # Set last activity to be old
            old_time = datetime.now() - timedelta(hours=1)
            function_group._sessions[session_id].last_activity = old_time

            # Cleanup should not raise exception despite close error
            await function_group._cleanup_inactive_sessions(timedelta(minutes=30))

            # Session should be removed from tracking even when close fails
            # (This is the new fail-safe behavior - cleanup always removes tracking)
            assert session_id not in function_group._sessions

    async def test_concurrent_session_creation(self, function_group):
        """Test that concurrent session creation is handled properly."""
        session_id = "session-123"

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_client_class.return_value = mock_session_client

            async def create_session():
                return await function_group._get_session_client(session_id)

            # Create multiple concurrent tasks
            tasks = [create_session() for _ in range(5)]
            clients = await asyncio.gather(*tasks)

            # All should return the same client instance
            assert all(client == clients[0] for client in clients)

            # Only one client should be created
            assert len(function_group._sessions) == 1
            assert session_id in function_group._sessions

        # Clean up session
        await self.cleanup_sessions(function_group)

    async def test_throttled_cleanup_on_access(self, function_group):
        """Test that cleanup is throttled and only runs periodically."""
        session_id = "session-123"

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_client_class.return_value = mock_session_client

            # Create session client
            await function_group._get_session_client(session_id)

            # Mock cleanup method to track calls
            cleanup_calls = 0
            original_cleanup = function_group._cleanup_inactive_sessions

            async def mock_cleanup(*args, **kwargs):
                nonlocal cleanup_calls
                cleanup_calls += 1
                return await original_cleanup(*args, **kwargs)

            function_group._cleanup_inactive_sessions = mock_cleanup

            # Manually trigger cleanup by setting last check time to be old
            old_time = datetime.now() - timedelta(minutes=10)
            function_group._last_cleanup_check = old_time

            # Access session - this should trigger cleanup due to old last_check time
            await function_group._get_session_client(session_id)

            # Access session multiple times quickly - cleanup should not be called again
            for _ in range(5):
                await function_group._get_session_client(session_id)

            # Cleanup should only be called once due to throttling
            assert cleanup_calls == 1

            # Clean up session
            await self.cleanup_sessions(function_group)

    async def test_manual_cleanup_sessions(self, function_group):
        """Test manual cleanup of sessions."""
        session1 = "session-1"
        session2 = "session-2"
        session3 = "session-3"

        # Create multiple sessions
        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_session_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_session_client

            await function_group._get_session_client(session1)
            await function_group._get_session_client(session2)
            await function_group._get_session_client(session3)

        # Verify all sessions exist
        assert function_group.session_count == 3
        assert session1 in function_group._sessions
        assert session2 in function_group._sessions
        assert session3 in function_group._sessions

        # Test 1: Manual cleanup with default timeout (should keep recent sessions)
        cleaned_count = await function_group.cleanup_sessions()
        assert cleaned_count == 0  # No sessions should be cleaned (they're recent)
        assert function_group.session_count == 3

        # Test 2: Manual cleanup with very short timeout (should clean all)
        cleaned_count = await function_group.cleanup_sessions(timedelta(seconds=0))
        assert cleaned_count == 3  # All sessions should be cleaned
        assert function_group.session_count == 0

        # Test 3: Manual cleanup when no sessions exist
        cleaned_count = await function_group.cleanup_sessions()
        assert cleaned_count == 0  # No sessions to clean

    async def test_manual_cleanup_with_active_sessions(self, function_group):
        """Test manual cleanup preserves sessions with active references."""
        session_id = "session-123"

        # Create session
        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_session_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_session_client

            await function_group._get_session_client(session_id)

        # Set reference count to indicate active usage
        function_group._sessions[session_id].ref_count = 1

        # Manual cleanup with 0 timeout (should not clean due to active reference)
        cleaned_count = await function_group.cleanup_sessions(timedelta(seconds=0))
        assert cleaned_count == 0  # Session should be preserved due to active reference
        assert session_id in function_group._sessions

        # Reset reference count and cleanup again
        function_group._sessions[session_id].ref_count = 0
        cleaned_count = await function_group.cleanup_sessions(timedelta(seconds=0))
        assert cleaned_count == 1  # Session should be cleaned now
        assert session_id not in function_group._sessions

    async def test_manual_cleanup_returns_correct_count(self, function_group):
        """Test that manual cleanup returns accurate count of cleaned sessions."""
        sessions = ["session-1", "session-2", "session-3", "session-4"]

        # Create sessions
        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_session_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_session_client

            for session_id in sessions:
                await function_group._get_session_client(session_id)

        # Verify all sessions created
        assert function_group.session_count == 4

        # Clean up 2 sessions by setting their activity to be old
        old_time = datetime.now() - timedelta(hours=1)
        function_group._sessions["session-1"].last_activity = old_time
        function_group._sessions["session-2"].last_activity = old_time

        # Manual cleanup with 30 minute timeout
        cleaned_count = await function_group.cleanup_sessions(timedelta(minutes=30))
        assert cleaned_count == 2  # Should clean exactly 2 sessions
        assert function_group.session_count == 2
        assert "session-1" not in function_group._sessions
        assert "session-2" not in function_group._sessions
        assert "session-3" in function_group._sessions
        assert "session-4" in function_group._sessions

        # Clean up remaining sessions
        await self.cleanup_sessions(function_group)

    async def test_lifetime_task_successful_initialization(self, function_group):
        """Test that lifetime task properly manages client lifecycle on success."""
        session_id = "test-session-123"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Mock the client creation
        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient', return_value=mock_client):
            client, stop_event, lifetime_task = await function_group._create_session_client(session_id)

        # Verify the client was created
        assert client == mock_client
        assert isinstance(stop_event, asyncio.Event)
        assert isinstance(lifetime_task, asyncio.Task)
        assert not lifetime_task.done()

        # Verify __aenter__ was called
        mock_client.__aenter__.assert_called_once()

        # Clean up
        stop_event.set()
        await lifetime_task
        assert lifetime_task.done()

        # Clean up any remaining sessions
        await self.cleanup_sessions(function_group)

    async def test_lifetime_task_initialization_failure(self, function_group):
        """Test that lifetime task properly handles __aenter__ failure."""
        session_id = "test-session-456"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(side_effect=RuntimeError("Connection failed"))
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient', return_value=mock_client):
            with pytest.raises(RuntimeError, match="Failed to initialize session client: Connection failed"):
                await function_group._create_session_client(session_id)

    async def test_lifetime_task_timeout(self, function_group):
        """Test that lifetime task times out if initialization hangs."""
        session_id = "test-session-timeout"
        mock_client = AsyncMock()

        # Make __aenter__ hang indefinitely
        async def hanging_aenter(self):
            await asyncio.sleep(1000)  # Never completes
            return mock_client

        mock_client.__aenter__ = hanging_aenter
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient', return_value=mock_client):
            with pytest.raises(RuntimeError, match="Session client initialization timed out after 60.0s"):
                await function_group._create_session_client(session_id)

    async def test_lifetime_task_cleanup_on_stop_event(self, function_group):
        """Test that lifetime task properly exits when stop_event is set."""
        session_id = "test-session-cleanup"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient', return_value=mock_client):
            _, stop_event, lifetime_task = await function_group._create_session_client(session_id)

        # Verify task is running
        assert not lifetime_task.done()

        # Signal stop
        stop_event.set()

        # Wait for task to complete
        await lifetime_task

        # Verify __aexit__ was called
        mock_client.__aexit__.assert_called_once_with(None, None, None)
        assert lifetime_task.done()

    async def test_lifetime_task_cancel_scope_respect(self, function_group):
        """Test that cancel scope is entered and exited in the same task."""
        session_id = "test-session-scope"
        mock_client = AsyncMock()
        enter_task_id = None
        exit_task_id = None

        # Track which task calls __aenter__ and __aexit__
        original_aenter = AsyncMock(return_value=mock_client)
        original_aexit = AsyncMock(return_value=None)

        async def tracked_aenter(self):
            nonlocal enter_task_id
            task = asyncio.current_task()
            enter_task_id = task.get_name() if task else "unknown"
            return await original_aenter()

        async def tracked_aexit(self, exc_type, exc_val, exc_tb):
            nonlocal exit_task_id
            task = asyncio.current_task()
            exit_task_id = task.get_name() if task else "unknown"
            return await original_aexit(exc_type, exc_val, exc_tb)

        mock_client.__aenter__ = tracked_aenter
        mock_client.__aexit__ = tracked_aexit

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient', return_value=mock_client):
            _, stop_event, lifetime_task = await function_group._create_session_client(session_id)

        # Signal stop and wait for completion
        stop_event.set()
        await lifetime_task
        assert lifetime_task.done()

        # Verify both enter and exit happened in the same task
        assert enter_task_id is not None
        assert exit_task_id is not None
        assert enter_task_id == exit_task_id
        assert isinstance(enter_task_id, str)
        assert "mcp-session-" in enter_task_id

        # Clean up any remaining sessions
        await self.cleanup_sessions(function_group)

    async def test_cleanup_with_lifetime_task(self, function_group):
        """Test that cleanup properly signals the lifetime task."""
        session_id = "test-cleanup-session"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient', return_value=mock_client):
            client, stop_event, lifetime_task = await function_group._create_session_client(session_id)

        # Create session data
        session_data = SessionData(client=client,
                                   last_activity=function_group._last_cleanup_check - timedelta(hours=2),
                                   ref_count=0,
                                   stop_event=stop_event,
                                   lifetime_task=lifetime_task)

        # Add to sessions
        function_group._sessions[session_id] = session_data

        # Perform cleanup
        await function_group._cleanup_inactive_sessions(timedelta(minutes=1))

        # Verify session was removed
        assert session_id not in function_group._sessions

        # Verify __aexit__ was called
        mock_client.__aexit__.assert_called_once_with(None, None, None)

    async def test_cleanup_skips_active_sessions_with_lifetime_task(self, function_group):
        """Test that cleanup skips sessions with ref_count > 0 using lifetime tasks."""
        session_id = "test-active-session"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient', return_value=mock_client):
            client, stop_event, lifetime_task = await function_group._create_session_client(session_id)

        # Create session data with active ref_count
        session_data = SessionData(
            client=client,
            last_activity=function_group._last_cleanup_check - timedelta(hours=2),
            ref_count=1,  # Active session
            stop_event=stop_event,
            lifetime_task=lifetime_task)

        # Add to sessions
        function_group._sessions[session_id] = session_data

        # Perform cleanup
        await function_group._cleanup_inactive_sessions(timedelta(minutes=1))

        # Verify session was NOT removed
        assert session_id in function_group._sessions

        # Verify __aexit__ was NOT called
        mock_client.__aexit__.assert_not_called()

        # Clean up manually
        stop_event.set()
        await lifetime_task

    async def test_cleanup_handles_already_done_task(self, function_group):
        """Test that cleanup handles tasks that are already done."""
        session_id = "test-done-session"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient', return_value=mock_client):
            client, stop_event, lifetime_task = await function_group._create_session_client(session_id)

        # Complete the task manually
        stop_event.set()
        await lifetime_task

        # Create session data with completed task
        session_data = SessionData(client=client,
                                   last_activity=function_group._last_cleanup_check - timedelta(hours=2),
                                   ref_count=0,
                                   stop_event=stop_event,
                                   lifetime_task=lifetime_task)

        # Add to sessions
        function_group._sessions[session_id] = session_data

        # Perform cleanup - should not hang or error
        await function_group._cleanup_inactive_sessions(timedelta(minutes=1))

        # Verify session was removed
        assert session_id not in function_group._sessions

    async def test_session_creation_and_usage_with_lifetime_task(self, function_group):
        """Test complete session lifecycle with lifetime tasks."""
        session_id = "test-full-session"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient', return_value=mock_client):
            # Create session
            client = await function_group._get_session_client(session_id)
            assert client == mock_client

            # Verify session exists
            assert session_id in function_group._sessions
            session_data = function_group._sessions[session_id]
            assert session_data.lifetime_task is not None
            assert not session_data.lifetime_task.done()

            # Use session context
            async with function_group._session_usage_context(session_id) as ctx_client:
                assert ctx_client == mock_client
                assert session_data.ref_count == 1

            # Verify ref_count was decremented
            assert session_data.ref_count == 0

            # Clean up
            session_data.stop_event.set()
            await session_data.lifetime_task
            assert session_data.lifetime_task.done()

            # Clean up any remaining sessions
            await self.cleanup_sessions(function_group)

    async def test_multiple_sessions_independence_with_lifetime_tasks(self, function_group):
        """Test that multiple sessions operate independently with lifetime tasks."""
        session1_id = "session-1"
        session2_id = "session-2"

        mock_client1 = AsyncMock()
        mock_client1.__aenter__ = AsyncMock(return_value=mock_client1)
        mock_client1.__aexit__ = AsyncMock(return_value=None)

        mock_client2 = AsyncMock()
        mock_client2.__aenter__ = AsyncMock(return_value=mock_client2)
        mock_client2.__aexit__ = AsyncMock(return_value=None)

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_client_class.side_effect = [mock_client1, mock_client2]

            # Create both sessions
            client1 = await function_group._get_session_client(session1_id)
            client2 = await function_group._get_session_client(session2_id)

            assert client1 == mock_client1
            assert client2 == mock_client2
            assert len(function_group._sessions) == 2

            # Clean up both sessions properly
            session1_data = function_group._sessions[session1_id]
            session2_data = function_group._sessions[session2_id]

            # Signal stop events
            session1_data.stop_event.set()
            session2_data.stop_event.set()

            # Wait for both tasks to complete
            await asyncio.gather(session1_data.lifetime_task, session2_data.lifetime_task, return_exceptions=True)

            # Verify both tasks are done
            assert session1_data.lifetime_task.done()
            assert session2_data.lifetime_task.done()

            # Clean up any remaining sessions
            await self.cleanup_sessions(function_group)


if __name__ == "__main__":
    pytest.main([__file__])
