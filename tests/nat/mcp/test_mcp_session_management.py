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


class TestMCPSessionManagement:
    """Test the per-session client management functionality in MCPFunctionGroup."""

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
        assert len(function_group._session_clients) == 0

    async def test_get_session_client_creates_new_session_client(self, function_group):
        """Test that a new session client is created for non-default session IDs."""
        session_id = "session-123"

        with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
            mock_session_client = AsyncMock()
            mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
            mock_client_class.return_value = mock_session_client

            client = await function_group._get_session_client(session_id)

            assert client == mock_session_client
            assert session_id in function_group._session_clients
            assert session_id in function_group._session_last_activity
            mock_client_class.assert_called_once()
            mock_session_client.__aenter__.assert_called_once()

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
            initial_time = function_group._session_last_activity[session_id]

            # Wait a small amount and access again
            await asyncio.sleep(0.01)
            await function_group._get_session_client(session_id)

            # Activity time should be updated
            updated_time = function_group._session_last_activity[session_id]
            assert updated_time > initial_time

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
            function_group._session_last_activity[session_id] = old_time

            # Cleanup inactive sessions
            await function_group._cleanup_inactive_sessions(timedelta(minutes=30))

            # Session should be removed
            assert session_id not in function_group._session_clients
            assert session_id not in function_group._session_last_activity
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
            function_group._session_ref_counts[session_id] = 1

            # Manually set last activity to be old
            old_time = datetime.now() - timedelta(hours=1)
            function_group._session_last_activity[session_id] = old_time

            # Cleanup inactive sessions
            await function_group._cleanup_inactive_sessions(timedelta(minutes=30))

            # Session should be preserved due to active reference
            assert session_id in function_group._session_clients
            assert session_id in function_group._session_last_activity

    async def test_session_usage_context_manager(self, function_group):
        """Test the session usage context manager for reference counting."""
        session_id = "session-123"

        # Initially no reference count
        assert session_id not in function_group._session_ref_counts

        # Use context manager
        async with function_group._session_usage_context(session_id):
            # Reference count should be incremented
            assert function_group._session_ref_counts[session_id] == 1

            # Nested usage
            async with function_group._session_usage_context(session_id):
                assert function_group._session_ref_counts[session_id] == 2

        # Reference count should be decremented and removed when it reaches 0
        assert session_id not in function_group._session_ref_counts

    async def test_session_usage_context_manager_multiple_sessions(self, function_group):
        """Test the session usage context manager with multiple sessions."""
        session1 = "session-1"
        session2 = "session-2"

        # Use context managers for different sessions
        async with function_group._session_usage_context(session1):
            async with function_group._session_usage_context(session2):
                assert function_group._session_ref_counts[session1] == 1
                assert function_group._session_ref_counts[session2] == 1

        # Both should be cleaned up
        assert session1 not in function_group._session_ref_counts
        assert session2 not in function_group._session_ref_counts

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
            function_group._session_last_activity[session_id] = old_time

            # Cleanup with 5 minute max_age (should remove session)
            await function_group._cleanup_inactive_sessions(timedelta(minutes=5))

            # Session should be removed
            assert session_id not in function_group._session_clients

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
            function_group._session_last_activity[session_id] = old_time

            # Cleanup with 20 minute max_age (should not remove session)
            await function_group._cleanup_inactive_sessions(timedelta(minutes=20))

            # Session should be preserved
            assert session_id in function_group._session_clients

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
            function_group._session_last_activity[session_id] = old_time

            # Cleanup should not raise exception despite close error
            await function_group._cleanup_inactive_sessions(timedelta(minutes=30))

            # Session should NOT be removed from tracking when close fails
            # (This is the actual behavior - cleanup only removes on successful close)
            assert session_id in function_group._session_clients
            assert session_id in function_group._session_last_activity

    async def test_concurrent_session_creation(self, function_group):
        """Test that concurrent session creation is handled properly."""
        session_id = "session-123"

        async def create_session():
            with patch('nat.plugins.mcp.client_base.MCPStreamableHTTPClient') as mock_client_class:
                mock_session_client = AsyncMock()
                mock_session_client.__aenter__ = AsyncMock(return_value=mock_session_client)
                mock_client_class.return_value = mock_session_client

                return await function_group._get_session_client(session_id)

        # Create multiple concurrent tasks
        tasks = [create_session() for _ in range(5)]
        clients = await asyncio.gather(*tasks)

        # All should return the same client instance
        assert all(client == clients[0] for client in clients)

        # Only one client should be created
        assert len(function_group._session_clients) == 1
        assert session_id in function_group._session_clients

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


if __name__ == "__main__":
    pytest.main([__file__])
