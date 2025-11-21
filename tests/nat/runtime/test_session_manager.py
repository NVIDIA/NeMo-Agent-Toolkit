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

from datetime import timedelta
from http.client import HTTPConnection
from unittest.mock import MagicMock
from unittest.mock import Mock

import pytest

from nat.data_models.config import Config
from nat.runtime.session import Session
from nat.runtime.session import SessionManager


def test_session_manager_init():
    config = Config()
    session_manager = SessionManager(config=config,
                                     max_concurrency=8,
                                     max_users=100,
                                     user_idle_timeout=timedelta(hours=1),
                                     cleanup_check_interval=timedelta(minutes=5),
                                     require_user_id=False)
    assert session_manager.config == config
    assert session_manager.max_concurrency == 8
    assert session_manager.max_users == 100
    assert session_manager.active_user_count == 0


async def test_session_manager_creates_shared_builder_once():
    """Test shared builder is created only once."""
    config = Config()
    session_manager = SessionManager(config=config)

    shared_builder_1 = await session_manager.ensure_shared_builder()
    shared_builder_2 = await session_manager.ensure_shared_builder()
    assert shared_builder_1 is shared_builder_2


async def test_session_manager_enforces_user_limit():
    """Test user limit is enforced."""
    config = Config()
    session_manager = SessionManager(config=config, max_users=2)

    await session_manager._get_or_create_user_workflow("user1")
    assert session_manager.active_user_count == 1
    await session_manager._get_or_create_user_workflow("user2")
    assert session_manager.active_user_count == 2

    with pytest.raises(RuntimeError, match="User limit reached"):
        await session_manager._get_or_create_user_workflow("user3")
    assert session_manager.active_user_count == 2


async def test_session_manager_session_returns_user_session():
    """Test SessionManager.session() returns UserSession."""
    config = Config()
    sm = SessionManager(config=config, require_user_id=False)

    # Mock HTTP connection
    mock_request = Mock()
    mock_request.cookies = {"nat-session": "test_user"}
    mock_request.headers = {}
    mock_request.url = Mock(path="/test", port=8000, scheme="http")
    mock_request.query_params = {}
    mock_request.path_params = {}
    mock_request.client = Mock(host="127.0.0.1", port=12345)

    async with sm.session(http_connection=mock_request) as user_session:
        # Should return UserSession
        assert isinstance(user_session, Session)
        assert user_session.workflow is not None
        assert user_session.config == config


async def test_session_manager_shutdown():
    """Test SessionManager shuts down cleanly."""
    config = Config()
    session_manager = SessionManager(config=config)

    # Create some users
    await session_manager._get_or_create_user_workflow("user1")
    await session_manager._get_or_create_user_workflow("user2")
    assert session_manager.active_user_count == 2

    # Shutdown
    await session_manager.shutdown()

    # All users should be cleaned up
    assert session_manager.active_user_count == 0
    assert session_manager._shared_builder is None
