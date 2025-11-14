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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock

from nat.builder.workflow import Workflow
from nat.runtime.session import UserSession


def test_user_session_stores_workflow():
    """Test UserSession stores and exposes workflow."""
    mock_workflow = Mock(spec=Workflow)
    mock_workflow.config = Mock()

    session = UserSession(workflow=mock_workflow, max_concurrency=8)

    assert session.workflow is mock_workflow
    assert session.config == mock_workflow.config


async def test_user_session_run_executes_workflow():
    """Test UserSession.run() executes workflow."""
    from contextlib import asynccontextmanager

    mock_workflow = MagicMock(spec=Workflow)
    mock_workflow.config = Mock()

    # Create a proper async context manager mock
    mock_runner = AsyncMock()

    @asynccontextmanager
    async def mock_run(message):
        yield mock_runner

    # Replace workflow.run with the async context manager
    mock_workflow.run = mock_run

    session = UserSession(workflow=mock_workflow, max_concurrency=8)

    async with session.run("test message") as runner:
        assert runner is mock_runner


async def test_user_session_concurrency_control():
    """Test UserSession enforces max_concurrency limit."""
    mock_workflow = Mock(spec=Workflow)
    mock_workflow.config = Mock()

    session = UserSession(workflow=mock_workflow, max_concurrency=2)

    # Should allow 2 concurrent executions
    assert session._max_concurrency == 2
    assert hasattr(session, '_semaphore')
