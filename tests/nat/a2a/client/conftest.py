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
"""Client-specific fixtures for A2A client tests."""

from datetime import timedelta
from unittest.mock import patch

import pytest

from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugins.a2a.client.client_config import A2AClientConfig


@pytest.fixture
async def a2a_function_group(mock_a2a_client):
    """A2A client function group with mocked agent.

    This fixture provides a fully configured A2A client function group
    with a mocked A2A agent, ready for testing function invocations.

    Args:
        mock_a2a_client: Mock A2A client from parent conftest

    Yields:
        Tuple of (function_group, mock_client) for testing
    """
    with patch('nat.plugins.a2a.client.client_base.A2ABaseClient') as mock_class:
        # Configure the mock to return our mock client
        mock_class.return_value.__aenter__.return_value = mock_a2a_client

        # Create A2A client configuration
        config = A2AClientConfig(
            url="http://localhost:10000",
            task_timeout=timedelta(seconds=30),
        )

        # Create workflow builder and add function group
        async with WorkflowBuilder() as builder:
            group = await builder.add_function_group("test_agent", config)
            yield group, mock_a2a_client
