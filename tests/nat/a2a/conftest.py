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
"""Shared fixtures for A2A tests."""

from unittest.mock import AsyncMock

import pytest
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentSkill


@pytest.fixture
def sample_agent_card():
    """Sample agent card for testing.

    Returns a complete AgentCard with multiple skills for testing
    client functionality.
    """
    return AgentCard(
        name="Test Agent",
        version="1.0.0",
        protocol_version="1.0",
        url="http://localhost:10000/",
        description="Test agent for unit tests",
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
        ),
        skills=[
            AgentSkill(
                id="calculator.add",
                name="Add",
                description="Add two or more numbers together",
                examples=["Add 5 and 3", "What is 10 plus 20?"],
                tags=["calculator", "math"],
            ),
            AgentSkill(
                id="calculator.multiply",
                name="Multiply",
                description="Multiply two or more numbers together",
                examples=["Multiply 4 by 6", "What is 3 times 7?"],
                tags=["calculator", "math"],
            ),
            AgentSkill(
                id="current_datetime",
                name="Current DateTime",
                description="Get the current date and time",
                examples=["What time is it?", "What is the current date?"],
                tags=["time", "datetime"],
            ),
        ],
        default_input_modes=["text", "text/plain"],
        default_output_modes=["text", "text/plain"],
    )


@pytest.fixture
def mock_a2a_client(sample_agent_card):  # noqa: F811
    """Mock A2A client that simulates agent responses.

    This fixture creates a mock A2A client with predefined responses
    for testing without requiring a real A2A server.

    Args:
        sample_agent_card: The agent card to use for the mock client

    Returns:
        AsyncMock configured with agent card and response methods
    """
    mock_client = AsyncMock()
    mock_client.agent_card = sample_agent_card

    # Create a proper async function for send_message
    async def mock_send_message(query, task_id=None, context_id=None):
        return "Mock response from agent"

    # Create a proper async generator for streaming
    async def mock_streaming(query, task_id=None, context_id=None):
        yield {"type": "message", "content": "Streaming response"}

    # Assign the actual async functions, not AsyncMock
    mock_client._client = AsyncMock()
    mock_client._client.send_message = mock_send_message
    mock_client._client.send_message_streaming = mock_streaming

    return mock_client
