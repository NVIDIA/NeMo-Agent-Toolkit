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
"""Client-specific fixtures for A2A client tests."""

from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentSkill

from nat.builder.function import FunctionGroup
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugins.a2a.client.client_config import A2AClientConfig


@pytest.fixture(name="sample_agent_card")
def fixture_sample_agent_card() -> AgentCard:
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
                id=f"calculator{FunctionGroup.SEPARATOR}add",
                name="Add",
                description="Add two or more numbers together",
                examples=["Add 5 and 3", "What is 10 plus 20?"],
                tags=["calculator", "math"],
            ),
            AgentSkill(
                id=f"calculator{FunctionGroup.SEPARATOR}multiply",
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


@pytest.fixture(name="mock_a2a_client")
def fixture_mock_a2a_client(sample_agent_card: AgentCard) -> AsyncMock:
    """Mock A2A client that simulates agent responses.

    This fixture creates a mock A2A client with predefined responses
    for testing without requiring a real A2A server.

    Args:
        sample_agent_card: The agent card to use for the mock client

    Returns:
        AsyncMock configured with agent card and response methods
    """
    mock_client = AsyncMock()
    # Configure the mock to properly return the agent_card as a property
    type(mock_client).agent_card = sample_agent_card

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


@pytest.fixture(name="a2a_function_group")
async def fixture_a2a_function_group(
    mock_a2a_client: AsyncMock,
    sample_agent_card: AgentCard,
    mock_user_context,
) -> tuple[FunctionGroup, AsyncMock]:
    """A2A client function group with mocked agent.

    This fixture provides a fully configured A2A client function group
    with a mocked A2A agent, ready for testing function invocations.

    Args:
        mock_a2a_client: Mock A2A client fixture
        sample_agent_card: Sample agent card fixture
        mock_user_context: Mock user context fixture

    Yields:
        Tuple of (function_group, mock_client) for testing
    """
    with patch('nat.plugins.a2a.client.client_impl.A2ABaseClient') as mock_class:
        # Configure the mock: the return_value is what gets assigned to self._client
        # Set agent_card on the mock instance that will be used
        mock_class.return_value.agent_card = sample_agent_card
        mock_class.return_value.__aenter__.return_value = mock_class.return_value

        # Create A2A client configuration
        config = A2AClientConfig(
            url="http://localhost:10000",
            task_timeout=timedelta(seconds=30),
        )

        # Mock the Context to provide a user_id (required for per-user A2A clients)
        with patch('nat.builder.context.Context') as mock_context:
            mock_context.get.return_value = mock_user_context

            # Create workflow builder and add function group
            async with WorkflowBuilder() as builder:
                group = await builder.add_function_group("test_agent", config)
                yield group, mock_class.return_value
