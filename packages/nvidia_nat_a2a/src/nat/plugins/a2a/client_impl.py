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
"""Minimal A2A client implementation for NAT workflows."""

import logging

from nat.builder.function import FunctionGroup
from nat.builder.workflow_builder import Builder
from nat.cli.register_workflow import register_function_group
from nat.plugins.a2a.client_base import A2ABaseClient
from nat.plugins.a2a.client_config import A2AClientConfig

logger = logging.getLogger(__name__)


class A2AClientFunctionGroup(FunctionGroup):
    """
    A minimal FunctionGroup for A2A agents.

    Exposes a simple `send_message` function to interact with A2A agents.
    """

    def __init__(self, config: A2AClientConfig, builder: Builder):
        super().__init__(config=config)
        self._builder = builder
        self._client: A2ABaseClient | None = None

    async def __aenter__(self):
        """Initialize the A2A client."""
        config: A2AClientConfig = self._config  # type: ignore[assignment]
        base_url = str(config.agent.url)

        # Create simple A2A client without auth
        self._client = A2ABaseClient(
            base_url=base_url,
            task_timeout=config.agent.task_timeout,
        )

        # Initialize the client
        await self._client.__aenter__()

        logger.info("Connected to A2A agent at %s", base_url)

        # Log agent card if available
        if self._client.agent_card:
            card = self._client.agent_card
            logger.info("Agent: %s v%s", card.name, card.version)

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Clean up the A2A client."""
        if self._client:
            await self._client.__aexit__(exc_type, exc_value, traceback)
            self._client = None
            logger.info("Disconnected from A2A agent")

    async def send_message(self, message: str) -> str:
        """
        Send a message to the A2A agent.

        Args:
            message: The message text to send

        Returns:
            str: The agent's response as a string
        """
        if not self._client:
            raise RuntimeError("A2A client not initialized")

        result = await self._client.send_message(message_text=message)

        # Extract text response
        from a2a.types import Message as A2AMessage
        from a2a.types import Task as A2ATask

        if isinstance(result, A2AMessage):
            # Extract text from message parts
            parts = []
            for part in result.parts:
                if hasattr(part, "text"):
                    parts.append(part.text)
            return "\n".join(parts) if parts else str(result)

        if isinstance(result, A2ATask):
            # Extract the last message from the task
            if result.messages:
                last_msg = result.messages[-1]
                parts = []
                for part in last_msg.parts:
                    if hasattr(part, "text"):
                        parts.append(part.text)
                return "\n".join(parts) if parts else str(last_msg)

        return str(result)


@register_function_group(config_type=A2AClientConfig)
async def a2a_client_function_group(config: A2AClientConfig, _builder: Builder):
    """
    Connect to an A2A agent and expose send_message as a function.

    Example workflow YAML:
    ```yaml
    functions:
      - type: a2a_client
        agent:
          url: http://localhost:9999
    ```
    """
    async with A2AClientFunctionGroup(config, _builder) as group:
        yield group
