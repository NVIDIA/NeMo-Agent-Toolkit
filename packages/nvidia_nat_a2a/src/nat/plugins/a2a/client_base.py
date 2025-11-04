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
"""Minimal A2A client for testing with hello_world agent."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from datetime import timedelta
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver
from a2a.client import Client
from a2a.client import ClientConfig
from a2a.client import ClientEvent
from a2a.client import ClientFactory
from a2a.types import AgentCard
from a2a.types import Message
from a2a.types import Part
from a2a.types import Role
from a2a.types import TextPart

logger = logging.getLogger(__name__)


class A2ABaseClient:
    """
    Minimal A2A client for connecting to an A2A agent.

    Args:
        base_url: The base URL of the A2A agent
        task_timeout: Timeout for task operations (default: 300 seconds)
    """

    def __init__(self, base_url: str, agent_card_path: str, task_timeout: timedelta):
        self._base_url = base_url
        self._agent_card_path = agent_card_path
        self._task_timeout = task_timeout

        self._exit_stack: AsyncExitStack | None = None
        self._httpx_client: httpx.AsyncClient | None = None
        self._client: Client | None = None
        self._agent_card: AgentCard | None = None

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def agent_card(self) -> AgentCard | None:
        return self._agent_card

    async def __aenter__(self):
        if self._exit_stack:
            raise RuntimeError("A2ABaseClient already initialized")

        self._exit_stack = AsyncExitStack()

        # Create httpx client
        self._httpx_client = await self._exit_stack.enter_async_context(
            httpx.AsyncClient(timeout=httpx.Timeout(self._task_timeout.total_seconds())))

        # Resolve agent card
        await self._resolve_agent_card()

        # Create client using ClientFactory
        if not self._agent_card:
            raise RuntimeError("Agent card not resolved")

        client_config = ClientConfig(httpx_client=self._httpx_client, streaming=True)
        factory = ClientFactory(client_config)
        self._client = factory.create(self._agent_card)

        logger.info("Connected to A2A agent at %s", self._base_url)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._httpx_client = None
            self._client = None
            self._agent_card = None

    async def _resolve_agent_card(self):
        """Fetch the agent card from the A2A agent."""
        if not self._httpx_client:
            raise RuntimeError("httpx_client is not initialized")

        try:
            resolver = A2ACardResolver(httpx_client=self._httpx_client,
                                       base_url=self._base_url,
                                       agent_card_path=self._agent_card_path)
            logger.info("Fetching agent card from: %s%s", self._base_url, self._agent_card_path)
            self._agent_card = await resolver.get_agent_card()
            logger.info("Successfully fetched public agent card")
            # TODO: add support for authenticated extended agent card
        except Exception as e:
            logger.error("Failed to fetch agent card: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to fetch agent card from {self._base_url}") from e

    async def send_message(self,
                           message_text: str,
                           task_id: str | None = None,
                           context_id: str | None = None) -> AsyncGenerator[ClientEvent | Message, None]:
        """
        Send a message to the agent and stream response events.

        This is the low-level A2A protocol method that yields events as they arrive.
        For simpler usage, prefer the high-level agent function registered by this client.

        Args:
            message_text: The message text to send
            task_id: Optional task ID to continue an existing conversation
            context_id: Optional context ID for the conversation

        Yields:
            ClientEvent | Message: The agent's response events as they arrive.
                ClientEvent is a tuple of (Task, UpdateEvent | None)
                Message is a direct message response
        """
        if not self._client:
            raise RuntimeError("A2ABaseClient not initialized")

        text_part = TextPart(text=message_text)
        parts: list[Part] = [Part(root=text_part)]
        message = Message(role=Role.user, parts=parts, message_id=uuid4().hex, task_id=task_id, context_id=context_id)

        async for response in self._client.send_message(message):
            yield response

    async def get_task(self, task_id: str, history_length: int | None = None):
        """
        Get the status and details of a specific task.

        This is an A2A protocol operation for retrieving task information.

        Args:
            task_id: The unique identifier of the task
            history_length: Optional limit on the number of history messages to retrieve

        Returns:
            Task: The task object with current status and history
        """
        if not self._client:
            raise RuntimeError("A2ABaseClient not initialized")

        from a2a.types import TaskQueryParams
        params = TaskQueryParams(id=task_id, history_length=history_length)
        return await self._client.get_task(params)

    async def cancel_task(self, task_id: str):
        """
        Cancel a running task.

        This is an A2A protocol operation for canceling tasks.

        Args:
            task_id: The unique identifier of the task to cancel

        Returns:
            Task: The task object with updated status
        """
        if not self._client:
            raise RuntimeError("A2ABaseClient not initialized")

        from a2a.types import TaskIdParams
        params = TaskIdParams(id=task_id)
        return await self._client.cancel_task(params)
