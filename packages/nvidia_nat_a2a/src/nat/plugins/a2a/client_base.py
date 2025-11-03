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
from a2a.client import A2AClient
from a2a.types import AgentCard
from a2a.types import Message
from a2a.types import MessageSendParams
from a2a.types import Part
from a2a.types import Role
from a2a.types import SendMessageRequest
from a2a.types import SendMessageResponse
from a2a.types import SendStreamingMessageRequest
from a2a.types import SendStreamingMessageResponse
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
        self._a2a_client: A2AClient | None = None
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

        # Create A2A client
        self._a2a_client = A2AClient(httpx_client=self._httpx_client, agent_card=self._agent_card)

        logger.info("Connected to A2A agent at %s", self._base_url)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._httpx_client = None
            self._a2a_client = None
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

    async def send_message(self, message_text: str, task_id: str | None = None) -> SendMessageResponse:
        """
        Send a message to the agent.

        Args:
            message_text: The message text to send
            task_id: Optional task ID to continue an existing conversation

        Returns:
            SendMessageResponse: The agent's response
        """
        if not self._a2a_client:
            raise RuntimeError("A2ABaseClient not initialized")

        text_part = TextPart(text=message_text)
        parts: list[Part] = [Part(root=text_part)]
        message = Message(role=Role.user, parts=parts, message_id=uuid4().hex, task_id=task_id)
        params = MessageSendParams(message=message)
        request = SendMessageRequest(id=str(uuid4()), params=params)
        return await self._a2a_client.send_message(request)

    async def send_message_streaming(self,
                                     message_text: str,
                                     task_id: str | None = None) -> AsyncGenerator[SendStreamingMessageResponse, None]:
        """
        Send a message to the agent and stream the response.

        Args:
            message_text: The message text to send
            task_id: Optional task ID to continue an existing conversation

        Returns:
            AsyncGenerator[SendStreamingMessageResponse, None]: The agent's response
        """
        if not self._a2a_client:
            raise RuntimeError("A2ABaseClient not initialized")

        text_part = TextPart(text=message_text)
        parts: list[Part] = [Part(root=text_part)]
        message = Message(role=Role.user, parts=parts, message_id=uuid4().hex, task_id=task_id)
        params = MessageSendParams(message=message)
        request = SendStreamingMessageRequest(id=str(uuid4()), params=params)
        async for chunk in self._a2a_client.send_message_streaming(request):
            yield chunk
