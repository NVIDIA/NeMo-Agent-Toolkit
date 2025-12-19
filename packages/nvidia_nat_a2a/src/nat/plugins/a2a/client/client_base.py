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

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from datetime import timedelta
from typing import TYPE_CHECKING
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
from a2a.types import Task
from a2a.types import TextPart

if TYPE_CHECKING:
    from nat.authentication.interfaces import AuthProviderBase

logger = logging.getLogger(__name__)


class A2ABaseClient:
    """
    Minimal A2A client for connecting to an A2A agent.

    Args:
        base_url: The base URL of the A2A agent
        agent_card_path: Path to agent card (default: /.well-known/agent-card.json)
        task_timeout: Timeout for task operations (default: 300 seconds)
        streaming: Enable streaming responses (default: True)
        auth_provider: Optional NAT authentication provider for securing requests
    """

    def __init__(
        self,
        base_url: str,
        agent_card_path: str = "/.well-known/agent-card.json",
        task_timeout: timedelta = timedelta(seconds=300),
        streaming: bool = True,
        auth_provider: AuthProviderBase | None = None,
    ):
        self._base_url = base_url
        self._agent_card_path = agent_card_path
        self._task_timeout = task_timeout
        self._streaming = streaming
        self._auth_provider = auth_provider

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
        if self._httpx_client is not None or self._client is not None:
            raise RuntimeError("A2ABaseClient already initialized")

        # 1) Create httpx client explicitly
        self._httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(self._task_timeout.total_seconds()))

        # 2) Resolve agent card
        await self._resolve_agent_card()
        if not self._agent_card:
            raise RuntimeError("Agent card not resolved")

        # 3) Setup authentication interceptors if auth is configured
        interceptors = []
        if self._auth_provider:
            try:
                from a2a.client import AuthInterceptor
                from nat.plugins.a2a.auth.credential_service import A2ACredentialService

                credential_service = A2ACredentialService(
                    auth_provider=self._auth_provider,
                    agent_card=self._agent_card,
                )
                interceptors.append(AuthInterceptor(credential_service))
                logger.info("Authentication configured for A2A client")
            except ImportError as e:
                logger.error("Failed to setup authentication: %s", e)
                raise RuntimeError("Authentication requires a2a-sdk with AuthInterceptor support") from e

        # 4) Create A2A client with interceptors
        client_config = ClientConfig(
            httpx_client=self._httpx_client,
            streaming=self._streaming,
        )
        factory = ClientFactory(client_config)
        self._client = factory.create(self._agent_card, interceptors=interceptors)

        logger.info("Connected to A2A agent at %s", self._base_url)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        # Close A2A client first (if it exposes aclose)
        if self._client is not None:
            aclose = getattr(self._client, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    logger.warning("Error while closing A2A client", exc_info=True)

        # Then close httpx client
        if self._httpx_client is not None:
            try:
                await self._httpx_client.aclose()
            except Exception:
                logger.warning("Error while closing HTTPX client", exc_info=True)

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

    async def get_task(self, task_id: str, history_length: int | None = None) -> Task:
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

    async def cancel_task(self, task_id: str) -> Task:
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

    async def send_message_streaming(self,
                                     message_text: str,
                                     task_id: str | None = None,
                                     context_id: str | None = None) -> AsyncGenerator[ClientEvent | Message, None]:
        """
        Send a message to the agent and stream response events (alias for send_message).

        This method provides an explicit streaming interface that mirrors the A2A SDK pattern.
        It is functionally identical to send_message(), which already streams events.

        Args:
            message_text: The message text to send
            task_id: Optional task ID to continue an existing conversation
            context_id: Optional context ID for the conversation

        Yields:
            ClientEvent | Message: The agent's response events as they arrive.
        """
        async for event in self.send_message(message_text, task_id=task_id, context_id=context_id):
            yield event

    def extract_text_from_parts(self, parts: list) -> list[str]:
        """
        Extract text content from A2A message parts.

        Args:
            parts: List of A2A Part objects

        Returns:
            List of text strings extracted from the parts
        """
        text_parts = []
        for part in parts:
            # Handle Part wrapper (RootModel)
            if hasattr(part, 'root'):
                part_content = part.root
            else:
                part_content = part

            # Extract text from TextPart
            if hasattr(part_content, 'text'):
                text_parts.append(part_content.text)

        return text_parts

    def extract_text_from_task(self, task) -> str:
        """
        Extract text response from an A2A Task object.

        This method understands the A2A protocol structure and extracts the final
        text response from a completed task, prioritizing artifacts over history.

        Args:
            task: A2A Task object

        Returns:
            Extracted text response or status message

        Priority order:
            1. Check task status (return error/progress if not completed)
            2. Extract from task.artifacts (structured output)
            3. Fallback to last agent message in task.history
        """
        from a2a.types import TaskState

        # Check task status
        if task.status and task.status.state != TaskState.completed:
            # Task not completed - return status message or indicate in progress
            if task.status.state == TaskState.failed:
                return f"Task failed: {task.status.message or 'Unknown error'}"
            return f"Task in progress (state: {task.status.state})"

        # Priority 1: Extract from artifacts (structured output)
        if task.artifacts:
            # Get text from all artifacts
            all_text = []
            for artifact in task.artifacts:
                if artifact.parts:
                    text_parts = self.extract_text_from_parts(artifact.parts)
                    if text_parts:
                        all_text.extend(text_parts)
            if all_text:
                return " ".join(all_text)

        # Priority 2: Fallback to history (conversational messages)
        if task.history:
            # Get the last agent message from history
            for msg in reversed(task.history):
                if msg.role.value == 'agent':  # Get last agent message
                    text_parts = self.extract_text_from_parts(msg.parts)
                    if text_parts:
                        return " ".join(text_parts)

        return "No response"

    def extract_text_from_events(self, events: list) -> str:
        """
        Extract text response from a list of A2A events.

        This is a convenience method that handles both Message and ClientEvent types.

        Args:
            events: List of A2A events (ClientEvent or Message objects)

        Returns:
            Extracted text response
        """
        from a2a.types import Message as A2AMessage

        if not events:
            return "No response"

        # Get the last event
        last_event = events[-1]

        # If it's a Message, extract text from parts
        if isinstance(last_event, A2AMessage):
            text_parts = self.extract_text_from_parts(last_event.parts)
            return " ".join(text_parts) if text_parts else str(last_event)

        # If it's a ClientEvent (Task, TaskStatusUpdateEvent), extract from task
        if isinstance(last_event, tuple):
            task, _ = last_event
            return self.extract_text_from_task(task)

        return str(last_event)
