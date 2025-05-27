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
import logging
from collections.abc import AsyncIterable
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver
from a2a.client import A2AClient
from a2a.types import AgentCard
from a2a.types import Artifact
from a2a.types import CancelTaskRequest
from a2a.types import CancelTaskResponse
from a2a.types import GetTaskRequest
from a2a.types import GetTaskResponse
from a2a.types import MessageSendParams
from a2a.types import SendMessageRequest
from a2a.types import SendMessageResponse
from a2a.types import SendStreamingMessageRequest
from a2a.types import SendStreamingMessageResponse
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskState
from a2a.types import TaskStatusUpdateEvent
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# enable debug logging
logger.setLevel(logging.DEBUG)


class A2AClientHelper:
    """
    A helper class for the A2A API that wraps the external a2a-sdk client.
    """

    def __init__(
        self,
        url: str,
        wait_timeout: int,
        retry_frequency: int,
    ):
        """
        Configure the A2AClientHelper
        """
        self.url: str = url.rstrip("/")

        # Initialize transport
        self._transport_client = httpx.AsyncClient()
        # TODO: Use a single session id for all tasks submitted by this client. Looks like this is no longer needed?
        self._session_id: str = uuid4().hex
        logger.debug("Create A2A Client with Session ID: %s", self._session_id)

        # Configuration for waiting for the task to complete
        self._wait_time = wait_timeout
        self._retry_frequency = retry_frequency

        # Initialize the external client with the card after it is fetched via get_card
        self._initialized: bool = False
        self._public_agent_card: AgentCard | None = None
        self._extended_agent_card: AgentCard | None = None
        self._a2a_client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Close any transport resources
        """
        await self._transport_client.aclose()

    @property
    def agent_card(self) -> AgentCard:
        """
        If extended card is supported, return the extended card, otherwise return the public card
        """
        return self._extended_agent_card if self._extended_agent_card else self._public_agent_card

    async def get_agent_card(self, agent_card_path: str, extended_agent_card_path: str) -> AgentCard:
        """
        Get the AgentCard from the server using a well-known path
        The agent card provides:
        1. Function description - used as the description of the tool
        2. Input schema - TODO: not used yet
        3. Output schema - TODO: not used yet
        4. Capabilities (streaming, push notifications etc.) - TODO: partial support
        5. Authentication - TODO: partial support
        """
        if not self._initialized:
            resolver = A2ACardResolver(httpx_client=self._transport_client,
                                       base_url=self.url,
                                       agent_card_path=agent_card_path)
            logger.info("Attempting to fetch public agent card from %s", self.url + agent_card_path)
            self._public_agent_card = await resolver.get_agent_card()
            logger.info("Got the public agent card: %s",
                        self._public_agent_card.model_dump_json(indent=2, exclude_none=True))

            if not self._public_agent_card:
                raise RuntimeError("No public agent card found")

            if self._public_agent_card.supportsAuthenticatedExtendedCard:
                logger.info("Attempting to fetch extended agent card from %s", self.url + extended_agent_card_path)
                auth_headers_dict = {"Authorization": "Bearer dummy-token-for-extended-card"}
                self._extended_agent_card = await resolver.get_agent_card(relative_card_path=extended_agent_card_path,
                                                                          http_kwargs={"headers": auth_headers_dict})
                logger.info("Got the extended agent card: %s",
                            self._extended_agent_card.model_dump_json(indent=2, exclude_none=True))

            # Initialize the external client with the card after it is fetched
            self._a2a_client = A2AClient(httpx_client=self._transport_client, agent_card=self.agent_card)
            self._initialized = True

        return self.agent_card

    def _build_message_payload(self, task_id: str, message: str) -> MessageSendParams:
        """
        Build the message payload for a task request
        TODO:
        1. Add support for other output modes
        2. Add support for other message types (DataPart, FilePart, etc.)
        """
        from a2a.types import Message
        from a2a.types import MessageSendConfiguration
        from a2a.types import Role
        from a2a.types import TextPart

        configuration = MessageSendConfiguration(acceptedOutputModes=["text"])
        message = Message(messageId=uuid4().hex, role=Role.user, parts=[TextPart(text=message)], taskId=task_id)

        return MessageSendParams(configuration=configuration, message=message, metadata=None)

    async def send_task(self, task_id: str, message: str) -> SendMessageResponse:
        """
        Send a single (non-streaming) task request and return the response
        """
        request = SendMessageRequest(params=self._build_message_payload(task_id=task_id, message=message))
        return await self._a2a_client.send_message(request)

    async def send_task_streaming(self, task_id: str, message: str) -> AsyncIterable[SendStreamingMessageResponse]:
        """
        Send a task streaming request.
        """
        request = SendStreamingMessageRequest(params=self._build_message_payload(task_id=task_id, message=message))
        async for response in self._a2a_client.send_message_streaming(request):
            yield response

    async def get_task(self, task_id: str) -> GetTaskResponse:
        """
        Get the task status and artifact by id
        """
        request = GetTaskRequest(params={"id": task_id})
        return await self._a2a_client.get_task(request)

    async def get_task_with_retry(self, task_id: str) -> GetTaskResponse:
        """
        Get the task status and artifact by id with retry logic
        TODO: use an external library for this such as tenacity
        """
        wait_time = self._wait_time
        retry_frequency = self._retry_frequency if self._retry_frequency < wait_time else wait_time

        for _ in range(wait_time // retry_frequency):
            result = await self.get_task(task_id)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Get task result: %s", result.model_dump_json(exclude_none=True))

            if result.result and result.result.status.state == TaskState.COMPLETED:
                return result
            await asyncio.sleep(retry_frequency)
        return result

    async def cancel_task(self, task_id: str) -> CancelTaskResponse:
        """
        Cancel a task by id
        """
        request = CancelTaskRequest(params={"id": task_id})
        return await self._a2a_client.cancel_task(request)

    def _artifact_to_output_string(self, artifact: Artifact) -> str:
        """
        Parse artifact part. This can be of type text, file, or data.
        TODO: This needs some refinement to handle the different types of artifacts.
        """
        if not artifact:
            return "No artifact found"
        elif not artifact.parts:
            return "No artifact parts found"
        elif artifact.parts[0].type == "text":
            return artifact.parts[0].text
        elif artifact.parts[0].type == "file":
            return f"File: {artifact.parts[0].file.name}"
        elif artifact.parts[0].type == "data":
            # TODO: Handle data parts
            return f"{artifact.parts[0].data}"
        else:
            return "Unknown artifact type"

    async def complete_task(self, task_id: str, prompt: str) -> dict[str, Any]:
        """
        Create a new task, wait for it to complete, and return the parsed result from the artifact
        """
        streaming = self.agent_card.capabilities.streaming
        artifacts = []
        final_state = None

        if streaming:
            response_stream = self.send_task_streaming(task_id=task_id, message=prompt)

            async for result in response_stream:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Stream event: %s", result.model_dump_json(exclude_none=True))

                if isinstance(result.result, TaskStatusUpdateEvent):
                    final_state = result.result.status.state
                    if final_state == TaskState.COMPLETED.name:
                        # This is the last event and the task is complete
                        break
                elif isinstance(result.result, TaskArtifactUpdateEvent):
                    if result.result.artifact:
                        artifacts.append(result.result.artifact)

            if artifacts:
                artifact = artifacts[0]
            else:
                # If the artifact was not present in the streaming response,
                # get the final result from the server
                task_result = await self.get_task_with_retry(task_id)
                try:
                    artifact = task_result.result.artifacts[0]
                except IndexError:
                    artifact = None
                except AttributeError:
                    artifact = None

            # TODO: Handle input required
            return self._artifact_to_output_string(artifact)
        else:
            taskResult = await self.send_task(task_id=task_id, message=prompt)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Task result: %s", taskResult.model_dump_json(exclude_none=True))

            state = getattr(taskResult.result.status, "name", None)
            if state == TaskState.INPUT_REQUIRED.name:
                # TODO: Handle input required
                return await self.complete_task(task_id=task_id, prompt=task_result.result.status.message.parts[0].text)
            else:
                return self._artifact_to_output_string(task_result.result.artifacts[0])


class A2AToolClient:
    """
    A client for the A2A API that includes tool schema information.
    """

    def __init__(
        self,
        url: str,
        agent_card_path: str,
        extended_agent_card_path: str,
        tool_name: str,
        tool_input_schema: dict,
        description: str,
        wait_timeout: int,
        retry_frequency: int,
    ):
        """
        Configure the A2AToolClient
        """
        # Setup the A2A client helper. This sets up transport and session.
        self._client = A2AClientHelper(
            url,
            wait_timeout=wait_timeout,
            retry_frequency=retry_frequency,
        )

        # Setup tool attributes
        self._tool_name: str = tool_name
        self._description: str = description
        # The description of the tool is set after the agent card is fetched
        self._tool_description: str | None = None
        # TODO: Create the input schema from the info in the AgentCard
        self._input_schema: type[BaseModel] | None = tool_input_schema

        # The agent card must be fetched before the tool can be used
        self._initialized: bool = False
        self._agent_card_path: str = agent_card_path
        self._extended_agent_card_path: str = extended_agent_card_path

    async def get_agent_card(self) -> None:
        """
        Initialize the client by fetching the card and setting the description.
        This must be called before using the client.
        """
        if not self._initialized:
            await self._client.get_agent_card(self._agent_card_path, self._extended_agent_card_path)
            # Set the description if provided, otherwise use the AgentCard description
            if self._description:
                self._tool_description = self._description
            elif self._client.agent_card:
                self._tool_description = self._client.agent_card.description
            self._initialized = True

    @property
    def name(self) -> str:
        return self._tool_name

    @property
    def description(self) -> str | None:
        return self._tool_description

    @property
    def input_schema(self) -> type[BaseModel] | None:
        return self._input_schema

    async def acall(self, tool_input: str) -> str:
        """
        Create a new task for each tool call
        """
        task_id = uuid4().hex
        logger.debug("Start new Task ID: %s", task_id)

        output = await self._client.complete_task(task_id=task_id, prompt=tool_input)
        logger.debug("Task result: %s", output)
        return output
