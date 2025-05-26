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
from a2a.card import A2ACardResolver
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

    def __init__(self, url: str):
        """
        Initialize the A2AClientHelper
        """
        self.url = url.rstrip("/")
        self.agent_card = None

        self._client = httpx.AsyncClient()
        # Use a single session id for all tasks submitted by this client
        self._session_id: str = uuid4().hex
        logger.debug("Create A2A Client with Session ID: %s", self._session_id)

        # Post configuration
        self._post_sync = False
        self._post_timeout = 30

        # Wait time and retry frequency if polling for the task to complete
        self._wait_time: int = 60
        self._retry_frequency: int = 1

        # Initialize the external client
        self._a2a_client = A2AClient(httpx_client=self._client, url=self.url)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Context manager exit method
        """
        await self._client.aclose()

    async def send_task(self, payload: dict[str, Any]) -> SendMessageResponse:
        """
        Send a single (non-streaming) task request and return the response
        """
        send_message_payload = {
            'message': {
                'role': 'user',
                'parts': [{
                    'kind': 'text', 'text': payload.get('message', '')
                }],
                'messageId': uuid4().hex,
            },
            'sessionId': self._session_id,
            'acceptedOutputModes': ['text'],
            'metadata': payload.get('metadata')
        }
        request = SendMessageRequest(params=MessageSendParams(**send_message_payload))
        return await self._a2a_client.send_message(request)

    async def send_task_streaming_async(self, payload: dict[str, Any]) -> AsyncIterable[SendStreamingMessageResponse]:
        """
        Send a task streaming request asynchronously.
        """
        send_message_payload = {
            'message': {
                'role': 'user',
                'parts': [{
                    'kind': 'text', 'text': payload.get('message', '')
                }],
                'messageId': uuid4().hex,
            },
            'sessionId': self._session_id,
            'acceptedOutputModes': ['text'],
            'metadata': payload.get('metadata')
        }
        request = SendStreamingMessageRequest(params=MessageSendParams(**send_message_payload))
        async for response in self._a2a_client.send_message_streaming(request):
            yield response

    async def send_task_streaming_sync(self, payload: dict[str, Any]) -> AsyncIterable[SendStreamingMessageResponse]:
        """
        Send a task streaming request synchronously. Not used by default.
        """
        send_message_payload = {
            'message': {
                'role': 'user',
                'parts': [{
                    'kind': 'text', 'text': payload.get('message', '')
                }],
                'messageId': uuid4().hex,
            },
            'sessionId': self._session_id,
            'acceptedOutputModes': ['text'],
            'metadata': payload.get('metadata')
        }
        request = SendStreamingMessageRequest(params=MessageSendParams(**send_message_payload))
        async for response in self._a2a_client.send_message_streaming(request):
            yield response

    async def get_task(self, payload: dict[str, Any]) -> GetTaskResponse:
        """
        Get the task status and artifact by id
        """
        request = GetTaskRequest(params={"id": payload["id"]})
        return await self._a2a_client.get_task(request)

    async def get_task_with_retry(self, payload: dict[str, Any]) -> GetTaskResponse:
        """
        Get the task status and artifact by id with retry logic
        """
        wait_time = self._wait_time
        retry_frequency = self._retry_frequency if self._retry_frequency < wait_time else wait_time

        for _ in range(wait_time // retry_frequency):
            result = await self.get_task(payload)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Get task result: %s", result.model_dump_json(exclude_none=True))

            if result.result and result.result.status.state == TaskState.COMPLETED:
                return result
            await asyncio.sleep(retry_frequency)
        return result

    async def cancel_task(self, payload: dict[str, Any]) -> CancelTaskResponse:
        """
        Cancel a task by id
        """
        request = CancelTaskRequest(params={"id": payload["id"]})
        return await self._a2a_client.cancel_task(request)

    async def get_card(self) -> AgentCard:
        """
        Get the AgentCard from the server using a well-known path
        The agent card provides:
        1. Function description
        2. Input schema
        3. Output schema
        4. Capabilities (streaming, push notifications etc.)
        5. Authentication
        """
        resolver = A2ACardResolver(
            httpx_client=self._client,
            base_url=self.url,
        )
        self.agent_card = await resolver.get_agent_card()
        return self.agent_card

    def artifact_to_output_string(self, artifact: Artifact) -> str:
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

    def set_post_config(self, post_timeout: int, post_sync: bool):
        """
        Set the post configuration.
        """
        self._post_timeout = post_timeout
        self._post_sync = post_sync

    def set_wait_time(self, wait_timeout: int, retry_frequency: int):
        """
        Set the wait time and retry frequency. These are used by the complete_task() method
        to wait for the task to complete.
        """
        self._wait_time = wait_timeout
        self._retry_frequency = retry_frequency

    async def complete_task(self, taskId: str, prompt: str) -> dict[str, Any]:
        """
        Create a new task, wait for it to complete, and return the parsed result from the artifact
        """
        message = {
            "role": "user", "parts": [{
                "type": "text",
                "text": prompt,
            }]
        }
        payload = {
            "id": taskId,
            "sessionId": self._session_id,
            "acceptedOutputModes": ["text"],
            "message": message,
        }

        streaming = self.agent_card.capabilities.streaming
        artifacts = []
        final_state = None

        if streaming:
            # set async generator based on post_sync configuration
            if self._post_sync:
                response_stream = self.send_task_streaming_sync(payload)
            else:
                response_stream = self.send_task_streaming_async(payload)

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
                taskResult = await self.get_task_with_retry({"id": taskId})
                try:
                    artifact = taskResult.result.artifacts[0]
                except IndexError:
                    artifact = None
                except AttributeError:
                    artifact = None

            return self.artifact_to_output_string(artifact)
        else:
            taskResult = await self.send_task(payload)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Task result: %s", taskResult.model_dump_json(exclude_none=True))

            state = getattr(taskResult.result.status, "name", None)
            if state == TaskState.INPUT_REQUIRED.name:
                # TODO: Handle input required
                return await self.complete_task(taskId=taskId, prompt=taskResult.result.status.message.parts[0].text)
            else:
                return self.artifact_to_output_string(taskResult.result.artifacts[0])


class A2AToolClient(A2AClientHelper):
    """
    A client for the A2A API that includes tool schema information.
    """

    def __init__(self, url: str | None = None, tool_name: str | None = None, tool_input_schema: dict | None = None):
        """
        Initialize the A2AToolClient
        """
        # Setup the A2AClientHelper using the provided URL
        super().__init__(url)

        # Setup tool attributes
        self._tool_name: str | None = tool_name
        self._tool_description: str | None = None
        # TODO: Create the input schema from the info in the AgentCard
        self._input_schema: type[BaseModel] | None = None

    @property
    def name(self):
        return self._tool_name

    @property
    def description(self):
        return self._tool_description

    @property
    def input_schema(self):
        return self._input_schema

    def set_description(self, description: str):
        """
        Set the description for the tool. If a description is provided, it will be used. Otherwise,
        the description from the AgentCard will be used.
        """
        if description:
            self._tool_description = description
        elif self.agent_card:
            self._tool_description = self.agent_card.description

    async def acall(self, tool_input: str) -> str:
        """
        Create a new task for each tool call
        """
        taskId = uuid4().hex
        logger.debug("Start new Task ID: %s", taskId)

        output = await self.complete_task(taskId=taskId, prompt=tool_input)
        logger.debug("Task result: %s", output)
        return output
