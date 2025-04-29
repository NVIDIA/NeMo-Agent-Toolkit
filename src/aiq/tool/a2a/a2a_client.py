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
import json
import logging
from collections.abc import AsyncIterable
from typing import Any
from uuid import uuid4

import httpx
from httpx_sse import connect_sse
from pydantic import BaseModel

from .types import A2AClientHTTPError
from .types import A2AClientJSONError
from .types import AgentCard
from .types import Artifact
from .types import CancelTaskRequest
from .types import CancelTaskResponse
from .types import GetTaskPushNotificationRequest
from .types import GetTaskPushNotificationResponse
from .types import GetTaskRequest
from .types import GetTaskResponse
from .types import JSONRPCRequest
from .types import JSONRPCResponse
from .types import SendTaskRequest
from .types import SendTaskResponse
from .types import SendTaskStreamingRequest
from .types import SendTaskStreamingResponse
from .types import SetTaskPushNotificationRequest
from .types import SetTaskPushNotificationResponse
from .types import TaskArtifactUpdateEvent
from .types import TaskState
from .types import TaskStatusUpdateEvent

logger = logging.getLogger(__name__)


class A2AClient:
    """
    A client for the A2A API. Uses A2A.samples.python.common.client.client.A2AClient as a reference.
    """

    def __init__(self, url: str):
        """
        Initialize the A2AClient
        """
        self.url = url.rstrip("/")
        self.agent_card = None

        self._client = httpx.AsyncClient(base_url=self.url)
        # Use a single session id for all tasks submitted by this client
        self._session_id: str = uuid4().hex

        self._post_timeout = 30
        self._wait_time: int = 0
        self._retry_frequency: int = 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Context manager exit method
        """
        await self._client.aclose()

    async def _send_request(self, request: JSONRPCRequest) -> dict[str, Any]:
        """
        Send a JSONRPC request and return the response
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Request payload: %s", request.model_dump_json(indent=2))

        try:
            response = await self._client.post(self.url, json=request.model_dump(), timeout=self._post_timeout)
            response.raise_for_status()
            data = response.json()
            rpc_response = JSONRPCResponse(**data)
            if rpc_response.error:
                raise A2AClientHTTPError(rpc_response.error.get("code", 400),
                                         rpc_response.error.get("message", "Unknown error"))
            return rpc_response.result or {}
        except httpx.HTTPStatusError as e:
            # log and raise
            logger.error("HTTP Error: %s", e, exc_info=True)
            raise A2AClientHTTPError(e.response.status_code, str(e)) from e
        except json.JSONDecodeError as e:
            # log and raise
            logger.error("Error parsing JSON: %s", e, exc_info=True)
            raise A2AClientJSONError(str(e)) from e

    async def send_task(self, payload: dict[str, Any]) -> SendTaskResponse:
        """
        Send a single (non-streaming) task request and return the response
        """
        request = SendTaskRequest(params=payload)
        return SendTaskResponse(**await self._send_request(request))

    async def send_task_streaming_sync(self, payload: dict[str, Any]) -> AsyncIterable[SendTaskStreamingResponse]:
        """
        Send a task streaming request synchronously.
        The caller must parse the intermediate events as they come in.

        This approach is from A2A.samples.python.common.client.client.A2AClient.send_task_streaming().
        """
        request = SendTaskStreamingRequest(params=payload)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Request payload: %s", request.model_dump_json(indent=2))

        # Use a separate httpx client to avoid timeout issues.
        with httpx.Client(timeout=None) as client:
            with connect_sse(client, "POST", self.url, json=request.model_dump()) as event_source:
                try:
                    for sse in event_source.iter_sse():
                        yield SendTaskStreamingResponse(**json.loads(sse.data))
                except json.JSONDecodeError as e:
                    raise A2AClientJSONError(str(e)) from e
                except httpx.RequestError as e:
                    raise A2AClientHTTPError(400, str(e)) from e

    async def send_task_streaming_async(self, payload: dict[str, Any]) -> AsyncIterable[SendTaskStreamingResponse]:
        """
        Send a task streaming request asynchronously. The caller must use get_task() to get the final result
        """
        request = SendTaskStreamingRequest(params=payload)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Request payload: %s", request.model_dump_json(indent=2))

        async with self._client.stream("POST", self.url, json=request.model_dump(),
                                       timeout=self._post_timeout) as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    try:
                        event_data = line[len("data:"):].strip()
                        response_obj = JSONRPCResponse(**json.loads(event_data))
                        if response_obj.error:
                            logger.error("Error in streaming response: %s", response_obj.error, exc_info=True)
                            raise A2AClientHTTPError(response_obj.error.get("code", 400),
                                                     response_obj.error.get("message", "Unknown error"))
                        yield SendTaskStreamingResponse(**response_obj.result)
                    except json.JSONDecodeError as e:
                        logger.error("Error parsing JSON: %s", e, exc_info=True)
                        raise A2AClientJSONError(str(e)) from e

    async def send_task_streaming(self, payload: dict[str, Any],
                                  use_sync: bool) -> AsyncIterable[SendTaskStreamingResponse]:
        """
        Send a task streaming request.
        If use_sync is True, the caller must parse the intermediate events as they come in.
        If use_sync is False, the caller must use get_task() to get the final result
        """
        if use_sync:
            return self.send_task_streaming_sync(payload)
        else:
            return self.send_task_streaming_async(payload)

    async def get_task(self, payload: dict[str, Any]) -> GetTaskResponse:
        """
        Get the task status and artifact by id
        """
        request = GetTaskRequest(params=payload)
        return GetTaskResponse(**await self._send_request(request))

    async def get_task_with_retry(self,
                                  payload: dict[str, Any],
                                  wait_time: int = 60,
                                  retry_frequency: int = 1) -> GetTaskResponse:
        """
        Get the task status and artifact by id with retry logic
        """
        for _ in range(wait_time // retry_frequency):

            result = await self.get_task(payload)
            if result.result.status.state == TaskState.COMPLETED.name:
                return result
            await asyncio.sleep(retry_frequency)
        return result

    async def cancel_task(self, payload: dict[str, Any]) -> CancelTaskResponse:
        """
        Cancel a task by id
        """
        request = CancelTaskRequest(params=payload)
        return CancelTaskResponse(**await self._send_request(request))

    async def set_task_callback(self, payload: dict[str, Any]) -> SetTaskPushNotificationResponse:
        """
        Set a task callback. This is only used if the client needs to be
        notified when a task is completed.
        """
        request = SetTaskPushNotificationRequest(params=payload)
        return SetTaskPushNotificationResponse(**await self._send_request(request))

    async def get_task_callback(self, payload: dict[str, Any]) -> GetTaskPushNotificationResponse:
        """
        Get the task callback configuration.
        """
        request = GetTaskPushNotificationRequest(params=payload)
        return GetTaskPushNotificationResponse(**await self._send_request(request))

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

    def set_wait_time(self, wait_time: int, retry_frequency: int):
        """
        Set the wait time and retry frequency. These are used by the complete_task() method
        to wait for the task to complete.
        """
        self._wait_time = wait_time
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
        artifact = None
        final_state = None

        if streaming:
            use_sync = self._wait_time == 0
            response_stream = await self.send_task_streaming(payload, use_sync=use_sync)
            async for result in response_stream:
                # Parse event contents if they are not empty -
                # 1. events are empty if use_sync=False and get_task() is called to
                # get the final result
                # 2. events are non-empty if use_sync=True
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Stream event: %s", result.model_dump_json(exclude_none=True))

                if isinstance(result.result, TaskStatusUpdateEvent):
                    final_state = result.result.status.state
                    if final_state == TaskState.COMPLETED.name:
                        # This is the last event and the task is complete
                        break
                elif isinstance(result.result, TaskArtifactUpdateEvent):
                    if result.result.artifact:
                        artifact = result.result.artifact

            if artifact is None:
                # If the artifact was not present in the streaming response,
                # get the final result from the server
                taskResult = await self.get_task({"id": taskId})
                artifact = taskResult.result.artifacts[0] if taskResult.result.artifacts else None

            return self.artifact_to_output_string(artifact)
        else:
            taskResult = await self.send_task(payload)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Task result: %s", taskResult.model_dump_json(exclude_none=True))

            state = getattr(taskResult.result.status, "name", None)
            if state == TaskState.INPUT_REQUIRED.name:
                # TODO: Handle input required
                return await self.complete_task(taskId=taskId,
                                                sessionId=self._session_id,
                                                prompt=taskResult.result.status.message.parts[0].text)
            else:
                return self.artifact_to_output_string(taskResult.result.artifacts[0])

    async def get_card(self) -> AgentCard:
        """
        Get the AgentCard from the server using a well-known path
        The agent card provides:
        1. Function description
        2. Input schema
        3. Output schema
        4. Capabilities (streaming, push notifications etc.)
        5. Authentication

        TODO: More work is needed to use the info in the agent card better.
        """
        base_url = self.url.rstrip("/")
        agent_card_path = "/.well-known/agent.json".lstrip("/")
        response = await self._client.get(f"{base_url}/{agent_card_path}")
        response.raise_for_status()
        try:
            logger.info("AgentCard: %s", response.json())
            self.agent_card = AgentCard(**response.json())
        except json.JSONDecodeError as e:
            # log and raise
            logger.error("Error parsing AgentCard: %s", e, exc_info=True)
            raise A2AClientJSONError(str(e)) from e


class A2AToolClient(A2AClient):

    def __init__(self, url: str | None = None, tool_name: str | None = None, tool_input_schema: dict | None = None):
        """
        Initialize the A2AToolClient
        """
        # Setup the A2AClient using the provided URL
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
        output = await self.complete_task(taskId=uuid4().hex, prompt=tool_input)
        logger.debug("Task result: %s", output)
        return output
