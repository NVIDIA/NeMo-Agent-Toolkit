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

import json
import logging
from collections.abc import AsyncIterable
from enum import Enum
from typing import Any
from uuid import uuid4

import httpx
from httpx_sse import connect_sse
from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model

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
        # Make the timeout configurable
        self._timeout = 30

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Context manager exit method
        """
        await self._client.aclose()

    async def _send_request(self, request: JSONRPCRequest) -> dict[str, Any]:
        logger.info("Request payload: %s", request.model_dump_json(indent=2))

        try:
            response = await self._client.post(self.url, json=request.model_dump(), timeout=self._timeout)
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
        request = SendTaskRequest(params=payload)
        return SendTaskResponse(**await self._send_request(request))

    async def send_task_streaming_sync(self, payload: dict[str, Any]) -> AsyncIterable[SendTaskStreamingResponse]:
        request = SendTaskStreamingRequest(params=payload)
        logger.info("Request payload: %s", request.model_dump_json(indent=2))

        # use a se
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
        Send a task streaming request asynchronously - not working
        """
        request = SendTaskStreamingRequest(params=payload)
        logger.info("Request payload: %s", request.model_dump_json(indent=2))

        async with self._client.stream("POST", self.url, json=request.model_dump(), timeout=self._timeout) as response:
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

    async def send_task_streaming(self,
                                  payload: dict[str, Any],
                                  use_sync: bool = True) -> AsyncIterable[SendTaskStreamingResponse]:
        if use_sync:
            return self.send_task_streaming_sync(payload)
        else:
            return self.send_task_streaming_async(payload)

    async def get_task(self, payload: dict[str, Any]) -> GetTaskResponse:
        request = GetTaskRequest(params=payload)
        return GetTaskResponse(**await self._send_request(request))

    async def cancel_task(self, payload: dict[str, Any]) -> CancelTaskResponse:
        request = CancelTaskRequest(params=payload)
        return CancelTaskResponse(**await self._send_request(request))

    async def set_task_callback(self, payload: dict[str, Any]) -> SetTaskPushNotificationResponse:
        request = SetTaskPushNotificationRequest(params=payload)
        return SetTaskPushNotificationResponse(**await self._send_request(request))

    async def get_task_callback(self, payload: dict[str, Any]) -> GetTaskPushNotificationResponse:
        request = GetTaskPushNotificationRequest(params=payload)
        return GetTaskPushNotificationResponse(**await self._send_request(request))

    def artifact_to_output_string(self, artifact: Artifact) -> str:
        """
        This is a temporary helper
        """
        if not artifact:
            return "No artifact found"
        elif not artifact.parts:
            return "No artifact parts found"
        elif artifact.parts[0].type == "text":
            return artifact.parts[0].text
        elif artifact.parts[0].type == "file":
            return f"File: {artifact.parts[0].file.name}"
        else:
            return "Unknown artifact type"

    async def complete_task(self, taskId: str, sessionId: str, prompt: str) -> dict[str, Any]:
        """
        Complete a task
        """
        message = {
            "role": "user", "parts": [{
                "type": "text",
                "text": prompt,
            }]
        }
        payload = {
            "id": taskId,
            "sessionId": sessionId,
            "acceptedOutputModes": ["text"],
            "message": message,
        }

        streaming = self.agent_card.capabilities.streaming
        artifact = None
        final_state = None

        if streaming:
            response_stream = await self.send_task_streaming(payload, use_sync=True)
            async for result in response_stream:
                logger.info("Stream event: %s", result.model_dump_json(exclude_none=True))

                # Change to use the pydantic model
                if isinstance(result.result, TaskStatusUpdateEvent):
                    final_state = result.result.status.state
                    if result.result.status.state == TaskState.COMPLETED.name:
                        break
                elif isinstance(result.result, TaskArtifactUpdateEvent):
                    if result.result.artifact:
                        artifact = result.result.artifact

            if artifact is None or final_state != TaskState.COMPLETED:
                # TODO: Get task is not working
                taskResult = await self.get_task({"id": taskId})
                return taskResult.result.artifacts[0] if taskResult.result.artifacts else None

            return self.artifact_to_output_string(artifact)
        else:
            taskResult = await self.send_task(payload)
            logger.info("Task result: %s", taskResult.model_dump_json(exclude_none=True))

            state = getattr(taskResult.result.status, "name", None)
            if state == TaskState.INPUT_REQUIRED.name:
                # TODO: Handle input required
                return await self.complete_task(taskId=taskId,
                                                sessionId=sessionId,
                                                prompt=taskResult.result.status.message.parts[0].text)
            else:
                return self.artifact_to_output_string(taskResult.result.artifacts[0])

    async def get_card(self) -> AgentCard:
        """Get the AgentCard from the server"""
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
        self._input_schema: type[BaseModel] | None = \
            self.model_from_a2a_schema(self._tool_name, tool_input_schema) if tool_input_schema else None
        self._session_id: str = uuid4().hex

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

    def model_from_a2a_schema(self, name: str, a2a_input_schema: dict) -> type[BaseModel]:
        """
        Create a pydantic model from the input schema of the A2A tool
        Note: This is a simplified version of the model_from_mcp_schema function in mcp_client.py
        """
        _type_map = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "array": list,
            "null": None,
            "object": dict,
        }

        properties = a2a_input_schema.get("properties", {})
        schema_dict = {}

        def _generate_valid_classname(class_name: str):
            return class_name.replace('_', ' ').replace('-', ' ').title().replace(' ', '')

        def _generate_field(field_name: str, field_properties: dict[str, Any]) -> tuple:
            json_type = field_properties.get("type", "string")
            enum_vals = field_properties.get("enum")

            if enum_vals:
                enum_name = f"{field_name.capitalize()}Enum"
                field_type = Enum(enum_name, {item: item for item in enum_vals})

            elif json_type == "object" and "properties" in field_properties:
                field_type = self.model_from_a2a_schema(name=field_name, a2a_input_schema=field_properties)
            elif json_type == "array" and "items" in field_properties:
                item_properties = field_properties.get("items", {})
                if item_properties.get("type") == "object":
                    item_type = self.model_from_a2a_schema(name=field_name, a2a_input_schema=field_properties)
                else:
                    item_type = _type_map.get(json_type, Any)
                field_type = list[item_type]
            else:
                field_type = _type_map.get(json_type, Any)

            default_value = field_properties.get("default", ...)
            nullable = field_properties.get("nullable", False)
            description = field_properties.get("description", "")

            field_type = field_type | None if nullable else field_type

            return field_type, Field(default=default_value, description=description)

        for field_name, field_props in properties.items():
            schema_dict[field_name] = _generate_field(field_name=field_name, field_properties=field_props)
        return create_model(f"{_generate_valid_classname(name)}InputSchema", **schema_dict)

    async def acall(self, tool_input: str) -> str:
        """
        Call the tool
        """
        output = await self.complete_task(taskId=uuid4().hex, sessionId=self._session_id, prompt=tool_input)
        logger.info("Task result: %s", output)
        return output
