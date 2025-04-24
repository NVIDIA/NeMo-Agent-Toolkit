import json
import logging
from typing import Any
from typing import AsyncIterable
from uuid import uuid4

import httpx
from httpx_sse import connect_sse_async
from pydantic import BaseModel
from pydantic import Enum
from pydantic import Field
from pydantic import create_model

from .types import AgentCard

logger = logging.getLogger(__name__)


# --- Error Types ---
class A2AClientError(Exception):
    pass


class A2AClientHTTPError(A2AClientError):

    def __init__(self, status_code: int, message: str):
        super().__init__(f"HTTP Error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class A2AClientJSONError(A2AClientError):

    def __init__(self, message: str):
        super().__init__(f"JSON Error: {message}")
        self.message = message


# --- JSON-RPC Base Models ---
class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any]
    id: str = Field(default_factory=lambda: str(uuid4()))


class JSONRPCResponse(BaseModel):
    jsonrpc: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    id: str


class A2AClient:

    def __init__(self, url: str):
        """
        Initialize the A2AClient
        """
        self.url = url.rstrip("/")
        self.agent_card = None

        self._client = httpx.AsyncClient(base_url=self.url)
        # Make the timeout configurable
        self._timeout = 30

    async def __aexit__(self, exc_type, exc, tb):
        """
        Context manager exit method
        """
        await self._client.aclose()

    async def _send_request(self, method: str, params: dict[str, Any] = None) -> dict[str, Any]:
        request = JSONRPCRequest(method=method, params=params or {})
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

    async def send_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._send_request("send_task", payload)

    async def send_task_streaming(self, payload: dict[str, Any]) -> AsyncIterable[dict[str, Any]]:
        request = JSONRPCRequest(method="send_task_streaming", params=payload)
        async with connect_sse_async(self._client, "POST", self.url, json=request.model_dump()) as event_source:
            async for sse in event_source.aiter_sse():
                try:
                    response = JSONRPCResponse(**json.loads(sse.data))
                    if response.error:
                        # log and raise
                        logger.error("Error in streaming response: %s", response.error, exc_info=True)
                        raise A2AClientHTTPError(response.error.get("code", 400),
                                                 response.error.get("message", "Unknown error"))
                    yield response.result or {}
                except json.JSONDecodeError as e:
                    # log and raise
                    logger.error("Error parsing JSON: %s", e, exc_info=True)
                    raise A2AClientJSONError(str(e)) from e

    async def get_card(self) -> AgentCard:
        """Get the AgentCard from the server"""
        base_url = self.url.rstrip("/")
        agent_card_path = "/.well-known/agent.json".lstrip("/")
        response = await self._client.get(f"{base_url}/{agent_card_path}")
        response.raise_for_status()
        try:
            return AgentCard(**response.json())
        except json.JSONDecodeError as e:
            # log and raise
            logger.error("Error parsing AgentCard: %s", e, exc_info=True)
            raise A2AClientJSONError(str(e)) from e

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
        taskResult = None
        streaming = self.agent_card.capabilities.streaming
        if streaming:
            response_stream = self.send_task_streaming(payload)
            async for result in response_stream:
                print(f"stream event => {result.model_dump_json(exclude_none=True)}")
        else:
            taskResult = await self.send_task(payload)
        return taskResult


class A2AToolClient(A2AClient):

    def __init__(self,
                 agent_card: AgentCard | None = None,
                 url: str | None = None,
                 tool_name: str | None = None,
                 tool_input_schema: dict | None = None):
        """
        Initialize the A2AToolClient
        """
        # Setup the A2AClient using the provided URL
        super().__init__(agent_card, url)

        # Setup tool attributes
        self._tool_name: str | None = tool_name
        self._tool_description: str | None = None
        self._input_schema: type[BaseModel] | None = \
            self.model_from_a2a_schema(self._tool_name, tool_input_schema) if tool_input_schema else None

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

    async def acall(self, tool_args: dict) -> str:
        """
        Call the tool
        """
        return await self.complete_task(taskId=uuid4().hex, sessionId=uuid4().hex, prompt=tool_args)
