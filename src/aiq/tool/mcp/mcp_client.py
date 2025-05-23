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
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model

logger = logging.getLogger(__name__)


def model_from_mcp_schema(name: str, mcp_input_schema: dict) -> type[BaseModel]:
    """
    Create a pydantic model from the input schema of the MCP tool
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

    properties = mcp_input_schema.get("properties", {})
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
            field_type = model_from_mcp_schema(name=field_name, mcp_input_schema=field_properties)
        elif json_type == "array" and "items" in field_properties:
            item_properties = field_properties.get("items", {})
            if item_properties.get("type") == "object":
                item_type = model_from_mcp_schema(name=field_name, mcp_input_schema=field_properties)
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


class MCPBaseClient(ABC):
    """
    Base client for creating a session and connecting to an MCP server
    """

    def __init__(self, client_type: str = 'sse'):
        self._tools = None
        self._client_type = client_type.lower()
        if self._client_type not in ['sse', 'stdio']:
            raise ValueError("client_type must be either 'sse' or 'stdio'")

    @abstractmethod
    @asynccontextmanager
    async def connect_to_server(self):
        """
        Establish a session with an MCP server within an async context
        """
        pass

    async def get_tools(self):
        """
        Retrieve a dictionary of all tools served by the MCP server.
        """
        async with self.connect_to_server() as session:
            response = await session.list_tools()

        return {
            tool.name:
                MCPToolClient(connect_fn=self.connect_to_server,
                              tool_name=tool.name,
                              tool_description=tool.description,
                              tool_input_schema=tool.inputSchema)
            for tool in response.tools
        }

    async def get_tool(self, tool_name: str) -> MCPToolClient:
        """
        Get an MCP Tool by name.

        Args:
            tool_name (str): Name of the tool to load.

        Returns:
            MCPToolClient for the configured tool.

        Raise:
            ValueError if no tool is available with that name.
        """
        if not self._tools:
            self._tools = await self.get_tools()

        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not available")
        return tool

    async def call_tool(self, tool_name: str, tool_args: dict | None):
        async with self.connect_to_server() as session:
            result = await session.call_tool(tool_name, tool_args)
            return result


class MCPSSEClient(MCPBaseClient):
    """
    Client for creating a session and connecting to an MCP server using SSE

    Args:
      url (str): The url of the MCP server
      client_type (str): The type of client to use ('sse' or 'stdio')
    """

    def __init__(self, url: str, client_type: str = 'sse'):
        super().__init__(client_type)
        self._url = url

    @asynccontextmanager
    async def connect_to_server(self):
        """
        Establish a session with an MCP SSE server within an async context
        """
        async with sse_client(url=self._url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


class MCPStdioClient(MCPBaseClient):
    """
    Client for creating a session and connecting to an MCP server using stdio

    Args:
      command (str): The command to run
      args (list[str] | None): Additional arguments for the command
      env (dict[str, str] | None): Environment variables to set for the process
      client_type (str): The type of client to use ('sse' or 'stdio')
    """

    def __init__(self,
                 command: str,
                 args: list[str] | None = None,
                 env: dict[str, str] | None = None,
                 client_type: str = 'stdio'):
        super().__init__(client_type)
        self._command = command
        self._args = args
        self._env = env
        self._session = None  # hold session if persistent
        self._session_cm = None

    async def start_persistent_session(self):
        """Starts and holds a persistent session."""
        server_params = StdioServerParameters(command=self._command, args=self._args, env=self._env)
        self._session_cm = stdio_client(server_params)
        read, write = await self._session_cm.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.initialize()

    async def stop_persistent_session(self):
        """Ends the persistent session."""
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._session_cm:
            await self._session_cm.__aexit__(None, None, None)
            self._session_cm = None

    @asynccontextmanager
    async def connect_to_server(self):
        """
        Establish a session with an MCP server via stdio within an async context
        """
        if self._session:
            yield self._session
        else:
            server_params = StdioServerParameters(command=self._command, args=self._args or [], env=self._env)
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session


class MCPToolClient:
    """
    Client wrapper used to call an MCP tool.

    Args:
        connect_fn (callable): Function that returns an async context manager for connecting to the server
        tool_name (str): The name of the tool to wrap
        tool_description (str): The description of the tool provided by the MCP server.
        tool_input_schema (dict): The input schema for the tool.
    """

    def __init__(self,
                 connect_fn: Callable[[], AbstractAsyncContextManager[ClientSession]],
                 tool_name: str,
                 tool_description: str | None,
                 tool_input_schema: dict | None = None):
        self._connect_fn = connect_fn
        self._tool_name = tool_name
        self._tool_description = tool_description
        self._input_schema = (model_from_mcp_schema(self._tool_name, tool_input_schema) if tool_input_schema else None)

    @property
    def name(self):
        """Returns the name of the tool."""
        return self._tool_name

    @property
    def description(self):
        """
        Returns the tool's description. If none was provided. Provides a simple description using the tool's name
        """
        if not self._tool_description:
            return f"MCP Tool {self._tool_name}"
        return self._tool_description

    @property
    def input_schema(self):
        """
        Returns the tool's input_schema.
        """
        return self._input_schema

    def set_description(self, description: str):
        """
        Manually define the tool's description using the provided string.
        """
        self._tool_description = description

    async def acall(self, tool_args: dict) -> str:
        """
        Call the MCP tool with the provided arguments.

        Args:
            tool_args (dict[str, Any]): A dictionary of key value pairs to serve as inputs for the MCP tool.
        """
        async with self._connect_fn() as session:
            result = await session.call_tool(self._tool_name, tool_args)

        output = []

        for res in result.content:
            if isinstance(res, TextContent):
                output.append(res.text)
            else:
                # Log non-text content for now
                logger.warning("Got not-text output from %s of type %s", self.name, type(res))
        return "\n".join(output)
