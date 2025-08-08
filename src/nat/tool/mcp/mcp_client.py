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
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import TextContent
from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model

from nat.tool.mcp.exceptions import MCPToolNotFoundError
from nat.utils.exception_handlers.mcp import mcp_exception_handler

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
    required_fields = set(mcp_input_schema.get("required", []))
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
                item_type = model_from_mcp_schema(name=field_name, mcp_input_schema=item_properties)
            else:
                item_type = _type_map.get(item_properties.get("type", "string"), Any)
            field_type = list[item_type]
        elif isinstance(json_type, list):
            field_type = None
            for t in json_type:
                mapped = _type_map.get(t, Any)
                field_type = mapped if field_type is None else field_type | mapped

            return field_type, Field(
                default=field_properties.get("default", None if "null" in json_type else ...),
                description=field_properties.get("description", "")
            )
        else:
            field_type = _type_map.get(json_type, Any)

        # Determine the default value based on whether the field is required
        if field_name in required_fields:
            # Field is required - use explicit default if provided, otherwise make it required
            default_value = field_properties.get("default", ...)
        else:
            # Field is optional - use explicit default if provided, otherwise None
            default_value = field_properties.get("default", None)
            # Make the type optional if no default was provided
            if "default" not in field_properties:
                field_type = field_type | None

        nullable = field_properties.get("nullable", False)
        description = field_properties.get("description", "")

        field_type = field_type | None if nullable else field_type

        return field_type, Field(default=default_value, description=description)

    for field_name, field_props in properties.items():
        schema_dict[field_name] = _generate_field(field_name=field_name, field_properties=field_props)
    return create_model(f"{_generate_valid_classname(name)}InputSchema", **schema_dict)


class MCPSSEClient:
    """
    Client for creating a session and connecting to an MCP server using SSE

    Args:
      url (str): The url of the MCP server
    """

    def __init__(self, url: str):
        self.url = url

    @asynccontextmanager
    async def connect_to_sse_server(self):
        """
        Establish a session with an MCP SSE server within an aync context
        """
        async with sse_client(url=self.url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


class MCPBuilder(MCPSSEClient):
    """
    Builder class used to connect to an MCP Server and generate ToolClients

    Args:
        url (str): The url of the MCP server
    """

    def __init__(self, url):
        super().__init__(url)
        self._tools = None

    @mcp_exception_handler
    async def get_tools(self):
        """
        Retrieve a dictionary of all tools served by the MCP server.

        Returns:
            Dict of tool name to MCPToolClient

        Raises:
            MCPError: If connection or tool retrieval fails
        """
        async with self.connect_to_sse_server() as session:
            response = await session.list_tools()

        return {
            tool.name: MCPToolClient(self.url, tool.name, tool.description, tool_input_schema=tool.inputSchema)
            for tool in response.tools
        }

    @mcp_exception_handler
    async def get_tool(self, tool_name: str) -> MCPToolClient:
        """
        Get an MCP Tool by name.

        Args:
            tool_name (str): Name of the tool to load.

        Returns:
            MCPToolClient for the configured tool.

        Raises:
            MCPToolNotFoundError: If no tool is available with that name
            MCPError: If connection fails
        """
        if not self._tools:
            self._tools = await self.get_tools()

        tool = self._tools.get(tool_name)
        if not tool:
            raise MCPToolNotFoundError(tool_name, self.url)
        return tool

    @mcp_exception_handler
    async def call_tool(self, tool_name: str, tool_args: dict | None):
        async with self.connect_to_sse_server() as session:
            result = await session.call_tool(tool_name, tool_args)
            return result


class MCPToolClient(MCPSSEClient):
    """
    Client wrapper used to call an MCP tool.

    Args:
        url (str): The url of the MCP server
        tool_name (str): The name of the tool to wrap
        tool_description (str): The description of the tool provided by the MCP server.
        tool_input_schema (dict): The input schema for the tool.
    """

    def __init__(self, url: str, tool_name: str, tool_description: str | None, tool_input_schema: dict | None = None):
        super().__init__(url)
        self._tool_name = tool_name
        self._tool_description = tool_description
        self._input_schema = model_from_mcp_schema(self._tool_name, tool_input_schema) if tool_input_schema else None

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

    @mcp_exception_handler
    async def acall(self, tool_args: dict) -> str:
        """
        Call the MCP tool with the provided arguments.

        Args:
            tool_args (dict[str, Any]): A dictionary of key value pairs to serve as inputs for the MCP tool.
        """
        async with self.connect_to_sse_server() as session:
            result = await session.call_tool(self._tool_name, tool_args)

        output = []
        for res in result.content:
            if isinstance(res, TextContent):
                output.append(res.text)
            else:
                # Log non-text content for now
                logger.warning("Got not-text output from %s of type %s", self.name, type(res))
        return "\n".join(output)
