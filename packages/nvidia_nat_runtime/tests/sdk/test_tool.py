# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for nat.sdk.tool.tool — Tool and @tool decorator."""

import pytest

from nat.sdk.tool.result import ToolResult
from nat.sdk.tool.tool import Tool
from nat.sdk.tool.tool import tool

# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class TestTool:

    def test_creation(self) -> None:
        t = Tool(name="test", description="A test tool")
        assert t.name == "test"
        assert t.description == "A test tool"
        assert t.parameters == {"type": "object", "properties": {}}

    def test_frozen(self) -> None:
        t = Tool(name="test", description="test")
        with pytest.raises(AttributeError):
            t.name = "other"  # type: ignore[misc]

    def test_to_openai_schema(self) -> None:
        t = Tool(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"],
            },
        )
        schema = t.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search the web"
        assert "query" in schema["function"]["parameters"]["properties"]

    async def test_call_sync_returning_tool_result(self) -> None:

        def _fn(x: int) -> ToolResult:
            return ToolResult(output=x * 2)

        t = Tool(name="double", description="Double", execute=_fn)
        result = await t(x=5)
        assert result.output == 10
        assert not result.is_error

    async def test_call_async_returning_tool_result(self) -> None:

        async def _fn(x: int) -> ToolResult:
            return ToolResult(output=x * 2)

        t = Tool(name="double", description="Double", execute=_fn)
        result = await t(x=5)
        assert result.output == 10

    async def test_call_sync_returning_raw_value(self) -> None:

        def _fn(x: int) -> int:
            return x * 3

        t = Tool(name="triple", description="Triple", execute=_fn)
        result = await t(x=4)
        assert result.output == 12

    async def test_call_async_returning_raw_value(self) -> None:

        async def _fn(x: int) -> str:
            return f"result={x}"

        t = Tool(name="format", description="Format", execute=_fn)
        result = await t(x=7)
        assert result.output == "result=7"

    async def test_call_exception_becomes_error(self) -> None:

        def _fn() -> None:
            raise ValueError("boom")

        t = Tool(name="fail", description="Fail", execute=_fn)
        result = await t()
        assert result.is_error
        assert "boom" in result.error


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------


class TestToolDecorator:

    def test_basic_decorator(self) -> None:

        @tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        assert isinstance(add, Tool)
        assert add.name == "add"
        assert add.description == "Add two numbers"
        assert "a" in add.parameters["properties"]
        assert "b" in add.parameters["properties"]
        assert "a" in add.parameters["required"]
        assert "b" in add.parameters["required"]

    def test_custom_name(self) -> None:

        @tool(description="Search", name="web_search")
        def search(query: str) -> str:
            return query

        assert search.name == "web_search"

    def test_schema_from_type_hints(self) -> None:

        @tool(description="Test types")
        def typed_fn(
            name: str,
            count: int,
            rate: float,
            flag: bool,
            items: list,
            data: dict,
        ) -> str:
            return ""

        props = typed_fn.parameters["properties"]
        assert props["name"]["type"] == "string"
        assert props["count"]["type"] == "integer"
        assert props["rate"]["type"] == "number"
        assert props["flag"]["type"] == "boolean"
        assert props["items"]["type"] == "array"
        assert props["data"]["type"] == "object"

    def test_optional_params_not_required(self) -> None:

        @tool(description="Test")
        def fn(required_arg: str, optional_arg: str = "default") -> str:
            return ""

        assert "required_arg" in fn.parameters["required"]
        assert "optional_arg" not in fn.parameters.get("required", [])

    async def test_execute_sync_function(self) -> None:

        @tool(description="Add")
        def add(a: int, b: int) -> int:
            return a + b

        result = await add(a=3, b=4)
        assert result.output == 7

    async def test_execute_async_function(self) -> None:

        @tool(description="Greet")
        async def greet(name: str) -> str:
            return f"Hello {name}"

        result = await greet(name="World")
        assert result.output == "Hello World"

    def test_camel_to_snake_conversion(self) -> None:

        @tool(description="Test")
        def MyToolName() -> str:
            return ""

        assert MyToolName.name == "my_tool_name"

    async def test_exception_captured(self) -> None:

        @tool(description="Fail")
        def bad() -> None:
            raise RuntimeError("oops")

        result = await bad()
        assert result.is_error
        assert "oops" in result.error
