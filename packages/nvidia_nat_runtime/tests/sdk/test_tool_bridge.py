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
"""Tests for Tool bridge methods: from_function, from_function_config, from_function_group_config."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from pydantic import BaseModel

from nat.sdk.tool.tool import Tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyInput(BaseModel):
    query: str


def _make_mock_function(*, display_name: str = "my_func", description: str = "A mock function") -> MagicMock:
    """Create a mock core Function with the expected interface."""
    fn = MagicMock()
    fn.display_name = display_name
    fn.description = description
    fn.input_schema = _DummyInput
    fn.acall_invoke = AsyncMock(return_value="mock_result")
    return fn


# ---------------------------------------------------------------------------
# Tool.from_function
# ---------------------------------------------------------------------------


class TestToolFromFunction:

    def test_wraps_name_and_description(self) -> None:
        fn = _make_mock_function(display_name="search", description="Search the web")
        tool = Tool.from_function(fn)
        assert tool.name == "search"
        assert tool.description == "Search the web"

    def test_schema_from_input_schema(self) -> None:
        fn = _make_mock_function()
        tool = Tool.from_function(fn)
        assert "properties" in tool.parameters
        assert "query" in tool.parameters["properties"]

    async def test_execute_calls_acall_invoke(self) -> None:
        fn = _make_mock_function()
        tool = Tool.from_function(fn)
        result = await tool(query="hello")
        fn.acall_invoke.assert_called_once_with(query="hello")
        assert result.output == "mock_result"

    def test_none_description_becomes_empty_string(self) -> None:
        fn = _make_mock_function(description=None)
        fn.description = None
        tool = Tool.from_function(fn)
        assert tool.description == ""


# ---------------------------------------------------------------------------
# Tool.from_function_config
# ---------------------------------------------------------------------------


class TestToolFromFunctionConfig:

    async def test_uses_builder(self) -> None:
        """from_function_config should get Builder.current() and use it to build."""
        mock_fn = _make_mock_function(display_name="built_func")

        mock_builder = MagicMock()
        mock_builder.add_function = AsyncMock(return_value=mock_fn)

        mock_config = MagicMock()
        mock_config.name = "custom_name"

        with patch("nat.builder.builder.Builder.current", return_value=mock_builder):
            tool = await Tool.from_function_config(mock_config)

        assert tool.name == "built_func"
        mock_builder.add_function.assert_awaited_once()

    async def test_auto_generates_name_when_config_name_is_none(self) -> None:
        mock_fn = _make_mock_function()
        mock_builder = MagicMock()
        mock_builder.add_function = AsyncMock(return_value=mock_fn)

        mock_config = MagicMock()
        mock_config.name = None

        with patch("nat.builder.builder.Builder.current", return_value=mock_builder):
            await Tool.from_function_config(mock_config)

        # Verify the name passed to add_function starts with sdk_fn_
        mock_builder.add_function.assert_awaited_once()
        name_arg = mock_builder.add_function.call_args[0][0]
        assert name_arg.startswith("sdk_fn_")


# ---------------------------------------------------------------------------
# Tool.from_function_group
# ---------------------------------------------------------------------------


class TestToolFromFunctionGroup:

    async def test_wraps_accessible_functions(self) -> None:
        fn1 = _make_mock_function(display_name="func_x", description="X")
        fn2 = _make_mock_function(display_name="func_y", description="Y")
        mock_functions = {"grp__func_x": fn1, "grp__func_y": fn2}

        mock_group = MagicMock()
        mock_group.get_accessible_functions = AsyncMock(return_value=mock_functions)

        tools = await Tool.from_function_group(mock_group)

        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"func_x", "func_y"}
        mock_group.get_accessible_functions.assert_awaited_once()

    async def test_empty_group_returns_empty_list(self) -> None:
        mock_group = MagicMock()
        mock_group.get_accessible_functions = AsyncMock(return_value={})

        tools = await Tool.from_function_group(mock_group)

        assert tools == []


# ---------------------------------------------------------------------------
# Tool.from_function_group_config
# ---------------------------------------------------------------------------


class TestToolFromFunctionGroupConfig:

    async def test_returns_list_of_tools(self) -> None:
        fn1 = _make_mock_function(display_name="func_a", description="A")
        fn2 = _make_mock_function(display_name="func_b", description="B")

        mock_group = MagicMock()
        mock_group.get_accessible_functions = AsyncMock(return_value={"group__func_a": fn1, "group__func_b": fn2})

        mock_builder = MagicMock()
        mock_builder.add_function_group = AsyncMock(return_value=mock_group)

        mock_config = MagicMock()

        with patch("nat.builder.builder.Builder.current", return_value=mock_builder):
            tools = await Tool.from_function_group_config(mock_config)

        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"func_a", "func_b"}

    async def test_empty_group_returns_empty_list(self) -> None:
        mock_group = MagicMock()
        mock_group.get_accessible_functions = AsyncMock(return_value={})

        mock_builder = MagicMock()
        mock_builder.add_function_group = AsyncMock(return_value=mock_group)

        mock_config = MagicMock()

        with patch("nat.builder.builder.Builder.current", return_value=mock_builder):
            tools = await Tool.from_function_group_config(mock_config)

        assert tools == []


# ---------------------------------------------------------------------------
# Auto-incrementing counter
# ---------------------------------------------------------------------------


class TestToolIdCounter:

    def test_counter_produces_unique_values(self) -> None:
        """The module-level counter should produce strictly increasing values."""
        from nat.sdk.tool.tool import _tool_id_counter

        val1 = next(_tool_id_counter)
        val2 = next(_tool_id_counter)
        assert val2 > val1
