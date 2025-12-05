# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

import logging
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.cli.register_workflow import register_tool_wrapper
from strands.types.tools import AgentTool  # type: ignore
from strands.types.tools import ToolSpec  # type: ignore
from strands.types.tools import ToolUse  # type: ignore

logger = logging.getLogger(__name__)


def _json_schema_from_pydantic(model: type[BaseModel]) -> dict[str, Any]:
    try:
        schema = model.model_json_schema()
        for k in ("title", "additionalProperties"):
            if k in schema:
                del schema[k]
        return {"json": schema}
    except Exception:
        logger.exception("Failed to generate JSON schema")
        return {"json": {}}


def _to_tool_result(tool_use_id: str, value: Any) -> dict[str, Any]:
    if isinstance(value, (dict, list, tuple)):  # noqa: UP038
        content_item = {"json": value}
    else:
        content_item = {"text": str(value)}
    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [content_item],
    }


def _to_error_result(tool_use_id: str, err: Exception) -> dict[str, Any]:
    return {
        "toolUseId": tool_use_id,
        "status": "error",
        "content": [{
            "text": f"{type(err).__name__}: {err!s}"
        }],
    }


class NATFunctionAgentTool(AgentTool):
    """Concrete Strands AgentTool that wraps a NAT Function."""

    def __init__(self, name: str, description: str | None, input_schema: dict[str, Any], fn: Function) -> None:
        super().__init__()

        self._tool_name = name
        self._tool_spec: ToolSpec = {
            "name": name,
            "description": description or name,
            "inputSchema": input_schema,
        }
        self._fn = fn

    @property
    def tool_name(self) -> str:
        return self._tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        return self._tool_spec

    @property
    def tool_type(self) -> str:
        return "function"

    async def stream(self, tool_use: ToolUse, _invocation_state: dict[str, Any],
                     **_kwargs: Any) -> AsyncGenerator[Any, None]:
        """
        Stream tool events and return the final result.

        Parameters
        ----------
            tool_use : ToolUse
                The tool use request containing tool ID and parameters.
            _invocation_state : dict[str, Any]
                Unused parameter for compatibility.
            _kwargs : Any
                Unused parameter for compatibility.

        Yields
        ------
            Tool events with the last being the tool result.
        """
        from strands.types._events import ToolResultEvent  # type: ignore
        from strands.types._events import ToolStreamEvent

        tool_use_id = tool_use.get("toolUseId", "unknown")
        tool_input = tool_use.get("input", {}) or {}

        try:
            if (self._fn.has_streaming_output and not self._fn.has_single_output):
                last_chunk: Any | None = None
                async for chunk in self._fn.acall_stream(**tool_input):
                    last_chunk = chunk
                    yield ToolStreamEvent(tool_use, chunk)
                final = _to_tool_result(tool_use_id, last_chunk if last_chunk is not None else "")
                yield ToolResultEvent(final)
                return

            result = await self._fn.acall_invoke(**tool_input)
            yield ToolResultEvent(_to_tool_result(tool_use_id, result))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Strands tool '%s' failed", self.tool_name)
            yield ToolResultEvent(_to_error_result(tool_use_id, exc))


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.STRANDS)
def strands_tool_wrapper(name: str, fn: Function, _builder: Builder) -> NATFunctionAgentTool:
    """Create a Strands `AgentTool` wrapper for a NAT `Function`."""
    if fn.input_schema is None:
        raise ValueError(f"Tool '{name}' must define an input schema")

    input_schema = _json_schema_from_pydantic(fn.input_schema)
    description = fn.description or name
    return NATFunctionAgentTool(name=name, description=description, input_schema=input_schema, fn=fn)
