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

import inspect
import itertools
import re
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import get_type_hints
from uuid import uuid4

from nat.sdk.tool.result import ToolResult

if TYPE_CHECKING:
    from nat.builder.function import Function
    from nat.builder.function import FunctionGroup
    from nat.data_models.function import FunctionBaseConfig
    from nat.data_models.function import FunctionGroupBaseConfig

_tool_id_counter = itertools.count()

ExecuteFn = Callable[..., ToolResult | Awaitable[ToolResult] | Any | Awaitable[Any]]


@dataclass(frozen=True)
class Tool:
    """An immutable, schema-bearing tool definition.

    ``parameters`` is a JSON Schema dict describing the tool's input.
    ``execute`` is a callable that performs the tool's action.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    execute: ExecuteFn = field(default=lambda **_kw: ToolResult(), repr=False)
    _fn: Any = field(default=None, repr=False, compare=False, hash=False)

    @staticmethod
    def from_function(fn: "Function") -> "Tool":
        """Wrap a core Function as an SDK Tool."""
        schema = fn.input_schema.model_json_schema()

        async def _execute(**kwargs: Any) -> ToolResult:
            result = await fn.acall_invoke(**kwargs)
            return ToolResult(output=result)

        return Tool(
            name=fn.display_name,
            description=fn.description or "",
            parameters=schema,
            execute=_execute,
            _fn=fn,
        )

    @staticmethod
    async def from_function_config(config: "FunctionBaseConfig") -> "Tool":
        """Build a Function from config via the type registry and Builder, then wrap as Tool."""
        from nat.builder.builder import Builder

        builder = Builder.current()
        name = config.name or f"sdk_fn_{next(_tool_id_counter)}"
        fn = await builder.add_function(name, config)
        return Tool.from_function(fn)

    @staticmethod
    async def from_function_group(group: "FunctionGroup") -> list["Tool"]:
        """Wrap each accessible function in a pre-built FunctionGroup as an SDK Tool."""
        functions = await group.get_accessible_functions()
        return [Tool.from_function(fn) for fn in functions.values()]

    @staticmethod
    async def from_function_group_config(config: "FunctionGroupBaseConfig") -> list["Tool"]:
        """Build a FunctionGroup from config via the type registry, then wrap each function as a Tool."""
        from nat.builder.builder import Builder

        builder = Builder.current()
        name = f"sdk_fg_{next(_tool_id_counter)}"
        group = await builder.add_function_group(name, config)
        functions = await group.get_accessible_functions()
        return [Tool.from_function(fn) for fn in functions.values()]

    def to_openai_schema(self) -> dict[str, Any]:
        """Serialize to the OpenAI function-calling tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @property
    def tool_schema(self) -> dict[str, Any]:
        """Return the OpenAI tool schema (alias for to_openai_schema)."""
        return self.to_openai_schema()

    async def invoke(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with tracing, normalising sync/async and wrapping raw returns."""
        from nat.builder.context import Context
        from nat.builder.framework_enum import LLMFrameworkEnum
        from nat.data_models.intermediate_step import IntermediateStepPayload
        from nat.data_models.intermediate_step import IntermediateStepType
        from nat.data_models.intermediate_step import StreamEventData

        step_id = str(uuid4())
        step_manager = Context.get().intermediate_step_manager
        step_manager.push_intermediate_step(
            IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_START,
                framework=LLMFrameworkEnum.RUNTIME,
                name=self.name,
                UUID=step_id,
                data=StreamEventData(input=kwargs),
            ))

        output = None
        result: ToolResult
        try:
            if self._fn is not None and self._fn.has_streaming_output and not self._fn.has_single_output:
                chunks: list[Any] = []
                text_parts: list[str] = []
                async for chunk in self._fn.acall_stream(**kwargs):
                    if isinstance(chunk, str):
                        text_parts.append(chunk)
                    else:
                        chunks.append(chunk)
                output = "".join(text_parts) if text_parts else (chunks if chunks else None)
                result = ToolResult(output=output)
            else:
                raw = self.execute(**kwargs)
                if inspect.isawaitable(raw):
                    raw = await raw
                if not isinstance(raw, ToolResult):
                    raw = ToolResult(output=raw)
                result = raw
                output = result.output
            return result
        finally:
            step_manager.push_intermediate_step(
                IntermediateStepPayload(
                    event_type=IntermediateStepType.TOOL_END,
                    framework=LLMFrameworkEnum.RUNTIME,
                    name=self.name,
                    UUID=step_id,
                    data=StreamEventData(input=kwargs, output=output),
                ))

    async def stream(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Stream results from the tool with tracing. Falls back to invoke for non-streaming tools."""
        from nat.builder.context import Context
        from nat.builder.framework_enum import LLMFrameworkEnum
        from nat.data_models.intermediate_step import IntermediateStepPayload
        from nat.data_models.intermediate_step import IntermediateStepType
        from nat.data_models.intermediate_step import StreamEventData

        step_id = str(uuid4())
        step_manager = Context.get().intermediate_step_manager
        step_manager.push_intermediate_step(
            IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_START,
                framework=LLMFrameworkEnum.RUNTIME,
                name=self.name,
                UUID=step_id,
                data=StreamEventData(input=kwargs),
            ))

        output = None
        try:
            if self._fn is not None and self._fn.has_streaming_output and not self._fn.has_single_output:
                chunks: list[Any] = []
                text_parts: list[str] = []
                async for chunk in self._fn.acall_stream(**kwargs):
                    if isinstance(chunk, str):
                        text_parts.append(chunk)
                    else:
                        chunks.append(chunk)
                    yield chunk
                output = "".join(text_parts) if text_parts else (chunks if chunks else None)
            else:
                result = await self(**kwargs)
                output = result.output
                yield result.output
        finally:
            step_manager.push_intermediate_step(
                IntermediateStepPayload(
                    event_type=IntermediateStepType.TOOL_END,
                    framework=LLMFrameworkEnum.RUNTIME,
                    name=self.name,
                    UUID=step_id,
                    data=StreamEventData(input=kwargs, output=output),
                ))

    async def __call__(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, normalising sync/async callables and wrapping raw returns."""
        try:
            result = self.execute(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, ToolResult):
                result = ToolResult(output=result)
            return result
        except Exception as exc:
            return ToolResult(error=str(exc))


# ---------------------------------------------------------------------------
# Convenience decorator
# ---------------------------------------------------------------------------

_PYTHON_TYPE_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _camel_to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _build_schema_from_function(fn: Callable[..., Any]) -> dict[str, Any]:
    """Derive a JSON Schema ``parameters`` object from a function's type hints."""
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param_name == "return":
            continue

        prop: dict[str, Any] = {}
        hint = hints.get(param_name)
        if hint is not None:
            json_type = _PYTHON_TYPE_TO_JSON.get(hint)
            if json_type:
                prop["type"] = json_type

        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def tool(
    description: str,
    *,
    name: str | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator that turns a function into a :class:`Tool`.

    Usage::

        @tool(description="Search the web")
        async def web_search(query: str) -> str:
            ...
    """

    def decorator(fn: Callable[..., Any]) -> Tool:
        tool_name = name or _camel_to_snake(fn.__name__)
        parameters = _build_schema_from_function(fn)

        async def _execute(**kwargs: Any) -> ToolResult:
            result = fn(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, ToolResult):
                return result
            return ToolResult(output=result)

        return Tool(
            name=tool_name,
            description=description,
            parameters=parameters,
            execute=_execute,
        )

    return decorator
