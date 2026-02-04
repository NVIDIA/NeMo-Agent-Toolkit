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
"""Runtime tool wrappers for OpenAI-compatible tool execution.

This module adapts toolkit functions into OpenAI tool schemas and emits
intermediate step events so NVIDIA NeMo Agent Toolkit workflows can trace tool
execution consistently.
"""

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TypeAlias
from uuid import uuid4

from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.cli.register_workflow import register_tool_wrapper
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData

logger = logging.getLogger(__name__)

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

ToolChunk: TypeAlias = object
ToolOutput: TypeAlias = ToolChunk | list[ToolChunk] | str | None
ToolStream: TypeAlias = AsyncGenerator[ToolChunk | str, None]


def _json_schema_from_pydantic(model: type[BaseModel]) -> JsonObject:
    """Convert a Pydantic model into an OpenAI-compatible JSON schema.

    Args:
        model: Pydantic model class describing tool inputs.

    Returns:
        JSON schema dictionary with OpenAI-specific defaults removed.
    """
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.pop("additionalProperties", None)
    return schema


@dataclass(slots=True)
class RuntimeToolWrapper:
    """OpenAI-compatible tool wrapper for a toolkit function.

    The wrapper emits intermediate step events for tool invocation, preserving
    end-to-end traces in NVIDIA NeMo Agent Toolkit runtime workflows.
    """

    name: str
    description: str
    parameters: JsonObject
    fn: Function

    @property
    def tool_schema(self) -> JsonObject:
        """Return the OpenAI tool schema for this function.

        Returns:
            Dictionary formatted for OpenAI tool registration.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def invoke(self, **kwargs: object) -> ToolOutput:
        """Invoke the wrapped function and emit lifecycle events.

        Args:
            kwargs: Keyword arguments forwarded to the wrapped function.

        Returns:
            Aggregated tool output. Streaming outputs are combined into a single
            text response when possible or a list of non-text chunks.
        """
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

        output: ToolOutput = None
        try:
            if self.fn.has_streaming_output and not self.fn.has_single_output:
                chunks: list[ToolChunk] = []
                text_parts: list[str] = []
                async for chunk in self.fn.acall_stream(**kwargs):
                    if isinstance(chunk, str):
                        text_parts.append(chunk)
                    else:
                        chunks.append(chunk)
                output = "".join(text_parts) if text_parts else (chunks if chunks else None)
            else:
                output = await self.fn.acall_invoke(**kwargs)
            return output
        finally:
            step_manager.push_intermediate_step(
                IntermediateStepPayload(
                    event_type=IntermediateStepType.TOOL_END,
                    framework=LLMFrameworkEnum.RUNTIME,
                    name=self.name,
                    UUID=step_id,
                    data=StreamEventData(input=kwargs, output=output),
                ))

    async def stream(self, **kwargs: object) -> ToolStream:
        """Stream results from the wrapped function while emitting events.

        Args:
            kwargs: Keyword arguments forwarded to the wrapped function.

        Yields:
            Streaming tool output chunks or a single non-streaming result.
        """
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

        output: ToolOutput = None
        try:
            if self.fn.has_streaming_output and not self.fn.has_single_output:
                chunks: list[ToolChunk] = []
                text_parts: list[str] = []
                async for chunk in self.fn.acall_stream(**kwargs):
                    if isinstance(chunk, str):
                        text_parts.append(chunk)
                    else:
                        chunks.append(chunk)
                    yield chunk
                output = "".join(text_parts) if text_parts else (chunks if chunks else None)
            else:
                output = await self.fn.acall_invoke(**kwargs)
                yield output
        finally:
            step_manager.push_intermediate_step(
                IntermediateStepPayload(
                    event_type=IntermediateStepType.TOOL_END,
                    framework=LLMFrameworkEnum.RUNTIME,
                    name=self.name,
                    UUID=step_id,
                    data=StreamEventData(input=kwargs, output=output),
                ))


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.RUNTIME)
def runtime_tool_wrapper(name: str, fn: Function, _builder: Builder) -> RuntimeToolWrapper:
    """Create a runtime tool wrapper for a toolkit function.

    Args:
        name: Tool name exposed to the OpenAI-compatible interface.
        fn: Toolkit `Function` to wrap.
        _builder: Builder instance (unused).

    Returns:
        Configured `RuntimeToolWrapper` instance.
    """
    return RuntimeToolWrapper(name=name,
                              description=fn.description or name,
                              parameters=_json_schema_from_pydantic(fn.input_schema),
                              fn=fn)
