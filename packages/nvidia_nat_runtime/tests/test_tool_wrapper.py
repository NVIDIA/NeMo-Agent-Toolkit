# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from collections.abc import AsyncGenerator
from typing import cast

import pytest
from pydantic import BaseModel

from nat.builder.function import Function
from nat.data_models.function import EmptyFunctionConfig
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.plugins.runtime.tool_wrapper import RuntimeToolWrapper
from nat.sdk.tool.tool import Tool


class InputSchema(BaseModel):
    """Input schema for the fake function."""

    name: str


class RuntimeTestFunction(Function[InputSchema, object, object]):
    """Concrete function implementation for tool wrapper tests."""

    def __init__(self,
                 *,
                 description: str,
                 has_streaming_output: bool,
                 has_single_output: bool,
                 stream_chunks: list[object] | None = None,
                 invoke_result: object | None = None,
                 invoke_error: Exception | None = None) -> None:
        super().__init__(config=EmptyFunctionConfig(), description=description, input_schema=InputSchema)
        self._has_streaming_output = has_streaming_output
        self._has_single_output = has_single_output
        self._stream_chunks = stream_chunks or []
        self._invoke_result = invoke_result
        self._invoke_error = invoke_error

    @property
    def has_streaming_output(self) -> bool:
        return self._has_streaming_output

    @property
    def has_single_output(self) -> bool:
        return self._has_single_output

    async def _ainvoke(self, value: InputSchema) -> object:
        if self._invoke_error:
            raise self._invoke_error
        return self._invoke_result

    async def _astream(self, value: InputSchema) -> AsyncGenerator[object, None]:
        for chunk in self._stream_chunks:
            yield chunk


def _event_types(payloads: list[IntermediateStepPayload]) -> list[IntermediateStepType]:
    return [payload.event_type for payload in payloads]


def _make_tool(fn: Function, name: str = "tool") -> Tool:
    """Create a Tool from a Function, mirroring the runtime_tool_wrapper factory."""
    return Tool.from_function(fn)


def test_backward_compat_alias() -> None:
    """RuntimeToolWrapper should be an alias for Tool."""
    assert RuntimeToolWrapper is Tool


def test_tool_schema() -> None:
    """Tool schema should include name, description, and parameters."""
    fn = RuntimeTestFunction(description="desc", has_streaming_output=False, has_single_output=True, invoke_result="ok")
    tool = _make_tool(fn)
    schema = tool.tool_schema
    function_schema = cast(dict[str, object], schema["function"])
    assert function_schema["description"] == "desc"
    assert "parameters" in function_schema


async def test_tool_invoke_streaming(step_payloads) -> None:
    """Streaming invoke aggregates text outputs."""
    fn = RuntimeTestFunction(description="desc",
                             has_streaming_output=True,
                             has_single_output=False,
                             stream_chunks=["hi", " there"])
    tool = _make_tool(fn)
    result = await tool.invoke(name="alice")

    assert result.output == "hi there"
    assert _event_types(step_payloads) == [
        IntermediateStepType.TOOL_START,
        IntermediateStepType.FUNCTION_START,
        IntermediateStepType.FUNCTION_END,
        IntermediateStepType.TOOL_END,
    ]
    assert step_payloads[-1].data.output == "hi there"


async def test_tool_invoke_non_streaming(step_payloads) -> None:
    """Non-streaming invoke forwards to acall_invoke."""
    fn = RuntimeTestFunction(description="desc",
                             has_streaming_output=False,
                             has_single_output=True,
                             invoke_result={"ok": True})
    tool = _make_tool(fn)

    result = await tool.invoke(name="bob")

    assert result.output == {"ok": True}
    assert _event_types(step_payloads) == [
        IntermediateStepType.TOOL_START,
        IntermediateStepType.FUNCTION_START,
        IntermediateStepType.FUNCTION_END,
        IntermediateStepType.TOOL_END,
    ]
    assert step_payloads[-1].data.output == {"ok": True}


async def test_tool_invoke_error_emits_end(step_payloads) -> None:
    """Errors should still emit TOOL_END with a None output."""
    fn = RuntimeTestFunction(description="desc",
                             has_streaming_output=False,
                             has_single_output=True,
                             invoke_error=RuntimeError("boom"))
    tool = _make_tool(fn)

    with pytest.raises(RuntimeError, match="boom"):
        await tool.invoke(name="bob")

    assert _event_types(step_payloads) == [
        IntermediateStepType.TOOL_START,
        IntermediateStepType.FUNCTION_START,
        IntermediateStepType.FUNCTION_END,
        IntermediateStepType.TOOL_END,
    ]
    assert step_payloads[-1].data.output is None


async def test_tool_stream_streaming(step_payloads) -> None:
    """Streaming API yields chunks and aggregates output."""
    fn = RuntimeTestFunction(description="desc",
                             has_streaming_output=True,
                             has_single_output=False,
                             stream_chunks=["a", "b"])
    tool = _make_tool(fn)

    results = [chunk async for chunk in tool.stream(name="cara")]

    assert results == ["a", "b"]
    assert _event_types(step_payloads) == [
        IntermediateStepType.TOOL_START,
        IntermediateStepType.FUNCTION_START,
        IntermediateStepType.FUNCTION_END,
        IntermediateStepType.TOOL_END,
    ]
    assert step_payloads[-1].data.output == "ab"


async def test_tool_stream_non_streaming(step_payloads) -> None:
    """Stream API should yield a single result for non-streaming tools."""
    fn = RuntimeTestFunction(description="desc", has_streaming_output=False, has_single_output=True, invoke_result="ok")
    tool = _make_tool(fn)

    results = [chunk async for chunk in tool.stream(name="dana")]

    assert results == ["ok"]
    assert _event_types(step_payloads) == [
        IntermediateStepType.TOOL_START,
        IntermediateStepType.FUNCTION_START,
        IntermediateStepType.FUNCTION_END,
        IntermediateStepType.TOOL_END,
    ]
    assert step_payloads[-1].data.output == "ok"
