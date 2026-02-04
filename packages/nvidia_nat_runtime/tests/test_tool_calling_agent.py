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
from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from typing import cast

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message_function_tool_call import Function as ToolFunction
from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.builder.function_info import FunctionInfo
from nat.data_models.api_server import ChatRequest
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import EmptyFunctionConfig
from nat.plugins.runtime import tool_calling_agent as runtime_agent
from nat.plugins.runtime.llm import RuntimeOpenAIClient
from nat.plugins.runtime.tool_wrapper import RuntimeToolWrapper


class InputSchema(BaseModel):
    """Input schema for runtime tool-calling agent tests."""

    q: str


class RuntimeTestFunction(Function[InputSchema, object, object]):
    """Concrete function implementation for agent tests."""

    def __init__(self, *, description: str, output: object | None = None, error: Exception | None = None) -> None:
        super().__init__(config=EmptyFunctionConfig(), description=description, input_schema=InputSchema)
        self._output = output
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def _ainvoke(self, value: InputSchema) -> object | None:
        self.calls.append(value.model_dump())
        if self._error:
            raise self._error
        return self._output

    async def _astream(self, value: InputSchema) -> AsyncGenerator[object, None]:
        yield self._output

    @property
    def has_streaming_output(self) -> bool:
        return False

    @property
    def has_single_output(self) -> bool:
        return True


class FakeChatCompletions:
    """Mock for OpenAI chat completion endpoint."""

    def __init__(self, responses: list[ChatCompletion]) -> None:
        self._responses = list(responses)

    async def create(self, **_kwargs: Any) -> ChatCompletion:
        return self._responses.pop(0)


class FakeAsyncOpenAI:
    """Fake OpenAI client with chat completions route."""

    def __init__(self, responses: list[ChatCompletion]) -> None:
        self.chat = type("Chat", (), {"completions": FakeChatCompletions(responses)})()
        self.responses = type("Responses", (), {"create": lambda **_kwargs: None})()


@dataclass
class FakeBuilder:
    """Minimal builder to supply LLMs and tools."""

    llm: RuntimeOpenAIClient
    tools: dict[str, RuntimeToolWrapper]

    async def get_llm(self, _name: str, wrapper_type: LLMFrameworkEnum) -> RuntimeOpenAIClient:
        return self.llm

    async def get_tools(self, tool_names: list[FunctionRef],
                        wrapper_type: LLMFrameworkEnum) -> list[RuntimeToolWrapper]:
        return [self.tools[str(name)] for name in tool_names]


def _tool_call(tool_id: str, name: str, arguments: str) -> ChatCompletionMessageFunctionToolCall:
    function = ToolFunction.model_construct(name=name, arguments=arguments)
    return ChatCompletionMessageFunctionToolCall.model_construct(id=tool_id, function=function, type="function")


def _chat_completion(content: str | None,
                     tool_calls: list[ChatCompletionMessageFunctionToolCall] | None) -> ChatCompletion:
    message = ChatCompletionMessage.model_construct(role="assistant", content=content, tool_calls=tool_calls)
    choice = ChatCompletionChoice.model_construct(index=0, message=message, finish_reason="stop", logprobs=None)
    return ChatCompletion.model_construct(
        id="completion-id",
        choices=[choice],
        created=0,
        model="model",
        object="chat.completion",
        service_tier=None,
        system_fingerprint=None,
        usage=None,
    )


async def _build_agent(config: runtime_agent.RuntimeToolCallingAgentConfig,
                       llm: RuntimeOpenAIClient,
                       tools: dict[str, RuntimeToolWrapper]) -> FunctionInfo:
    builder = cast(Builder, FakeBuilder(llm=llm, tools=tools))
    async with runtime_agent.runtime_tool_calling_agent_workflow(config, builder) as function_info:
        return cast(FunctionInfo, function_info)


async def _call_agent(function_info: FunctionInfo, request: ChatRequest) -> str:
    response_fn = cast(Callable[[ChatRequest], Any], function_info.single_fn)
    return await response_fn(request)


async def test_tool_calling_agent_returns_final_response() -> None:
    """Agent returns the final model response after tool use."""
    tool_call = _tool_call("call-1", "tool", '{"q": "x"}')
    responses = [
        _chat_completion(None, [tool_call]),
        _chat_completion("done", None),
    ]
    llm = RuntimeOpenAIClient(cast(AsyncOpenAI, FakeAsyncOpenAI(responses)), model_name="demo", default_params={})
    function = RuntimeTestFunction(description="desc", output={"ok": True})
    tool = RuntimeToolWrapper(name="tool", description="desc", parameters={}, fn=function)
    config = runtime_agent.RuntimeToolCallingAgentConfig(llm_name=LLMRef("demo"),
                                                         tool_names=[FunctionRef("tool")],
                                                         max_iterations=2,
                                                         max_history=5)
    function_info = await _build_agent(config, llm, {"tool": tool})

    result = await _call_agent(function_info, ChatRequest.from_string("hi"))

    assert result == "done"
    assert function.calls == [{"q": "x"}]


async def test_tool_calling_agent_return_direct() -> None:
    """Return-direct tools short-circuit to tool output."""
    tool_call = _tool_call("call-1", "tool", '{"q": "x"}')
    responses = [_chat_completion(None, [tool_call])]
    llm = RuntimeOpenAIClient(cast(AsyncOpenAI, FakeAsyncOpenAI(responses)), model_name="demo", default_params={})
    function = RuntimeTestFunction(description="desc", output={"ok": True})
    tool = RuntimeToolWrapper(name="tool", description="desc", parameters={}, fn=function)
    config = runtime_agent.RuntimeToolCallingAgentConfig(llm_name=LLMRef("demo"),
                                                         tool_names=[FunctionRef("tool")],
                                                         return_direct=[FunctionRef("tool")],
                                                         max_iterations=1)
    function_info = await _build_agent(config, llm, {"tool": tool})

    result = await _call_agent(function_info, ChatRequest.from_string("hi"))

    assert result == '{"ok": true}'


async def test_tool_calling_agent_handles_tool_error_when_enabled() -> None:
    """Tool errors are surfaced to the LLM when enabled."""
    tool_call = _tool_call("call-1", "tool", '{"q": "x"}')
    responses = [
        _chat_completion(None, [tool_call]),
        _chat_completion("done", None),
    ]
    llm = RuntimeOpenAIClient(cast(AsyncOpenAI, FakeAsyncOpenAI(responses)), model_name="demo", default_params={})
    function = RuntimeTestFunction(description="desc", error=RuntimeError("boom"))
    tool = RuntimeToolWrapper(name="tool", description="desc", parameters={}, fn=function)
    config = runtime_agent.RuntimeToolCallingAgentConfig(llm_name=LLMRef("demo"),
                                                         tool_names=[FunctionRef("tool")],
                                                         handle_tool_errors=True,
                                                         max_iterations=2)
    function_info = await _build_agent(config, llm, {"tool": tool})

    result = await _call_agent(function_info, ChatRequest.from_string("hi"))

    assert result == "done"


async def test_tool_calling_agent_raises_tool_error_when_disabled() -> None:
    """Tool errors raise when error handling is disabled."""
    tool_call = _tool_call("call-1", "tool", '{"q": "x"}')
    responses = [_chat_completion(None, [tool_call])]
    llm = RuntimeOpenAIClient(cast(AsyncOpenAI, FakeAsyncOpenAI(responses)), model_name="demo", default_params={})
    function = RuntimeTestFunction(description="desc", error=RuntimeError("boom"))
    tool = RuntimeToolWrapper(name="tool", description="desc", parameters={}, fn=function)
    config = runtime_agent.RuntimeToolCallingAgentConfig(llm_name=LLMRef("demo"),
                                                         tool_names=[FunctionRef("tool")],
                                                         handle_tool_errors=False,
                                                         max_iterations=1)
    function_info = await _build_agent(config, llm, {"tool": tool})

    with pytest.raises(RuntimeError, match="boom"):
        await _call_agent(function_info, ChatRequest.from_string("hi"))


async def test_tool_calling_agent_max_iterations() -> None:
    """Agent raises when exceeding max iterations."""
    tool_call = _tool_call("call-1", "tool", '{"q": "x"}')
    responses = [_chat_completion(None, [tool_call])]
    llm = RuntimeOpenAIClient(cast(AsyncOpenAI, FakeAsyncOpenAI(responses)), model_name="demo", default_params={})
    function = RuntimeTestFunction(description="desc", output={"ok": True})
    tool = RuntimeToolWrapper(name="tool", description="desc", parameters={}, fn=function)
    config = runtime_agent.RuntimeToolCallingAgentConfig(llm_name=LLMRef("demo"),
                                                         tool_names=[FunctionRef("tool")],
                                                         max_iterations=0)
    function_info = await _build_agent(config, llm, {"tool": tool})

    with pytest.raises(RuntimeError, match="max_iterations"):
        await _call_agent(function_info, ChatRequest.from_string("hi"))
