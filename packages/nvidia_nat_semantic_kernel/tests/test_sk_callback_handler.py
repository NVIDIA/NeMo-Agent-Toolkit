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

import json
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from nat.utils.reactive.subject import Subject


async def test_semantic_kernel_handler_deduplicates_llm_patch_methods(reactive_stream: Subject,
                                                                      monkeypatch: pytest.MonkeyPatch):
    """Test that multiple OpenAI LLMs do not wrap the same SK methods repeatedly."""
    from nat.plugins.semantic_kernel.callback_handler import SemanticKernelProfilerHandler

    handler = SemanticKernelProfilerHandler(
        workflow_llms={
            "llm_a": SimpleNamespace(provider_type="openai"),
            "llm_b": SimpleNamespace(provider_type="openai"),
            "llm_other": SimpleNamespace(provider_type="anthropic"),
        })
    build_llm_call_patch = Mock(side_effect=lambda original: original)
    monkeypatch.setattr(handler, "_build_llm_call_patch", build_llm_call_patch)
    monkeypatch.setattr(handler, "_build_tool_call_patch", lambda original: original)

    handler.instrument()

    assert build_llm_call_patch.call_count == 2


@pytest.mark.slow
async def test_semantic_kernel_handler_tool_call(reactive_stream: Subject):
    """
    Test that the SK callback logs tool usage events.
    """
    from uuid import uuid4

    from nat.builder.context import Context
    from nat.builder.framework_enum import LLMFrameworkEnum
    from nat.data_models.intermediate_step import IntermediateStepPayload
    from nat.data_models.intermediate_step import IntermediateStepType
    from nat.data_models.intermediate_step import TraceMetadata
    from nat.plugins.semantic_kernel.callback_handler import SemanticKernelProfilerHandler

    all_ = []
    _ = SemanticKernelProfilerHandler(workflow_llms={})
    _ = reactive_stream.subscribe(all_.append)
    step_manager = Context.get().intermediate_step_manager
    # We'll manually simulate the relevant methods.

    # Suppose we do a tool "invoke_function_call"
    # We'll simulate a call to handler's patched function
    run_id1 = str(uuid4())
    start_event = IntermediateStepPayload(UUID=run_id1,
                                          event_type=IntermediateStepType.TOOL_START,
                                          framework=LLMFrameworkEnum.SEMANTIC_KERNEL,
                                          metadata=TraceMetadata(tool_inputs={"args": ["some input"]}))
    step_manager.push_intermediate_step(start_event)

    end_event = IntermediateStepPayload(UUID=run_id1,
                                        event_type=IntermediateStepType.TOOL_END,
                                        framework=LLMFrameworkEnum.SEMANTIC_KERNEL,
                                        metadata=TraceMetadata(tool_outputs={"result": "some result"}))
    step_manager.push_intermediate_step(end_event)

    assert len(all_) == 2
    assert all_[0].event_type == IntermediateStepType.TOOL_START
    assert all_[1].event_type == IntermediateStepType.TOOL_END
    assert all_[1].payload.metadata.tool_outputs == {"result": "some result"}


def _openai_chat_service(handler_body: Callable[[httpx.Request], httpx.Response]):
    """Build a real Semantic Kernel OpenAI chat service whose HTTP calls are answered by ``handler_body``."""
    from openai import AsyncOpenAI
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    client = AsyncOpenAI(api_key="test-key",
                         base_url="http://test/v1",
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler_body)))
    return OpenAIChatCompletion(ai_model_id="test-model", async_client=client)


def _streaming_response(request: httpx.Request) -> httpx.Response:
    """Answer a streaming chat completion with two content chunks and a final usage chunk."""
    chunks = [
        {
            "choices": [{
                "index": 0, "delta": {
                    "role": "assistant", "content": "Hello"
                }, "finish_reason": None
            }]
        },
        {
            "choices": [{
                "index": 0, "delta": {
                    "content": " world"
                }, "finish_reason": "stop"
            }]
        },
        {
            "choices": [], "usage": {
                "prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9
            }
        },
    ]
    header = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "test-model"}
    body = "".join(f"data: {json.dumps({**header, **chunk})}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


@pytest.fixture(name="restore_sk_patch_targets")
def restore_sk_patch_targets_fixture(monkeypatch: pytest.MonkeyPatch):
    """Undo the class-level patches that ``SemanticKernelProfilerHandler.instrument`` applies."""
    import semantic_kernel
    from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion_base import OpenAIChatCompletionBase

    for name in ("_inner_get_chat_message_contents", "_inner_get_streaming_chat_message_contents"):
        monkeypatch.setattr(OpenAIChatCompletionBase, name, getattr(OpenAIChatCompletionBase, name))
    monkeypatch.setattr(semantic_kernel.Kernel, "invoke_function_call", semantic_kernel.Kernel.invoke_function_call)


@pytest.mark.usefixtures("restore_sk_patch_targets")
async def test_semantic_kernel_handler_streaming_chat_completion(reactive_stream: Subject):
    """Streaming chat completions keep working and are recorded once the profiler patches Semantic Kernel."""
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
    from semantic_kernel.contents import ChatHistory

    from nat.data_models.intermediate_step import IntermediateStepType
    from nat.plugins.semantic_kernel.callback_handler import SemanticKernelProfilerHandler

    events = []
    _ = reactive_stream.subscribe(events.append)
    SemanticKernelProfilerHandler(workflow_llms={"llm": SimpleNamespace(provider_type="openai")}).instrument()

    service = _openai_chat_service(_streaming_response)
    chat_history = ChatHistory()
    chat_history.add_user_message("Say hello")

    text = ""
    async for messages in service.get_streaming_chat_message_contents(chat_history,
                                                                      OpenAIChatPromptExecutionSettings()):
        text += "".join(str(message) for message in messages)

    assert text == "Hello world"
    assert [event.event_type for event in events] == [IntermediateStepType.LLM_START, IntermediateStepType.LLM_END]
    assert events[0].payload.data.input == "Say hello"
    assert events[1].payload.data.output == "Hello world"
    token_usage = events[1].payload.usage_info.token_usage
    assert (token_usage.prompt_tokens, token_usage.completion_tokens, token_usage.total_tokens) == (7, 2, 9)


@pytest.mark.usefixtures("restore_sk_patch_targets")
async def test_semantic_kernel_handler_chat_completion_token_usage(reactive_stream: Subject):
    """Non-streaming chat completions record the provider's token usage, including the total."""
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
    from semantic_kernel.contents import ChatHistory

    from nat.data_models.intermediate_step import IntermediateStepType
    from nat.plugins.semantic_kernel.callback_handler import SemanticKernelProfilerHandler

    def chat_response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200,
                              json={
                                  "id": "c1",
                                  "object": "chat.completion",
                                  "created": 0,
                                  "model": "test-model",
                                  "choices": [{
                                      "index": 0,
                                      "message": {
                                          "role": "assistant", "content": "Hello world"
                                      },
                                      "finish_reason": "stop"
                                  }],
                                  "usage": {
                                      "prompt_tokens": 7,
                                      "completion_tokens": 2,
                                      "total_tokens": 9,
                                      "prompt_tokens_details": {
                                          "cached_tokens": 3
                                      }
                                  },
                              })

    events = []
    _ = reactive_stream.subscribe(events.append)
    SemanticKernelProfilerHandler(workflow_llms={"llm": SimpleNamespace(provider_type="openai")}).instrument()

    service = _openai_chat_service(chat_response)
    chat_history = ChatHistory()
    chat_history.add_user_message("Say hello")

    messages = await service.get_chat_message_contents(chat_history, OpenAIChatPromptExecutionSettings())

    assert str(messages[0]) == "Hello world"
    assert [event.event_type for event in events] == [IntermediateStepType.LLM_START, IntermediateStepType.LLM_END]
    assert events[1].payload.data.output == "Hello world"
    token_usage = events[1].payload.usage_info.token_usage
    assert (token_usage.prompt_tokens,
            token_usage.completion_tokens,
            token_usage.cached_tokens,
            token_usage.total_tokens) == (7, 2, 3, 9)


def _bad_request(request: httpx.Request) -> httpx.Response:
    """Answer every chat completion with an error the OpenAI client does not retry."""
    return httpx.Response(400, json={"error": {"message": "boom", "type": "invalid_request_error"}})


@pytest.mark.usefixtures("restore_sk_patch_targets")
@pytest.mark.parametrize("streaming", [False, True], ids=["non_streaming", "streaming"])
async def test_semantic_kernel_handler_failed_chat_completion_ends_llm_step(reactive_stream: Subject, streaming: bool):
    """A chat completion that raises still pushes LLM_END, and the error reaches the caller."""
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
    from semantic_kernel.contents import ChatHistory
    from semantic_kernel.exceptions import ServiceResponseException

    from nat.data_models.intermediate_step import IntermediateStepType
    from nat.plugins.semantic_kernel.callback_handler import SemanticKernelProfilerHandler

    events = []
    _ = reactive_stream.subscribe(events.append)
    SemanticKernelProfilerHandler(workflow_llms={"llm": SimpleNamespace(provider_type="openai")}).instrument()

    service = _openai_chat_service(_bad_request)
    chat_history = ChatHistory()
    chat_history.add_user_message("Say hello")
    settings = OpenAIChatPromptExecutionSettings()

    with pytest.raises(ServiceResponseException):
        if streaming:
            async for _ in service.get_streaming_chat_message_contents(chat_history, settings):
                pass
        else:
            await service.get_chat_message_contents(chat_history, settings)

    assert [event.event_type for event in events] == [IntermediateStepType.LLM_START, IntermediateStepType.LLM_END]
    assert events[1].payload.data.output is None


@pytest.mark.usefixtures("restore_sk_patch_targets")
async def test_semantic_kernel_handler_closed_stream_ends_llm_step(reactive_stream: Subject):
    """A stream the consumer closes early pushes LLM_END with the chunks received so far."""
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
    from semantic_kernel.contents import ChatHistory

    from nat.data_models.intermediate_step import IntermediateStepType
    from nat.plugins.semantic_kernel.callback_handler import SemanticKernelProfilerHandler

    events = []
    _ = reactive_stream.subscribe(events.append)
    SemanticKernelProfilerHandler(workflow_llms={"llm": SimpleNamespace(provider_type="openai")}).instrument()

    service = _openai_chat_service(_streaming_response)
    chat_history = ChatHistory()
    chat_history.add_user_message("Say hello")

    stream = service._inner_get_streaming_chat_message_contents(chat_history, OpenAIChatPromptExecutionSettings())
    first = await anext(stream)
    await stream.aclose()

    assert "".join(str(message) for message in first) == "Hello"
    assert [event.event_type for event in events] == [IntermediateStepType.LLM_START, IntermediateStepType.LLM_END]
    assert events[1].payload.data.output == "Hello"
