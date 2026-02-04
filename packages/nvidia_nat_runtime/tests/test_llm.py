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
from types import SimpleNamespace
from typing import Any
from typing import cast

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionMessage
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChatCompletionChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from pydantic import SecretStr

from nat.data_models.common import OptionalSecretStr
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.retry_mixin import RetryMixin
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.runtime import llm as runtime_llm


class FakeRetryConfig(RetryMixin):
    """Retry config for `_apply_retry` verification."""


class FakeChatCompletions:
    """Mock for OpenAI chat completion endpoint."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self.response


class FakeResponses:
    """Mock for OpenAI responses endpoint."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self.response


class FakeAsyncOpenAI:
    """Fake OpenAI client with chat/responses routes."""

    def __init__(self, *, chat_response: object, response_response: object) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions(chat_response))
        self.responses = FakeResponses(response_response)


def _chat_completion(content: str | None) -> ChatCompletion:
    message = ChatCompletionMessage.model_construct(role="assistant", content=content)
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


def _chat_chunk(content: str | None) -> ChatCompletionChunk:
    delta = ChoiceDelta.model_construct(content=content)
    choice = ChatCompletionChunkChoice.model_construct(index=0, delta=delta, finish_reason=None, logprobs=None)
    return ChatCompletionChunk.model_construct(
        id="chunk-id",
        choices=[choice],
        created=0,
        model="model",
        object="chat.completion.chunk",
        service_tier=None,
        system_fingerprint=None,
        usage=None,
    )


def _response_text_event(text: str) -> ResponseTextDeltaEvent:
    return ResponseTextDeltaEvent.model_construct(
        content_index=0,
        delta=text,
        item_id="item-id",
        logprobs=[],
        output_index=0,
        sequence_number=0,
        type="response.output_text.delta",
        output_text=text,
    )


def _collect_event_types(payloads: list[IntermediateStepPayload]) -> list[IntermediateStepType]:
    return [payload.event_type for payload in payloads]


def test_build_default_params_excludes_internal_fields() -> None:
    """Ensure default params strip non-API fields."""
    api_key: OptionalSecretStr = SecretStr("secret")
    params = runtime_llm._build_default_params(
        OpenAIModelConfig(model_name="model-name", api_key=api_key, base_url="https://example.com", temperature=0.7))
    assert "model" not in params
    assert "type" not in params
    assert "api_key" not in params
    assert "base_url" not in params
    assert "stream_usage" not in params
    assert params["temperature"] == 0.7


def test_build_default_params_extra_exclude_removes_fields() -> None:
    """Verify extra exclude values are honored."""
    api_key: OptionalSecretStr = SecretStr("secret")
    params = runtime_llm._build_default_params(
        OpenAIModelConfig(model_name="model-name", api_key=api_key, base_url="https://example.com", temperature=0.7),
        extra_exclude={"temperature"},
    )
    assert "temperature" not in params


def test_ensure_stream_usage_injects_usage() -> None:
    """Stream params should always include usage metadata."""
    params: dict[str, object] = {"stream_options": {"foo": "bar"}}
    runtime_llm._ensure_stream_usage(params, stream=True)
    stream_options = cast(dict[str, object], params["stream_options"])
    assert stream_options["include_usage"] is True
    assert stream_options["foo"] == "bar"


def test_ensure_stream_usage_noop_when_not_streaming() -> None:
    """Non-streaming calls should not be modified."""
    params: dict[str, object] = {}
    runtime_llm._ensure_stream_usage(params, stream=False)
    assert params == {}


def test_extract_text_from_chunk_dict() -> None:
    """Dictionary chunks should extract content or output_text."""
    assert runtime_llm._extract_text_from_chunk({"choices": [{"delta": {"content": "hi"}}]}) == "hi"
    assert runtime_llm._extract_text_from_chunk({"output_text": "yo"}) == "yo"


def test_extract_text_from_chunk_object() -> None:
    """Object chunks should extract delta content."""
    chunk = _chat_chunk("hello")
    assert runtime_llm._extract_text_from_chunk(chunk) == "hello"


def test_apply_retry_wraps_retry_mixin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry mixin should route to retry patch helper."""
    sentinel = object()

    def _fake_patch(client: object, **_kwargs: object) -> object:
        return sentinel

    monkeypatch.setattr(runtime_llm, "patch_with_retry", _fake_patch)
    result = runtime_llm._apply_retry(client=object(), llm_config=FakeRetryConfig())
    assert result is sentinel


async def test_runtime_openai_chat_completions_non_stream(step_payloads) -> None:
    """Non-streaming chat calls emit start/end events."""
    response = _chat_completion("done")
    client = FakeAsyncOpenAI(chat_response=response, response_response=object())
    runtime_client = runtime_llm.RuntimeOpenAIClient(cast(AsyncOpenAI, client),
                                                     model_name="demo",
                                                     default_params={"temperature": 0.1})
    messages = cast(list[ChatCompletionMessageParam], [{"role": "user", "content": "hi"}])

    result = await runtime_client.chat_completions(messages)

    assert result is response
    event_types = _collect_event_types(step_payloads)
    assert event_types == [IntermediateStepType.LLM_START, IntermediateStepType.LLM_END]
    chat_calls = client.chat.completions.calls
    assert chat_calls[0]["model"] == "demo"
    assert chat_calls[0]["temperature"] == 0.1


async def test_runtime_openai_chat_completions_streaming(step_payloads) -> None:
    """Streaming chat calls emit token events and final output."""
    chunks = [_chat_chunk("hello"), _chat_chunk(" world")]

    async def _stream() -> AsyncGenerator[ChatCompletionChunk, None]:
        for chunk in chunks:
            yield chunk

    client = FakeAsyncOpenAI(chat_response=_stream(), response_response=object())
    runtime_client = runtime_llm.RuntimeOpenAIClient(cast(AsyncOpenAI, client), model_name="demo", default_params={})
    messages = cast(list[ChatCompletionMessageParam], [{"role": "user", "content": "hi"}])

    stream = cast(AsyncGenerator[ChatCompletionChunk, None],
                  await runtime_client.chat_completions(messages, stream=True))
    results = [chunk async for chunk in stream]

    assert results == chunks
    event_types = _collect_event_types(step_payloads)
    assert event_types == [
        IntermediateStepType.LLM_START,
        IntermediateStepType.LLM_NEW_TOKEN,
        IntermediateStepType.LLM_NEW_TOKEN,
        IntermediateStepType.LLM_END,
    ]
    assert step_payloads[-1].data.output == "hello world"


async def test_runtime_openai_responses_streaming(step_payloads) -> None:
    """Streaming responses emit token events and final output."""
    chunks = [
        _response_text_event("first"),
        _response_text_event(" second"),
    ]

    async def _stream() -> AsyncGenerator[object, None]:
        for chunk in chunks:
            yield chunk

    client = FakeAsyncOpenAI(chat_response=object(), response_response=_stream())
    runtime_client = runtime_llm.RuntimeOpenAIClient(cast(AsyncOpenAI, client), model_name="demo", default_params={})

    stream = cast(AsyncGenerator[ResponseTextDeltaEvent, None],
                  await runtime_client.responses([{
                      "role": "user", "content": "hi"
                  }], stream=True))
    results = [chunk async for chunk in stream]

    assert results == chunks
    event_types = _collect_event_types(step_payloads)
    assert event_types == [
        IntermediateStepType.LLM_START,
        IntermediateStepType.LLM_NEW_TOKEN,
        IntermediateStepType.LLM_NEW_TOKEN,
        IntermediateStepType.LLM_END,
    ]
    assert step_payloads[-1].data.output == "first second"


async def test_runtime_openai_responses_non_stream(step_payloads) -> None:
    """Non-streaming responses emit start/end events."""
    response = object()
    client = FakeAsyncOpenAI(chat_response=object(), response_response=response)
    runtime_client = runtime_llm.RuntimeOpenAIClient(cast(AsyncOpenAI, client), model_name="demo", default_params={})

    result = await runtime_client.responses([{"role": "user", "content": "hi"}])

    assert result is response
    event_types = _collect_event_types(step_payloads)
    assert event_types == [IntermediateStepType.LLM_START, IntermediateStepType.LLM_END]
