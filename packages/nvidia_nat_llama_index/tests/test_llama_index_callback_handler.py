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

from types import SimpleNamespace

from nat.utils.reactive.subject import Subject


def test_extract_token_usage_from_raw_openai_usage():
    """Test extraction from the provider usage object on a Chat Completions response."""
    from llama_index.core.llms import ChatMessage
    from llama_index.core.llms import ChatResponse

    from nat.plugins.llama_index.callback_handler import LlamaIndexProfilerHandler

    response = ChatResponse(
        message=ChatMessage.from_str("done"),
        raw=SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=18,
            completion_tokens=34,
            total_tokens=52,
            prompt_tokens_details=SimpleNamespace(cached_tokens=3),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        )),
    )

    usage = LlamaIndexProfilerHandler._extract_token_usage(response)

    assert usage.prompt_tokens == 18
    assert usage.completion_tokens == 34
    assert usage.total_tokens == 52
    assert usage.cached_tokens == 3
    assert usage.reasoning_tokens == 5


def test_extract_token_usage_falls_back_from_partial_raw_details():
    """Test detail counters can fall back independently from partial raw usage."""
    from llama_index.core.llms import ChatMessage
    from llama_index.core.llms import ChatResponse

    from nat.plugins.llama_index.callback_handler import LlamaIndexProfilerHandler

    response = ChatResponse(
        message=ChatMessage.from_str("done"),
        raw=SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=18,
            completion_tokens=34,
            total_tokens=52,
            prompt_tokens_details=SimpleNamespace(),
            completion_tokens_details=SimpleNamespace(),
        )),
        additional_kwargs={
            "usage":
                SimpleNamespace(
                    input_tokens_details=SimpleNamespace(cached_tokens=7),
                    output_tokens_details=SimpleNamespace(reasoning_tokens=9),
                )
        },
    )

    usage = LlamaIndexProfilerHandler._extract_token_usage(response)

    assert usage.prompt_tokens == 18
    assert usage.completion_tokens == 34
    assert usage.total_tokens == 52
    assert usage.cached_tokens == 7
    assert usage.reasoning_tokens == 9


def test_extract_token_usage_from_flattened_additional_kwargs():
    """Test fallback to LlamaIndex's flattened OpenAI-compatible usage fields."""
    from llama_index.core.llms import ChatMessage
    from llama_index.core.llms import ChatResponse

    from nat.plugins.llama_index.callback_handler import LlamaIndexProfilerHandler

    response = ChatResponse(
        message=ChatMessage.from_str("done"),
        raw=SimpleNamespace(usage=None),
        additional_kwargs={
            "prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18
        },
    )

    usage = LlamaIndexProfilerHandler._extract_token_usage(response)

    assert usage.prompt_tokens == 7
    assert usage.completion_tokens == 11
    assert usage.total_tokens == 18


def test_extract_token_usage_preserves_responses_style_usage():
    """Test compatibility with the existing nested Responses-style usage object."""
    from llama_index.core.llms import ChatMessage
    from llama_index.core.llms import ChatResponse

    from nat.plugins.llama_index.callback_handler import LlamaIndexProfilerHandler

    response = ChatResponse(
        message=ChatMessage.from_str("done"),
        raw=None,
        additional_kwargs={
            "usage":
                SimpleNamespace(
                    input_tokens=13,
                    output_tokens=21,
                    total_tokens=34,
                    input_tokens_details=SimpleNamespace(cached_tokens=2),
                    output_tokens_details=SimpleNamespace(reasoning_tokens=8),
                )
        },
    )

    usage = LlamaIndexProfilerHandler._extract_token_usage(response)

    assert usage.prompt_tokens == 13
    assert usage.completion_tokens == 21
    assert usage.total_tokens == 34
    assert usage.cached_tokens == 2
    assert usage.reasoning_tokens == 8


async def test_llama_index_handler_order(reactive_stream: Subject):
    """
    Test that the LlamaIndexProfilerHandler usage stats occur in correct order for LLM events.
    """
    from nat.data_models.intermediate_step import IntermediateStepType
    from nat.plugins.llama_index.callback_handler import LlamaIndexProfilerHandler
    handler = LlamaIndexProfilerHandler()
    stats_list = []
    _ = reactive_stream.subscribe(stats_list.append)

    # Simulate an LLM start event
    from llama_index.core.callbacks import CBEventType
    from llama_index.core.callbacks import EventPayload
    from llama_index.core.llms import ChatMessage
    from llama_index.core.llms import ChatResponse

    payload_start = {EventPayload.PROMPT: "Say something wise."}
    handler.on_event_start(event_type=CBEventType.LLM, payload=payload_start, event_id="evt-1")

    # Simulate an LLM end event
    payload_end = {
        EventPayload.RESPONSE:
            ChatResponse(message=ChatMessage.from_str("42 is the meaning of life."), raw="42 is the meaning of life.")
    }
    handler.on_event_end(event_type=CBEventType.LLM, payload=payload_end, event_id="evt-1")

    assert len(stats_list) == 2
    assert stats_list[0].event_type == IntermediateStepType.LLM_START
    assert stats_list[0].payload.data.input == "Say something wise."
    assert stats_list[1].payload.event_type == IntermediateStepType.LLM_END
    assert stats_list[0].payload.usage_info.num_llm_calls == 1
    # chat_responses is a bit short in this test, but we confirm at least we get something
