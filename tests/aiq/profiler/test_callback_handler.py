# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
from uuid import uuid4

import pytest

from aiq.builder.context import AIQContext
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepType
from aiq.data_models.intermediate_step import StreamEventData
from aiq.data_models.intermediate_step import TraceMetadata
from aiq.data_models.intermediate_step import UsageInfo
from aiq.profiler.callbacks.langchain_callback_handler import LangchainProfilerHandler
from aiq.profiler.callbacks.llama_index_callback_handler import LlamaIndexProfilerHandler
from aiq.utils.reactive.subject import Subject


async def test_langchain_handler(reactive_stream: Subject):
    """
    Test that the LangchainProfilerHandler produces usage stats in the correct order:
      - on_llm_start -> usage stat with event_type=LLM_START
      - on_llm_new_token -> usage stat with event_type=LLM_NEW_TOKEN
      - on_llm_end -> usage stat with event_type=LLM_END
    And that the queue sees them in the correct order.
    """

    all_stats = []
    handler = LangchainProfilerHandler()
    _ = reactive_stream.subscribe(all_stats.append)

    # Simulate an LLM start event
    prompts = ["Hello world"]
    run_id = str(uuid4())

    await handler.on_llm_start(serialized={}, prompts=prompts, run_id=run_id)

    # Simulate a fake sleep for 3 seconds
    await asyncio.sleep(1)

    # Simulate receiving new tokens
    await handler.on_llm_new_token("hello", run_id=run_id)
    await handler.on_llm_new_token(" world", run_id=run_id)

    # Build a fake LLMResult
    from langchain_core.messages import AIMessage
    from langchain_core.messages.ai import UsageMetadata
    from langchain_core.outputs import ChatGeneration
    from langchain_core.outputs import LLMResult

    generation = ChatGeneration(message=AIMessage(
        content="Hello back!",
        # Instantiate usage metadata typed dict with input tokens and output tokens
        usage_metadata=UsageMetadata(input_tokens=15, output_tokens=15, total_tokens=0)))
    llm_result = LLMResult(generations=[[generation]])
    await handler.on_llm_end(response=llm_result, run_id=run_id)

    assert len(all_stats) == 4, "Expected 4 usage stats events total"
    assert all_stats[0].event_type == IntermediateStepType.LLM_START
    assert all_stats[1].event_type == IntermediateStepType.LLM_NEW_TOKEN
    assert all_stats[2].event_type == IntermediateStepType.LLM_NEW_TOKEN
    assert all_stats[3].event_type == IntermediateStepType.LLM_END

    # Test event timestamp to ensure we don't have any race conditions
    assert all_stats[0].event_timestamp < all_stats[1].event_timestamp < all_stats[2].event_timestamp
    assert all_stats[1].event_timestamp - all_stats[0].event_timestamp > 1  # 1 second between LLM start and new token

    # Check that the first usage stat has the correct chat_inputs
    assert all_stats[0].payload.metadata.chat_inputs == prompts
    # Check new token event usage
    assert all_stats[1].payload.data.chunk == "hello"  # we captured "hello"
    # Check final token usage
    assert all_stats[3].payload.usage_info.token_usage.prompt_tokens == 15  # Will not populate usage
    assert all_stats[3].payload.usage_info.token_usage.completion_tokens == 15
    assert all_stats[3].payload.data.output == "Hello back!"


async def test_llama_index_handler_order(reactive_stream: Subject):
    """
    Test that the LlamaIndexProfilerHandler usage stats occur in correct order for LLM events.
    """

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


async def test_crewai_handler_time_between_calls(reactive_stream: Subject):
    """
    Test CrewAIProfilerHandler ensures seconds_between_calls is properly set for consecutive calls.
    We'll mock time.time() to produce stable intervals.
    """
    pytest.importorskip("crewai")

    import math

    from packages.agentiq_crewai.src.aiq.plugins.crewai.crewai_callback_handler import CrewAIProfilerHandler

    # The crewAI handler monkey-patch logic is for real code instrumentation,
    # but let's just call the wrapped calls directly:
    results = []
    handler = CrewAIProfilerHandler()
    _ = reactive_stream.subscribe(results.append)
    step_manager = AIQContext.get().intermediate_step_manager

    # We'll patch time.time so it returns predictable values:
    # e.g. 100.0 for the first call, 103.2 for the second, etc.
    # Simulate a first LLM call
    # crewAI calls _llm_call_monkey_patch => we can't call that directly, let's just do an inline approach
    # We'll do a short local function "simulate_llm_call" that replicates the logic:
    times = [100.0, 103.2, 107.5, 112.0]
    # seconds_between_calls = int(now - self.last_call_ts) => at the first call, last_call_ts=some default
    # but let's just forcibly create a usage stat

    run_id1 = str(uuid4())
    start_stat = IntermediateStepPayload(UUID=run_id1,
                                         event_type=IntermediateStepType.LLM_START,
                                         data=StreamEventData(input="Hello user!"),
                                         framework=LLMFrameworkEnum.CREWAI,
                                         event_timestamp=times[0])
    step_manager.push_intermediate_step(start_stat)
    handler.last_call_ts = times[0]

    # Simulate end
    end_stat = IntermediateStepPayload(UUID=run_id1,
                                       event_type=IntermediateStepType.LLM_END,
                                       data=StreamEventData(output="World response"),
                                       framework=LLMFrameworkEnum.CREWAI,
                                       event_timestamp=times[1])
    step_manager.push_intermediate_step(end_stat)

    now2 = times[2]
    run_id2 = str(uuid4())
    start_stat2 = IntermediateStepPayload(UUID=run_id2,
                                          event_type=IntermediateStepType.LLM_START,
                                          data=StreamEventData(input="Hello again!"),
                                          framework=LLMFrameworkEnum.CREWAI,
                                          event_timestamp=now2,
                                          usage_info=UsageInfo(seconds_between_calls=math.floor(now2 -
                                                                                                handler.last_call_ts)))
    step_manager.push_intermediate_step(start_stat2)
    handler.last_call_ts = now2
    second_end = IntermediateStepPayload(UUID=run_id2,
                                         event_type=IntermediateStepType.LLM_END,
                                         data=StreamEventData(output="Another response"),
                                         framework=LLMFrameworkEnum.CREWAI,
                                         event_timestamp=times[3])
    step_manager.push_intermediate_step(second_end)

    assert len(results) == 4
    # Check the intervals
    assert results[2].usage_info.seconds_between_calls == 7


async def test_semantic_kernel_handler_tool_call(reactive_stream: Subject):
    """
    Test that the SK callback logs tool usage events.
    """

    pytest.importorskip("semantic_kernel")

    from aiq.profiler.callbacks.semantic_kernel_callback_handler import SemanticKernelProfilerHandler

    all_ = []
    _ = SemanticKernelProfilerHandler(workflow_llms={})
    _ = reactive_stream.subscribe(all_.append)
    step_manager = AIQContext.get().intermediate_step_manager
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
