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

import asyncio
import logging
from uuid import uuid4

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import Generation
from langchain_core.outputs import LLMResult

from nat.builder.context import Context
from nat.data_models.intermediate_step import IntermediateStepType
from nat.plugins.langchain.callback_handler import LangchainProfilerHandler
from nat.plugins.langchain.callback_handler import _extract_tools_schema
from nat.utils.reactive.subject import Subject


@pytest.fixture(name="chain_handler_fixture")
def fixture_chain_handler(reactive_stream: Subject):
    all_stats = []
    handler = LangchainProfilerHandler()
    _ = reactive_stream.subscribe(all_stats.append)
    return handler, all_stats


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

    # Simulate a fake sleep for 0.05 second
    await asyncio.sleep(0.05)

    # Simulate receiving new tokens with delay between them
    await handler.on_llm_new_token("hello", run_id=run_id)
    await asyncio.sleep(0.05)  # Ensure a small delay between token events
    await handler.on_llm_new_token(" world", run_id=run_id)

    # Simulate a delay before ending
    await asyncio.sleep(0.05)

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
    # Use >= instead of < to handle cases where timestamps might be identical or very close
    assert all_stats[0].event_timestamp <= all_stats[1].event_timestamp
    assert all_stats[1].event_timestamp <= all_stats[2].event_timestamp
    assert all_stats[2].event_timestamp <= all_stats[3].event_timestamp

    # Check that there's a delay between start and first token
    assert all_stats[1].event_timestamp - all_stats[0].event_timestamp > 0.05

    # Check that the first usage stat has the correct chat_inputs
    assert all_stats[0].payload.metadata.chat_inputs == prompts
    # Check new token event usage
    assert all_stats[1].payload.data.chunk == "hello"  # we captured "hello"
    # Check final token usage
    assert all_stats[3].payload.usage_info.token_usage.prompt_tokens == 15  # Will not populate usage
    assert all_stats[3].payload.usage_info.token_usage.completion_tokens == 15
    assert all_stats[3].payload.data.output == "Hello back!"


async def test_langchain_handler_traces_native_completion_model(chain_handler_fixture):
    """A NeMo-compatible BaseLLM emits plain Generation objects, not chat messages."""
    handler, all_stats = chain_handler_fixture
    context = Context.get()
    initial_span = context.active_span_id
    model = FakeListLLM(responses=["allowed"], callbacks=[handler])

    output = await model.ainvoke("Check this input", config={"metadata": {"ls_model_name": "rail-model"}})

    assert output == "allowed"
    assert [step.event_type for step in all_stats] == [IntermediateStepType.LLM_START, IntermediateStepType.LLM_END]
    assert all_stats[0].UUID == all_stats[1].UUID
    assert all_stats[1].payload.name == "rail-model"
    assert all_stats[1].payload.data.input == "Check this input"
    assert all_stats[1].payload.data.output == "allowed"
    assert context.active_span_id == initial_span
    assert not handler._run_id_to_model_name
    assert not handler._run_id_to_llm_input
    assert not handler._run_id_to_parent_span
    assert not handler._run_id_to_start_time


@pytest.mark.parametrize("chat_model", [False, True])
async def test_langchain_handler_reads_provider_token_usage(chain_handler_fixture, chat_model):
    """Provider-level token counts are retained when a message has no usage_metadata."""
    handler, all_stats = chain_handler_fixture
    run_id = uuid4()
    await handler.on_llm_start(serialized={"kwargs": {
        "model_name": "native-rail"
    }},
                               prompts=["policy prompt"],
                               run_id=run_id)
    generation = ChatGeneration(message=AIMessage(content="yes")) if chat_model else Generation(text="yes")
    response = LLMResult(generations=[[generation]],
                         llm_output={
                             "token_usage": {
                                 "prompt_tokens": 11,
                                 "completion_tokens": 4,
                                 "total_tokens": 15,
                                 "prompt_tokens_details": {
                                     "cached_tokens": 3
                                 },
                                 "completion_tokens_details": {
                                     "reasoning_tokens": 2
                                 },
                             }
                         })
    await handler.on_llm_end(response, run_id=run_id)

    end = all_stats[-1].payload
    assert end.name == "native-rail"
    assert end.data.output == "yes"
    assert end.usage_info.token_usage.model_dump() == {
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
        "cached_tokens": 3,
        "reasoning_tokens": 2,
    }


@pytest.mark.parametrize(
    "details",
    [
        {
            "prompt_tokens_details": None, "completion_tokens_details": None
        },
        {
            "prompt_tokens_details": {
                "cached_tokens": None
            }, "completion_tokens_details": {
                "reasoning_tokens": None
            }
        },
        {
            "input_token_details": None, "output_token_details": None
        },
        {
            "input_token_details": {
                "cache_read": None
            }, "output_token_details": {
                "reasoning": None
            }
        },
    ],
    ids=["provider-null-details", "provider-null-counts", "message-null-details", "message-null-counts"])
async def test_langchain_handler_completes_with_nullable_token_details(chain_handler_fixture, details):
    """Optional provider details must not turn a successful completion into an error."""
    handler, all_stats = chain_handler_fixture
    context = Context.get()
    initial_span = context.active_span_id
    run_id = uuid4()
    await handler.on_llm_start(serialized={"name": "rail-model"}, prompts=["check"], run_id=run_id)
    response = LLMResult(
        generations=[[Generation(text="allowed")]],
        llm_output={"token_usage": {
            "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3, **details
        }})
    await handler.on_llm_end(response, run_id=run_id)

    assert [step.event_type for step in all_stats] == [IntermediateStepType.LLM_START, IntermediateStepType.LLM_END]
    assert all_stats[-1].payload.data.output == "allowed"
    usage = all_stats[-1].payload.usage_info.token_usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (2, 1, 3)
    assert (usage.cached_tokens, usage.reasoning_tokens) == (0, 0)
    assert context.active_span_id == initial_span
    assert not handler._run_id_to_parent_span
    assert not handler._run_id_to_model_name


async def test_langchain_handler_prefers_message_usage(chain_handler_fixture):
    handler, all_stats = chain_handler_fixture
    run_id = uuid4()
    await handler.on_chat_model_start(serialized={},
                                      messages=[],
                                      run_id=run_id,
                                      metadata={"ls_model_name": "rail-model"})
    response = LLMResult(generations=[[
        ChatGeneration(message=AIMessage(content="allowed",
                                         usage_metadata={
                                             "input_tokens": 7, "output_tokens": 2, "total_tokens": 9
                                         }))
    ]],
                         llm_output={"token_usage": {
                             "prompt_tokens": 99, "completion_tokens": 99
                         }})
    await handler.on_llm_end(response, run_id=run_id)

    usage = all_stats[-1].payload.usage_info.token_usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (7, 2, 9)


async def test_langchain_handler_closes_empty_result(chain_handler_fixture):
    handler, all_stats = chain_handler_fixture
    context = Context.get()
    initial_span = context.active_span_id
    run_id = uuid4()
    await handler.on_llm_start(serialized={"name": "EmptyModel"}, prompts=[], run_id=run_id)
    await handler.on_llm_end(LLMResult(generations=[], llm_output=None), run_id=run_id)

    assert [step.event_type for step in all_stats] == [IntermediateStepType.LLM_START, IntermediateStepType.LLM_END]
    assert all_stats[-1].payload.name == "EmptyModel"
    assert all_stats[-1].payload.data.output == ""
    assert all_stats[-1].payload.metadata.chat_responses == []
    assert context.active_span_id == initial_span
    assert not handler._run_id_to_start_time


@pytest.mark.parametrize("error", [RuntimeError("provider failed"), asyncio.CancelledError()], ids=["error", "cancel"])
async def test_langchain_handler_pairs_error_and_cancellation(chain_handler_fixture, error):
    """Failures close the span and allow the same callback instance to handle another call."""
    handler, all_stats = chain_handler_fixture
    context = Context.get()
    initial_span = context.active_span_id
    run_id = uuid4()
    await handler.on_llm_start(serialized={},
                               prompts=["check"],
                               run_id=run_id,
                               invocation_params={"model": "rail-model"})
    await handler.on_llm_error(error, run_id=run_id)
    await handler.on_llm_error(error, run_id=run_id)

    assert len(all_stats) == 2
    assert all_stats[-1].event_type == IntermediateStepType.LLM_END
    assert all_stats[-1].UUID == str(run_id)
    assert all_stats[-1].payload.name == "rail-model"
    assert all_stats[-1].payload.metadata == {"status": type(error).__name__}
    assert context.active_span_id == initial_span
    assert not handler._run_id_to_model_name
    assert not handler._run_id_to_llm_input
    assert not handler._run_id_to_parent_span
    assert not handler._run_id_to_start_time

    model = FakeListLLM(responses=["recovered"], callbacks=[handler])
    assert await model.ainvoke("new call") == "recovered"
    assert len(all_stats) == 4
    assert all_stats[-1].UUID != str(run_id)
    assert context.active_span_id == initial_span


async def test_langchain_handler_closes_only_policy_owned_runs(chain_handler_fixture):
    """Boundary cleanup leaves another request's unresolved calls untouched."""
    handler, all_stats = chain_handler_fixture
    cancelled_run = uuid4()
    other_run = uuid4()
    context = Context.get()
    initial_span = context.active_span_id
    with Context.scope(active_span_id_stack=[initial_span, "policy-a"]):
        await handler.on_llm_start(serialized={}, prompts=["request a"], run_id=cancelled_run)
        with Context.scope(active_span_id_stack=[initial_span, "policy-b"]):
            await handler.on_llm_start(serialized={}, prompts=["request b"], run_id=other_run)

        await handler.close_llm_runs("policy-a", asyncio.CancelledError())
        assert context.active_span_id == "policy-a"
        assert handler._run_id_to_parent_span == {str(other_run): "policy-b"}
        assert str(other_run) in handler._run_id_to_model_name
        assert all_stats[-1].UUID == str(cancelled_run)
        assert all_stats[-1].parent_id == "policy-a"

        # A provider may finish after the policy boundary already closed its span.
        await handler.on_llm_new_token("late token", run_id=cancelled_run)
        await handler.on_llm_end(LLMResult(generations=[[Generation(text="late completion")]]), run_id=cancelled_run)
        await handler.close_llm_runs("policy-a", asyncio.CancelledError())
        assert len(all_stats) == 3

    with Context.scope(active_span_id_stack=[initial_span, "policy-b"]):
        await handler.on_llm_end(LLMResult(generations=[[Generation(text="completed b")]]), run_id=other_run)
        assert context.active_span_id == "policy-b"

    assert len(all_stats) == 4
    assert all_stats[-1].UUID == str(other_run)
    assert all_stats[-1].payload.data.output == "completed b"
    assert not handler._run_id_to_parent_span
    assert not handler._run_id_to_model_name
    assert not handler._run_id_to_llm_input
    assert not handler._run_id_to_start_time
    assert context.active_span_id == initial_span


async def test_langchain_handler_tracks_chain_runnable_events(chain_handler_fixture):
    """
    Test that LangChain Runnable/chain callbacks produce paired function stats.

      - on_chain_start -> usage stat with event_type=FUNCTION_START
      - on_chain_end -> usage stat with event_type=FUNCTION_END
    """

    handler, all_stats = chain_handler_fixture

    run_id = uuid4()
    serialized = {
        "id": ["langchain", "schema", "runnable", "RunnableLambda"],
        "name": "RunnableLambda",
    }
    inputs = {"question": "What is NAT?"}
    outputs = {"answer": "NeMo Agent Toolkit"}
    metadata = {"component": "unit-test"}
    tags = ["chain-test"]

    await handler.on_chain_start(serialized=serialized, inputs=inputs, run_id=run_id, tags=tags, metadata=metadata)
    await asyncio.sleep(0.01)
    await handler.on_chain_end(outputs=outputs, run_id=run_id, tags=tags)

    assert len(all_stats) == 2
    assert all_stats[0].event_type == IntermediateStepType.FUNCTION_START
    assert all_stats[1].event_type == IntermediateStepType.FUNCTION_END
    assert all_stats[0].UUID == str(run_id)
    assert all_stats[1].UUID == str(run_id)
    assert all_stats[0].payload.name == "RunnableLambda"
    assert all_stats[1].payload.name == "RunnableLambda"
    assert all_stats[0].payload.tags == tags
    assert all_stats[1].payload.tags == tags
    assert all_stats[0].payload.data.input == inputs
    assert all_stats[0].payload.data.payload == serialized
    assert all_stats[0].payload.metadata.span_inputs == inputs
    assert all_stats[0].payload.metadata.provided_metadata == metadata
    assert all_stats[1].payload.data.input == inputs
    assert all_stats[1].payload.data.output == outputs
    assert all_stats[1].payload.data.payload == outputs
    assert all_stats[1].payload.metadata.span_outputs == outputs
    assert all_stats[1].span_event_timestamp is not None
    assert all_stats[1].span_event_timestamp <= all_stats[1].event_timestamp
    assert str(run_id) not in handler._run_id_to_chain_input
    assert str(run_id) not in handler._run_id_to_chain_name
    assert str(run_id) not in handler._run_id_to_start_time


async def test_langchain_handler_tracks_chain_events_with_default_metadata(chain_handler_fixture):
    """Test chain callback stats when tags and metadata are omitted."""

    handler, all_stats = chain_handler_fixture

    run_id = uuid4()
    inputs = {"input": "hello"}
    outputs = {"output": "world"}

    await handler.on_chain_start(
        serialized={"id": ["langchain", "schema", "runnable", "RunnableSequence"]},
        inputs=inputs,
        run_id=run_id,
    )
    await handler.on_chain_end(outputs=outputs, run_id=run_id)

    assert len(all_stats) == 2
    assert all_stats[0].event_type == IntermediateStepType.FUNCTION_START
    assert all_stats[1].event_type == IntermediateStepType.FUNCTION_END
    assert all_stats[0].payload.name == "RunnableSequence"
    assert all_stats[1].payload.name == "RunnableSequence"
    assert all_stats[0].payload.tags is None
    assert all_stats[1].payload.tags is None
    assert all_stats[0].payload.data.input == inputs
    assert all_stats[0].payload.metadata.span_inputs == inputs
    assert all_stats[0].payload.metadata.provided_metadata is None
    assert all_stats[1].payload.data.input == inputs
    assert all_stats[1].payload.data.output == outputs
    assert all_stats[1].payload.metadata.span_outputs == outputs
    assert str(run_id) not in handler._run_id_to_chain_input
    assert str(run_id) not in handler._run_id_to_chain_name
    assert str(run_id) not in handler._run_id_to_start_time


async def test_langchain_handler_clears_chain_state_on_error(chain_handler_fixture):
    """Test that failed LangChain Runnable/chain callbacks do not leak run state."""

    handler, all_stats = chain_handler_fixture

    run_id = uuid4()
    await handler.on_chain_start(
        serialized={"name": "FailingRunnable"},
        inputs={"question": "boom?"},
        run_id=run_id,
    )
    await handler.on_chain_error(RuntimeError("boom"), run_id=run_id)

    assert len(all_stats) == 1
    assert all_stats[0].event_type == IntermediateStepType.FUNCTION_START
    assert str(run_id) not in handler._run_id_to_chain_input
    assert str(run_id) not in handler._run_id_to_chain_name
    assert str(run_id) not in handler._run_id_to_start_time


def test_extract_tools_schema_openai_format():
    """Test that OpenAI-style tool definitions are parsed correctly."""
    invocation_params = {
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "properties": {
                        "location": {
                            "type": "string"
                        }
                    },
                    "required": ["location"],
                },
            },
        }]
    }
    result = _extract_tools_schema(invocation_params)
    assert len(result) == 1
    assert result[0].function.name == "get_weather"
    assert result[0].function.description == "Get the current weather"
    assert "location" in result[0].function.parameters.properties


def test_extract_tools_schema_anthropic_format():
    """Test that Anthropic-style tool definitions (top-level name/description/input_schema) are parsed."""
    invocation_params = {
        "tools": [{
            "name": "search_database",
            "description": "Search the internal database",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", "description": "Search query"
                    },
                    "limit": {
                        "type": "integer", "description": "Max results"
                    },
                },
                "required": ["query"],
            },
        }]
    }
    result = _extract_tools_schema(invocation_params)
    assert len(result) == 1
    assert result[0].type == "function"
    assert result[0].function.name == "search_database"
    assert result[0].function.description == "Search the internal database"
    assert "query" in result[0].function.parameters.properties
    assert "limit" in result[0].function.parameters.properties
    assert result[0].function.parameters.required == ["query"]


def test_extract_tools_schema_mixed_formats():
    """Test that a mix of OpenAI and Anthropic tool formats are both parsed."""
    invocation_params = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "openai_tool",
                    "description": "An OpenAI-format tool",
                    "parameters": {
                        "properties": {
                            "x": {
                                "type": "integer"
                            }
                        },
                        "required": ["x"],
                    },
                },
            },
            {
                "name": "anthropic_tool",
                "description": "An Anthropic-format tool",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "y": {
                            "type": "string"
                        }
                    },
                    "required": [],
                },
            },
        ]
    }
    result = _extract_tools_schema(invocation_params)
    assert len(result) == 2
    assert result[0].function.name == "openai_tool"
    assert result[1].function.name == "anthropic_tool"


def test_extract_tools_schema_anthropic_additional_properties():
    """Test that additionalProperties from Anthropic input_schema is preserved."""
    invocation_params = {
        "tools": [{
            "name": "flexible_tool",
            "description": "A tool that allows extra keys",
            "input_schema": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "string"
                    }
                },
                "required": [],
                "additionalProperties": True,
            },
        }]
    }
    result = _extract_tools_schema(invocation_params)
    assert len(result) == 1
    assert result[0].function.parameters.additionalProperties is True
    assert "a" in result[0].function.parameters.properties


def test_extract_tools_schema_skips_unparseable_tool():
    """Test that an unparseable tool is skipped while valid tools are kept."""
    invocation_params = {
        "tools": [
            {
                "name": "good_tool",
                "description": "A valid Anthropic tool",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "q": {
                            "type": "string"
                        }
                    },
                    "required": ["q"],
                },
            },
            # Missing "name" — should be skipped by both parsers
            {
                "description": "no name field"
            },
        ]
    }
    result = _extract_tools_schema(invocation_params)
    assert len(result) == 1
    assert result[0].function.name == "good_tool"


def test_extract_tools_schema_skips_non_mapping_input_schema(caplog):
    """Test that a tool with a non-mapping input_schema is skipped and logged."""
    invocation_params = {
        "tools": [
            {
                "name": "good_tool",
                "description": "A valid Anthropic tool",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "q": {
                            "type": "string"
                        }
                    },
                    "required": ["q"],
                },
            },
            {
                "name": "bad_tool",
                "description": "Malformed schema",
                "input_schema": [{
                    "type": "string"
                }],
            },
        ]
    }

    with caplog.at_level(logging.DEBUG, logger="nat.plugins.langchain.callback_handler"):
        result = _extract_tools_schema(invocation_params)

    assert [tool.function.name for tool in result] == ["good_tool"]
    assert "Failed to parse tool schema" in caplog.text


def test_extract_tools_schema_empty_and_none():
    """Test edge cases: empty tools list and None invocation_params."""
    assert _extract_tools_schema({}) == []
    assert _extract_tools_schema({"tools": []}) == []
    assert _extract_tools_schema(None) == []
