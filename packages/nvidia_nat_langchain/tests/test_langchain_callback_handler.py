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

from nat.data_models.intermediate_step import IntermediateStepType
from nat.plugins.langchain.callback_handler import LangchainProfilerHandler
from nat.plugins.langchain.callback_handler import _extract_tools_schema
from nat.utils.reactive.subject import Subject


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
