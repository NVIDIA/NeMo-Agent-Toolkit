# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate ATOF example JSONL files for EXMP-01, EXMP-02, and EXMP-03.

Constructs ATOF event objects directly using ``nat.atof`` Pydantic models —
no NeMo-Flow runtime dependency. The generated files are self-contained
examples of the ATOF wire format.

Usage:
    python generate_examples.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nat.atof import (
    AnnotatedLLMResponse,
    LLMEndEvent,
    LLMStartEvent,
    ResponseToolCall,
    ScopeEndEvent,
    ScopeStartEvent,
    ToolEndEvent,
    ToolStartEvent,
    write_jsonl,
)

OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# EXMP-01: Simple Tool Call
# ---------------------------------------------------------------------------


def generate_exmp01() -> list:
    """Single LLM → calculator__add → LLM cycle."""
    return [
        ScopeStartEvent(
            uuid="scope-agent-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:00Z",
            name="simple_calculator_agent",
            scope_type="agent",
            attributes=0,
        ),
        LLMStartEvent(
            uuid="llm-001",
            parent_uuid="scope-agent-001",
            timestamp="2026-01-01T00:00:01Z",
            name="nvidia/nemotron-3-super-v3",
            attributes=0,
            input={
                "messages": [{"role": "user", "content": "What is 3 + 4?"}],
                "model": "nvidia/nemotron-3-super-v3",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "calculator__add",
                            "description": "Add two numbers",
                            "parameters": {
                                "type": "object",
                                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            },
                        },
                    }
                ],
            },
            model_name="nvidia/nemotron-3-super-v3",
        ),
        LLMEndEvent(
            uuid="llm-001",
            parent_uuid="scope-agent-001",
            timestamp="2026-01-01T00:00:02Z",
            name="nvidia/nemotron-3-super-v3",
            attributes=0,
            output={
                "choices": [
                    {
                        "message": {
                            "content": "I'll calculate 3 + 4 for you.",
                            "tool_calls": [
                                {
                                    "id": "call_calc_001",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator__add",
                                        "arguments": '{"a": 3, "b": 4}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            model_name="nvidia/nemotron-3-super-v3",
            annotated_response=AnnotatedLLMResponse(
                message="I'll calculate 3 + 4 for you.",
                tool_calls=[
                    ResponseToolCall(id="call_calc_001", name="calculator__add", arguments={"a": 3, "b": 4})
                ],
                finish_reason="tool_use",
            ),
        ),
        ToolStartEvent(
            uuid="tool-001",
            parent_uuid="scope-agent-001",
            timestamp="2026-01-01T00:00:03Z",
            name="calculator__add",
            attributes=0,
            input={"a": 3, "b": 4},
            tool_call_id="call_calc_001",
        ),
        ToolEndEvent(
            uuid="tool-001",
            parent_uuid="scope-agent-001",
            timestamp="2026-01-01T00:00:04Z",
            name="calculator__add",
            attributes=0,
            output={"result": 7},
            tool_call_id="call_calc_001",
        ),
        LLMStartEvent(
            uuid="llm-002",
            parent_uuid="scope-agent-001",
            timestamp="2026-01-01T00:00:05Z",
            name="nvidia/nemotron-3-super-v3",
            attributes=0,
            input={
                "messages": [
                    {"role": "user", "content": "What is 3 + 4?"},
                    {
                        "role": "assistant",
                        "content": "I'll calculate 3 + 4 for you.",
                        "tool_calls": [
                            {
                                "id": "call_calc_001",
                                "type": "function",
                                "function": {"name": "calculator__add", "arguments": '{"a": 3, "b": 4}'},
                            }
                        ],
                    },
                    {"role": "tool", "content": '{"result": 7}', "tool_call_id": "call_calc_001"},
                ],
                "model": "nvidia/nemotron-3-super-v3",
            },
            model_name="nvidia/nemotron-3-super-v3",
        ),
        LLMEndEvent(
            uuid="llm-002",
            parent_uuid="scope-agent-001",
            timestamp="2026-01-01T00:00:06Z",
            name="nvidia/nemotron-3-super-v3",
            attributes=0,
            output={
                "choices": [
                    {"message": {"content": "The result of 3 + 4 is 7."}, "finish_reason": "stop"}
                ]
            },
            model_name="nvidia/nemotron-3-super-v3",
            annotated_response=AnnotatedLLMResponse(
                message="The result of 3 + 4 is 7.",
                finish_reason="complete",
            ),
        ),
        ScopeEndEvent(
            uuid="scope-agent-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:07Z",
            name="simple_calculator_agent",
            scope_type="agent",
            attributes=0,
        ),
    ]


# ---------------------------------------------------------------------------
# EXMP-02: Nested Tool Chain (2-level)
# ---------------------------------------------------------------------------


def generate_exmp02() -> list:
    """LLM → weather__lookup (containing temperature__to_celsius) → LLM."""
    return [
        ScopeStartEvent(
            uuid="scope-agent-002",
            parent_uuid=None,
            timestamp="2026-01-01T00:01:00Z",
            name="weather_converter_agent",
            scope_type="agent",
            attributes=0,
        ),
        LLMStartEvent(
            uuid="llm-010",
            parent_uuid="scope-agent-002",
            timestamp="2026-01-01T00:01:01Z",
            name="nvidia/nemotron-3-super-v3",
            attributes=0,
            input={
                "messages": [{"role": "user", "content": "What's the temperature in San Francisco in Celsius?"}],
                "model": "nvidia/nemotron-3-super-v3",
            },
            model_name="nvidia/nemotron-3-super-v3",
        ),
        LLMEndEvent(
            uuid="llm-010",
            parent_uuid="scope-agent-002",
            timestamp="2026-01-01T00:01:02Z",
            name="nvidia/nemotron-3-super-v3",
            attributes=0,
            output={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_weather_001",
                                    "type": "function",
                                    "function": {
                                        "name": "weather__lookup",
                                        "arguments": '{"city": "San Francisco"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            model_name="nvidia/nemotron-3-super-v3",
            annotated_response=AnnotatedLLMResponse(
                message="",
                tool_calls=[
                    ResponseToolCall(
                        id="call_weather_001", name="weather__lookup", arguments={"city": "San Francisco"}
                    )
                ],
                finish_reason="tool_use",
            ),
        ),
        # Function scope wraps the nested tool chain
        ScopeStartEvent(
            uuid="scope-fn-001",
            parent_uuid="scope-agent-002",
            timestamp="2026-01-01T00:01:03Z",
            name="weather__lookup",
            scope_type="function",
            attributes=0,
        ),
        ToolStartEvent(
            uuid="tool-010",
            parent_uuid="scope-fn-001",
            timestamp="2026-01-01T00:01:04Z",
            name="weather__lookup",
            attributes=0,
            input={"city": "San Francisco"},
            tool_call_id="call_weather_001",
        ),
        # Inner tool: temperature conversion
        ToolStartEvent(
            uuid="tool-011",
            parent_uuid="scope-fn-001",
            timestamp="2026-01-01T00:01:05Z",
            name="temperature__to_celsius",
            attributes=0,
            input={"fahrenheit": 68.0},
            tool_call_id="call_temp_001",
        ),
        ToolEndEvent(
            uuid="tool-011",
            parent_uuid="scope-fn-001",
            timestamp="2026-01-01T00:01:06Z",
            name="temperature__to_celsius",
            attributes=0,
            output={"celsius": 20.0},
            tool_call_id="call_temp_001",
        ),
        ToolEndEvent(
            uuid="tool-010",
            parent_uuid="scope-fn-001",
            timestamp="2026-01-01T00:01:07Z",
            name="weather__lookup",
            attributes=0,
            output={"city": "San Francisco", "temp_f": 68.0, "temp_c": 20.0, "condition": "sunny"},
            tool_call_id="call_weather_001",
        ),
        ScopeEndEvent(
            uuid="scope-fn-001",
            parent_uuid="scope-agent-002",
            timestamp="2026-01-01T00:01:08Z",
            name="weather__lookup",
            scope_type="function",
            attributes=0,
        ),
        # Final LLM turn
        LLMStartEvent(
            uuid="llm-011",
            parent_uuid="scope-agent-002",
            timestamp="2026-01-01T00:01:09Z",
            name="nvidia/nemotron-3-super-v3",
            attributes=0,
            input={
                "messages": [
                    {"role": "user", "content": "What's the temperature in San Francisco in Celsius?"},
                    {"role": "tool", "content": '{"city":"San Francisco","temp_c":20.0}', "tool_call_id": "call_weather_001"},
                ],
                "model": "nvidia/nemotron-3-super-v3",
            },
            model_name="nvidia/nemotron-3-super-v3",
        ),
        LLMEndEvent(
            uuid="llm-011",
            parent_uuid="scope-agent-002",
            timestamp="2026-01-01T00:01:10Z",
            name="nvidia/nemotron-3-super-v3",
            attributes=0,
            output={
                "choices": [
                    {"message": {"content": "It's currently 20°C (68°F) and sunny in San Francisco."}, "finish_reason": "stop"}
                ]
            },
            model_name="nvidia/nemotron-3-super-v3",
            annotated_response=AnnotatedLLMResponse(
                message="It's currently 20°C (68°F) and sunny in San Francisco.",
                finish_reason="complete",
            ),
        ),
        ScopeEndEvent(
            uuid="scope-agent-002",
            parent_uuid=None,
            timestamp="2026-01-01T00:01:11Z",
            name="weather_converter_agent",
            scope_type="agent",
            attributes=0,
        ),
    ]


# ---------------------------------------------------------------------------
# EXMP-03: Branching Nested
# ---------------------------------------------------------------------------


def generate_exmp03() -> list:
    """LLM → T1(search__web) + T2(text__word_count) as siblings, T3(text__summarize) nested under T1 → LLM."""
    return [
        ScopeStartEvent(
            uuid="scope-agent-003",
            parent_uuid=None,
            timestamp="2026-01-01T00:02:00Z",
            name="search_and_analyze_agent",
            scope_type="agent",
            attributes=0,
        ),
        LLMStartEvent(
            uuid="llm-020",
            parent_uuid="scope-agent-003",
            timestamp="2026-01-01T00:02:01Z",
            name="nvidia/nemotron-3-super-v3",
            attributes=0,
            input={
                "messages": [{"role": "user", "content": "Search for ATIF spec and summarize it, also count the words."}],
                "model": "nvidia/nemotron-3-super-v3",
            },
            model_name="nvidia/nemotron-3-super-v3",
        ),
        LLMEndEvent(
            uuid="llm-020",
            parent_uuid="scope-agent-003",
            timestamp="2026-01-01T00:02:02Z",
            name="nvidia/nemotron-3-super-v3",
            attributes=0,
            output={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_search_001",
                                    "type": "function",
                                    "function": {"name": "search__web", "arguments": '{"query": "ATIF spec"}'},
                                },
                                {
                                    "id": "call_wc_001",
                                    "type": "function",
                                    "function": {
                                        "name": "text__word_count",
                                        "arguments": '{"text": "The ATIF specification defines..."}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            model_name="nvidia/nemotron-3-super-v3",
            annotated_response=AnnotatedLLMResponse(
                message="",
                tool_calls=[
                    ResponseToolCall(id="call_search_001", name="search__web", arguments={"query": "ATIF spec"}),
                    ResponseToolCall(
                        id="call_wc_001",
                        name="text__word_count",
                        arguments={"text": "The ATIF specification defines..."},
                    ),
                ],
                finish_reason="tool_use",
            ),
        ),
        # T1: search__web with nested T3
        ScopeStartEvent(
            uuid="scope-fn-010",
            parent_uuid="scope-agent-003",
            timestamp="2026-01-01T00:02:03Z",
            name="search__web",
            scope_type="function",
            attributes=0,
        ),
        ToolStartEvent(
            uuid="tool-020",
            parent_uuid="scope-fn-010",
            timestamp="2026-01-01T00:02:04Z",
            name="search__web",
            attributes=0,
            input={"query": "ATIF spec"},
            tool_call_id="call_search_001",
        ),
        # T3: nested under T1's scope
        ToolStartEvent(
            uuid="tool-022",
            parent_uuid="scope-fn-010",
            timestamp="2026-01-01T00:02:05Z",
            name="text__summarize",
            attributes=0,
            input={"text": "ATIF is a trajectory format for agent evaluation..."},
            tool_call_id="call_summarize_001",
        ),
        ToolEndEvent(
            uuid="tool-022",
            parent_uuid="scope-fn-010",
            timestamp="2026-01-01T00:02:06Z",
            name="text__summarize",
            attributes=0,
            output={"summary": "ATIF defines a standard trajectory format for evaluating AI agents."},
            tool_call_id="call_summarize_001",
        ),
        ToolEndEvent(
            uuid="tool-020",
            parent_uuid="scope-fn-010",
            timestamp="2026-01-01T00:02:07Z",
            name="search__web",
            attributes=0,
            output={"results": ["ATIF spec found"], "summary": "ATIF defines a standard trajectory format."},
            tool_call_id="call_search_001",
        ),
        ScopeEndEvent(
            uuid="scope-fn-010",
            parent_uuid="scope-agent-003",
            timestamp="2026-01-01T00:02:08Z",
            name="search__web",
            scope_type="function",
            attributes=0,
        ),
        # T2: text__word_count (sibling of T1)
        ToolStartEvent(
            uuid="tool-021",
            parent_uuid="scope-agent-003",
            timestamp="2026-01-01T00:02:09Z",
            name="text__word_count",
            attributes=0,
            input={"text": "The ATIF specification defines..."},
            tool_call_id="call_wc_001",
        ),
        ToolEndEvent(
            uuid="tool-021",
            parent_uuid="scope-agent-003",
            timestamp="2026-01-01T00:02:10Z",
            name="text__word_count",
            attributes=0,
            output={"word_count": 5},
            tool_call_id="call_wc_001",
        ),
        # Final LLM turn
        LLMStartEvent(
            uuid="llm-021",
            parent_uuid="scope-agent-003",
            timestamp="2026-01-01T00:02:11Z",
            name="nvidia/nemotron-3-super-v3",
            attributes=0,
            input={
                "messages": [
                    {"role": "user", "content": "Search for ATIF spec and summarize it, also count the words."},
                    {"role": "tool", "content": '{"results":["ATIF spec found"]}', "tool_call_id": "call_search_001"},
                    {"role": "tool", "content": '{"word_count":5}', "tool_call_id": "call_wc_001"},
                ],
                "model": "nvidia/nemotron-3-super-v3",
            },
            model_name="nvidia/nemotron-3-super-v3",
        ),
        LLMEndEvent(
            uuid="llm-021",
            parent_uuid="scope-agent-003",
            timestamp="2026-01-01T00:02:12Z",
            name="nvidia/nemotron-3-super-v3",
            attributes=0,
            output={
                "choices": [
                    {
                        "message": {
                            "content": "ATIF defines a standard trajectory format for AI agent evaluation. The text contains 5 words."
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
            model_name="nvidia/nemotron-3-super-v3",
            annotated_response=AnnotatedLLMResponse(
                message="ATIF defines a standard trajectory format for AI agent evaluation. The text contains 5 words.",
                finish_reason="complete",
            ),
        ),
        ScopeEndEvent(
            uuid="scope-agent-003",
            parent_uuid=None,
            timestamp="2026-01-01T00:02:13Z",
            name="search_and_analyze_agent",
            scope_type="agent",
            attributes=0,
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ATOF example JSONL files")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    scenarios = [
        ("exmp01_simple_tool_call.jsonl", generate_exmp01),
        ("exmp02_nested_tool_chain.jsonl", generate_exmp02),
        ("exmp03_branching_nested.jsonl", generate_exmp03),
    ]

    for filename, generator in scenarios:
        events = generator()
        path = args.output_dir / filename
        write_jsonl(events, path)
        print(f"Wrote {len(events)} events to {path}")


if __name__ == "__main__":
    main()
