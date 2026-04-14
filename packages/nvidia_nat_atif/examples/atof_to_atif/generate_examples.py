# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate ATOF v0.2 example JSONL files for EXMP-01, EXMP-02, and EXMP-03.

Each scenario preserves the structural workflow of its v0.1 counterpart
(simple calculator / nested weather lookup / branching search-and-summarize)
but demonstrates one of the three profile declaration modes introduced by
spec v0.2 §5 via a prepended ``StreamHeaderEvent``:

- **EXMP-01 (header mode)**: ``StreamHeaderEvent`` advertises
  ``profile_mode_default="header"`` and declares both reference schemas
  (``default/llm.v1`` and ``default/tool.v1``). Subsequent profiles reference
  these schemas by string ``$schema`` ID.
- **EXMP-02 (inline mode)**: ``StreamHeaderEvent`` advertises
  ``profile_mode_default="inline"`` with an empty ``schemas`` registry. Every
  profile carries an inline JSON Schema dict as its ``$schema``.
- **EXMP-03 (mixed mode)**: ``StreamHeaderEvent`` advertises
  ``profile_mode_default="header"`` and declares both reference schemas; one
  LLM event overrides the stream default via ``$mode="inline"`` + inline
  ``$schema`` dict.

Event counts: EXMP-01 = 9 events; EXMP-02 = 13 events; EXMP-03 = 15 events
(each is the v0.1 event count + 1 StreamHeaderEvent). ATIF step counts
remain unchanged from v0.1 (5 steps per scenario).

Usage:
    python generate_examples.py [--output-dir DIR]

See ATOF spec §4 (Profile Contract Protocol), §5 (Stream Header Event),
§6 (Reference Profile Implementations).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nat.atof import DefaultLlmV1
from nat.atof import DefaultToolV1
from nat.atof import Event
from nat.atof import ScopeEndEvent
from nat.atof import ScopeStartEvent
from nat.atof import StreamHeaderEvent
from nat.atof import write_jsonl

OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# EXMP-01: Simple Tool Call — HEADER MODE
# ---------------------------------------------------------------------------


def generate_exmp01() -> list[Event]:
    """EXMP-01 (9 events, header mode): single LLM -> calculator__add -> LLM cycle.

    The StreamHeaderEvent declares both reference schemas (``default/llm.v1``
    and ``default/tool.v1``); all profiles reference them by string ``$schema``
    ID. This is the compact-stream case: many profiles, small fixed schema set.
    """
    header = StreamHeaderEvent(
        uuid="stream-header-001",
        parent_uuid=None,
        timestamp="2026-01-01T00:00:00Z",
        name="exmp01_header",
        profile_mode_default="header",
        schemas={
            "default/llm.v1": DefaultLlmV1.JSON_SCHEMA,
            "default/tool.v1": DefaultToolV1.JSON_SCHEMA,
        },
    )
    return [
        header,
        ScopeStartEvent(
            uuid="scope-agent-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:00Z",
            name="simple_calculator_agent",
            scope_type="agent",
            flags=[],
        ),
        ScopeStartEvent(
            uuid="llm-001",
            parent_uuid="scope-agent-001",
            timestamp="2026-01-01T00:00:01Z",
            name="nvidia/nemotron-3-super-v3",
            scope_type="llm",
            flags=[],
            profile=DefaultLlmV1(model_name="nvidia/nemotron-3-super-v3"),
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
                                "properties": {
                                    "a": {"type": "number"},
                                    "b": {"type": "number"},
                                },
                            },
                        },
                    }
                ],
            },
        ),
        ScopeEndEvent(
            uuid="llm-001",
            parent_uuid="scope-agent-001",
            timestamp="2026-01-01T00:00:02Z",
            name="nvidia/nemotron-3-super-v3",
            scope_type="llm",
            flags=[],
            status="ok",
            profile=DefaultLlmV1(model_name="nvidia/nemotron-3-super-v3"),
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
                ],
            },
        ),
        ScopeStartEvent(
            uuid="tool-001",
            parent_uuid="scope-agent-001",
            timestamp="2026-01-01T00:00:03Z",
            name="calculator__add",
            scope_type="tool",
            flags=[],
            profile=DefaultToolV1(tool_call_id="call_calc_001"),
            input={"a": 3, "b": 4},
        ),
        ScopeEndEvent(
            uuid="tool-001",
            parent_uuid="scope-agent-001",
            timestamp="2026-01-01T00:00:04Z",
            name="calculator__add",
            scope_type="tool",
            flags=[],
            status="ok",
            profile=DefaultToolV1(tool_call_id="call_calc_001"),
            output={"result": 7},
        ),
        ScopeStartEvent(
            uuid="llm-002",
            parent_uuid="scope-agent-001",
            timestamp="2026-01-01T00:00:05Z",
            name="nvidia/nemotron-3-super-v3",
            scope_type="llm",
            flags=[],
            profile=DefaultLlmV1(model_name="nvidia/nemotron-3-super-v3"),
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
                                "function": {
                                    "name": "calculator__add",
                                    "arguments": '{"a": 3, "b": 4}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "content": '{"result": 7}', "tool_call_id": "call_calc_001"},
                ],
                "model": "nvidia/nemotron-3-super-v3",
            },
        ),
        ScopeEndEvent(
            uuid="llm-002",
            parent_uuid="scope-agent-001",
            timestamp="2026-01-01T00:00:06Z",
            name="nvidia/nemotron-3-super-v3",
            scope_type="llm",
            flags=[],
            status="ok",
            profile=DefaultLlmV1(model_name="nvidia/nemotron-3-super-v3"),
            output={
                "choices": [{"message": {"content": "The result of 3 + 4 is 7."}, "finish_reason": "stop"}],
            },
        ),
        ScopeEndEvent(
            uuid="scope-agent-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:07Z",
            name="simple_calculator_agent",
            scope_type="agent",
            flags=[],
            status="ok",
        ),
    ]


# ---------------------------------------------------------------------------
# EXMP-02: Nested Tool Chain — INLINE MODE
# ---------------------------------------------------------------------------


def _inline_llm_profile(model_name: str) -> DefaultLlmV1:
    """Construct a DefaultLlmV1 with inline ``$schema`` dict (inline-mode helper).

    The profile carries the full JSON Schema body under ``$schema`` (not a
    string ID); this is the canonical wire shape for ``profile_mode_default``
    = ``"inline"`` (spec §4.1, §5.3).
    """
    return DefaultLlmV1.model_validate(
        {
            "$schema": DefaultLlmV1.JSON_SCHEMA,
            "$version": "1.0",
            "model_name": model_name,
        }
    )


def _inline_tool_profile(tool_call_id: str) -> DefaultToolV1:
    """Construct a DefaultToolV1 with inline ``$schema`` dict (inline-mode helper)."""
    return DefaultToolV1.model_validate(
        {
            "$schema": DefaultToolV1.JSON_SCHEMA,
            "$version": "1.0",
            "tool_call_id": tool_call_id,
        }
    )


def generate_exmp02() -> list[Event]:
    """EXMP-02 (13 events, inline mode): LLM -> weather__lookup (containing temperature__to_celsius) -> LLM.

    The StreamHeaderEvent advertises ``profile_mode_default="inline"`` with an
    empty ``schemas`` registry; each profile carries its full JSON Schema body
    inline. This is the self-describing-stream case: no central registry
    required.
    """
    header = StreamHeaderEvent(
        uuid="stream-header-002",
        parent_uuid=None,
        timestamp="2026-01-01T00:01:00Z",
        name="exmp02_header",
        profile_mode_default="inline",
        schemas={},
    )
    return [
        header,
        ScopeStartEvent(
            uuid="scope-agent-002",
            parent_uuid=None,
            timestamp="2026-01-01T00:01:00Z",
            name="weather_converter_agent",
            scope_type="agent",
            flags=[],
        ),
        ScopeStartEvent(
            uuid="llm-010",
            parent_uuid="scope-agent-002",
            timestamp="2026-01-01T00:01:01Z",
            name="nvidia/nemotron-3-super-v3",
            scope_type="llm",
            flags=[],
            profile=_inline_llm_profile("nvidia/nemotron-3-super-v3"),
            input={
                "messages": [{"role": "user", "content": "What's the temperature in San Francisco in Celsius?"}],
                "model": "nvidia/nemotron-3-super-v3",
            },
        ),
        ScopeEndEvent(
            uuid="llm-010",
            parent_uuid="scope-agent-002",
            timestamp="2026-01-01T00:01:02Z",
            name="nvidia/nemotron-3-super-v3",
            scope_type="llm",
            flags=[],
            status="ok",
            profile=_inline_llm_profile("nvidia/nemotron-3-super-v3"),
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
                ],
            },
        ),
        # Function scope wraps the nested tool chain
        ScopeStartEvent(
            uuid="scope-fn-001",
            parent_uuid="scope-agent-002",
            timestamp="2026-01-01T00:01:03Z",
            name="weather__lookup",
            scope_type="function",
            flags=[],
        ),
        ScopeStartEvent(
            uuid="tool-010",
            parent_uuid="scope-fn-001",
            timestamp="2026-01-01T00:01:04Z",
            name="weather__lookup",
            scope_type="tool",
            flags=[],
            profile=_inline_tool_profile("call_weather_001"),
            input={"city": "San Francisco"},
        ),
        # Inner tool: temperature conversion
        ScopeStartEvent(
            uuid="tool-011",
            parent_uuid="scope-fn-001",
            timestamp="2026-01-01T00:01:05Z",
            name="temperature__to_celsius",
            scope_type="tool",
            flags=[],
            profile=_inline_tool_profile("call_temp_001"),
            input={"fahrenheit": 68.0},
        ),
        ScopeEndEvent(
            uuid="tool-011",
            parent_uuid="scope-fn-001",
            timestamp="2026-01-01T00:01:06Z",
            name="temperature__to_celsius",
            scope_type="tool",
            flags=[],
            status="ok",
            profile=_inline_tool_profile("call_temp_001"),
            output={"celsius": 20.0},
        ),
        ScopeEndEvent(
            uuid="tool-010",
            parent_uuid="scope-fn-001",
            timestamp="2026-01-01T00:01:07Z",
            name="weather__lookup",
            scope_type="tool",
            flags=[],
            status="ok",
            profile=_inline_tool_profile("call_weather_001"),
            output={"city": "San Francisco", "temp_f": 68.0, "temp_c": 20.0, "condition": "sunny"},
        ),
        ScopeEndEvent(
            uuid="scope-fn-001",
            parent_uuid="scope-agent-002",
            timestamp="2026-01-01T00:01:08Z",
            name="weather__lookup",
            scope_type="function",
            flags=[],
            status="ok",
        ),
        # Final LLM turn
        ScopeStartEvent(
            uuid="llm-011",
            parent_uuid="scope-agent-002",
            timestamp="2026-01-01T00:01:09Z",
            name="nvidia/nemotron-3-super-v3",
            scope_type="llm",
            flags=[],
            profile=_inline_llm_profile("nvidia/nemotron-3-super-v3"),
            input={
                "messages": [
                    {"role": "user", "content": "What's the temperature in San Francisco in Celsius?"},
                    {
                        "role": "tool",
                        "content": '{"city":"San Francisco","temp_c":20.0}',
                        "tool_call_id": "call_weather_001",
                    },
                ],
                "model": "nvidia/nemotron-3-super-v3",
            },
        ),
        ScopeEndEvent(
            uuid="llm-011",
            parent_uuid="scope-agent-002",
            timestamp="2026-01-01T00:01:10Z",
            name="nvidia/nemotron-3-super-v3",
            scope_type="llm",
            flags=[],
            status="ok",
            profile=_inline_llm_profile("nvidia/nemotron-3-super-v3"),
            output={
                "choices": [
                    {
                        "message": {"content": "It's currently 20°C (68°F) and sunny in San Francisco."},
                        "finish_reason": "stop",
                    }
                ],
            },
        ),
        ScopeEndEvent(
            uuid="scope-agent-002",
            parent_uuid=None,
            timestamp="2026-01-01T00:01:11Z",
            name="weather_converter_agent",
            scope_type="agent",
            flags=[],
            status="ok",
        ),
    ]


# ---------------------------------------------------------------------------
# EXMP-03: Branching Nested — MIXED MODE
# ---------------------------------------------------------------------------


def _inline_override_llm_profile(model_name: str) -> DefaultLlmV1:
    """Construct a DefaultLlmV1 with ``$mode="inline"`` override.

    Used in EXMP-03 to override the stream-level ``profile_mode_default =
    "header"`` on a single event. The profile still validates against the
    declared inline JSON Schema (D-15 producer-side validation applies
    regardless of mode).
    """
    return DefaultLlmV1.model_validate(
        {
            "$schema": DefaultLlmV1.JSON_SCHEMA,
            "$version": "1.0",
            "$mode": "inline",
            "model_name": model_name,
        }
    )


def generate_exmp03() -> list[Event]:
    """EXMP-03 (15 events, mixed mode): branching search-and-summarize.

    Workflow: LLM -> T1(search__web) + T2(text__word_count) as siblings,
    with T3(text__summarize) nested under T1 -> LLM.

    The StreamHeaderEvent advertises ``profile_mode_default="header"`` with
    both reference schemas declared; the final LLM event overrides via
    ``$mode="inline"`` + inline ``$schema`` dict to demonstrate the per-event
    override pattern (e.g., staging a new schema version without touching the
    central registry).
    """
    header = StreamHeaderEvent(
        uuid="stream-header-003",
        parent_uuid=None,
        timestamp="2026-01-01T00:02:00Z",
        name="exmp03_header",
        profile_mode_default="header",
        schemas={
            "default/llm.v1": DefaultLlmV1.JSON_SCHEMA,
            "default/tool.v1": DefaultToolV1.JSON_SCHEMA,
        },
    )
    return [
        header,
        ScopeStartEvent(
            uuid="scope-agent-003",
            parent_uuid=None,
            timestamp="2026-01-01T00:02:00Z",
            name="search_and_analyze_agent",
            scope_type="agent",
            flags=[],
        ),
        ScopeStartEvent(
            uuid="llm-020",
            parent_uuid="scope-agent-003",
            timestamp="2026-01-01T00:02:01Z",
            name="nvidia/nemotron-3-super-v3",
            scope_type="llm",
            flags=[],
            profile=DefaultLlmV1(model_name="nvidia/nemotron-3-super-v3"),
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": "Search for ATIF spec and summarize it, also count the words.",
                    }
                ],
                "model": "nvidia/nemotron-3-super-v3",
            },
        ),
        ScopeEndEvent(
            uuid="llm-020",
            parent_uuid="scope-agent-003",
            timestamp="2026-01-01T00:02:02Z",
            name="nvidia/nemotron-3-super-v3",
            scope_type="llm",
            flags=[],
            status="ok",
            profile=DefaultLlmV1(model_name="nvidia/nemotron-3-super-v3"),
            output={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_search_001",
                                    "type": "function",
                                    "function": {
                                        "name": "search__web",
                                        "arguments": '{"query": "ATIF spec"}',
                                    },
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
                ],
            },
        ),
        # T1: search__web with nested T3
        ScopeStartEvent(
            uuid="scope-fn-010",
            parent_uuid="scope-agent-003",
            timestamp="2026-01-01T00:02:03Z",
            name="search__web",
            scope_type="function",
            flags=[],
        ),
        ScopeStartEvent(
            uuid="tool-020",
            parent_uuid="scope-fn-010",
            timestamp="2026-01-01T00:02:04Z",
            name="search__web",
            scope_type="tool",
            flags=[],
            profile=DefaultToolV1(tool_call_id="call_search_001"),
            input={"query": "ATIF spec"},
        ),
        # T3: nested under T1's scope
        ScopeStartEvent(
            uuid="tool-022",
            parent_uuid="scope-fn-010",
            timestamp="2026-01-01T00:02:05Z",
            name="text__summarize",
            scope_type="tool",
            flags=[],
            profile=DefaultToolV1(tool_call_id="call_summarize_001"),
            input={"text": "ATIF is a trajectory format for agent evaluation..."},
        ),
        ScopeEndEvent(
            uuid="tool-022",
            parent_uuid="scope-fn-010",
            timestamp="2026-01-01T00:02:06Z",
            name="text__summarize",
            scope_type="tool",
            flags=[],
            status="ok",
            profile=DefaultToolV1(tool_call_id="call_summarize_001"),
            output={"summary": "ATIF defines a standard trajectory format for evaluating AI agents."},
        ),
        ScopeEndEvent(
            uuid="tool-020",
            parent_uuid="scope-fn-010",
            timestamp="2026-01-01T00:02:07Z",
            name="search__web",
            scope_type="tool",
            flags=[],
            status="ok",
            profile=DefaultToolV1(tool_call_id="call_search_001"),
            output={
                "results": ["ATIF spec found"],
                "summary": "ATIF defines a standard trajectory format.",
            },
        ),
        ScopeEndEvent(
            uuid="scope-fn-010",
            parent_uuid="scope-agent-003",
            timestamp="2026-01-01T00:02:08Z",
            name="search__web",
            scope_type="function",
            flags=[],
            status="ok",
        ),
        # T2: text__word_count (sibling of T1)
        ScopeStartEvent(
            uuid="tool-021",
            parent_uuid="scope-agent-003",
            timestamp="2026-01-01T00:02:09Z",
            name="text__word_count",
            scope_type="tool",
            flags=[],
            profile=DefaultToolV1(tool_call_id="call_wc_001"),
            input={"text": "The ATIF specification defines..."},
        ),
        ScopeEndEvent(
            uuid="tool-021",
            parent_uuid="scope-agent-003",
            timestamp="2026-01-01T00:02:10Z",
            name="text__word_count",
            scope_type="tool",
            flags=[],
            status="ok",
            profile=DefaultToolV1(tool_call_id="call_wc_001"),
            output={"word_count": 5},
        ),
        # Final LLM turn — MIXED-MODE PER-EVENT OVERRIDE.
        # Stream default is "header"; this profile overrides via $mode="inline"
        # and carries the full JSON Schema dict inline. Consumers see $mode
        # and validate against the inline schema rather than the registry.
        ScopeStartEvent(
            uuid="llm-021",
            parent_uuid="scope-agent-003",
            timestamp="2026-01-01T00:02:11Z",
            name="nvidia/nemotron-3-super-v3",
            scope_type="llm",
            flags=[],
            profile=_inline_override_llm_profile("meta/llama-3.1-70b-instruct"),
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": "Search for ATIF spec and summarize it, also count the words.",
                    },
                    {
                        "role": "tool",
                        "content": '{"results":["ATIF spec found"]}',
                        "tool_call_id": "call_search_001",
                    },
                    {"role": "tool", "content": '{"word_count":5}', "tool_call_id": "call_wc_001"},
                ],
                "model": "meta/llama-3.1-70b-instruct",
            },
        ),
        ScopeEndEvent(
            uuid="llm-021",
            parent_uuid="scope-agent-003",
            timestamp="2026-01-01T00:02:12Z",
            name="nvidia/nemotron-3-super-v3",
            scope_type="llm",
            flags=[],
            status="ok",
            profile=_inline_override_llm_profile("meta/llama-3.1-70b-instruct"),
            output={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "ATIF defines a standard trajectory format"
                                " for AI agent evaluation."
                                " The text contains 5 words."
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        ),
        ScopeEndEvent(
            uuid="scope-agent-003",
            parent_uuid=None,
            timestamp="2026-01-01T00:02:13Z",
            name="search_and_analyze_agent",
            scope_type="agent",
            flags=[],
            status="ok",
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ATOF v0.2 example JSONL files")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    scenarios = [
        ("exmp01_atof.jsonl", "header", generate_exmp01),
        ("exmp02_atof.jsonl", "inline", generate_exmp02),
        ("exmp03_atof.jsonl", "mixed", generate_exmp03),
    ]

    for filename, mode, generator in scenarios:
        events = generator()
        path = args.output_dir / filename
        write_jsonl(events, path)
        print(f"Wrote {len(events)} events ({mode} mode) to {path}")


if __name__ == "__main__":
    main()
