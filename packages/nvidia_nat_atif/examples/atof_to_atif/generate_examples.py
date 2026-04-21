# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate ATOF v0.1 example JSONL files for the three enrichment tiers.

Each scenario opens with a ``StreamHeaderEvent`` (always at position 0 per
spec §3.4). The three main scenarios walk through tiers 1 → 2 → 3 in order;
the ``3b`` variant is a tier-2 side example showing error-recovery semantics.

- **EXMP-01 — tier-1 raw pass-through**: A calculator-shaped workflow where the
  producer can't classify any scope. Every scope carries ``scope_type:
  "unknown"``, ``profile: null``, ``schema: null``, with opaque raw JSON in
  ``input`` / ``output``. Demonstrates the floor — a valid ATOF stream carrying
  only timing + raw payloads. Converts to a sequence of opaque ATIF system
  steps (each ``ScopeEnd`` becomes one system step).

- **EXMP-02 — tier-2 semantic-tagged**: Same calculator workflow as EXMP-01 but
  with every scope classified (``scope_type: "agent"/"llm"/"tool"``) and
  ``profile`` populated (``profile.model_name`` for llm events,
  ``profile.tool_call_id`` for tool events). ``schema`` remains null — no
  tier-3 decoding. Converts to a 5-step rich ATIF trajectory (user → agent →
  system → user → agent).

- **EXMP-03 — tier-3 schema-annotated**: Same calculator workflow as EXMP-02
  but the producer declares a schema (``openai/chat-completions.v1``) on each
  LLM event and attaches structured ``annotated_request`` /
  ``annotated_response`` payloads. The StreamHeader declares the schema in its
  registry (priority-2 fallback for any consumer that doesn't have it locally).

- **EXMP-03b — tier-2 with error recovery**: A web-search tool times out
  (``status: "error"`` + ``ErrorInfo``); the parent agent catches the failure
  and reports ``status: "ok"`` with a graceful message. Demonstrates
  cascading-status semantics from spec §5.2-5.3. Uses tier-2 shape — no
  schema layer.

Usage:
    python generate_examples.py [--output-dir DIR]

See ATOF spec §1.1 (three enrichment tiers), §3 (event kinds), §5 (status),
and ``atof-schema-profiles.md`` §6.1 (schema resolution protocol).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nat.atof import ErrorInfo
from nat.atof import Event
from nat.atof import ScopeEndEvent
from nat.atof import ScopeStartEvent
from nat.atof import StreamHeaderEvent
from nat.atof import write_jsonl

OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# Shared timestamps (deterministic for diff-able output)
# ---------------------------------------------------------------------------


def _ts(scenario: int, second: int) -> str:
    """RFC 3339 timestamp helper. Maps scenario index to a deterministic day
    in January 2026 so each example's events are sorted and diff-able.
    """
    return f"2026-01-{scenario:02d}T00:00:{second:02d}Z"


# ---------------------------------------------------------------------------
# EXMP-01: Raw pass-through — tier-1 (all scopes opaque / scope_type=unknown)
# ---------------------------------------------------------------------------


def generate_exmp01() -> list[Event]:
    """A calculator-shaped workflow where the producer can't classify any
    scope. Every scope carries ``scope_type: "unknown"``, ``profile: None``,
    ``schema: None``, and opaque raw JSON in ``input`` / ``output``. Eight
    events total (4 ScopeStart/ScopeEnd pairs).

    No ``StreamHeader`` is emitted — tier-1 producers have nothing to declare
    and the header is optional per spec §3.4. Consumers that see no leading
    header immediately know there is no stream-level schema registry; the
    4-priority resolution chain falls straight through to priority-4 (opaque).

    Demonstrates the tier-1 floor: a valid ATOF stream capturing only timing +
    raw payloads. Converts to a 4-step ATIF trajectory of opaque system steps
    via the reference converter's generic ScopeEnd fall-through
    (see README → Conversion reference).
    """
    events: list[Event] = [
        # Outer wrapper scope — opaque, no semantic class
        ScopeStartEvent(
            uuid="root-001",
            parent_uuid=None,
            timestamp=_ts(1, 0),
            name="opaque_workflow",
            attributes=[],
            scope_type="unknown",
            input={"raw_query": "What is 3 + 4?"},
        ),
        # Inner callback 1 — opaque
        ScopeStartEvent(
            uuid="inner-001",
            parent_uuid="root-001",
            timestamp=_ts(1, 1),
            name="provider_callback_1",
            attributes=[],
            scope_type="unknown",
            input={"raw_payload": "<provider request 1>"},
        ),
        ScopeEndEvent(
            uuid="inner-001",
            parent_uuid="root-001",
            timestamp=_ts(1, 2),
            name="provider_callback_1",
            attributes=[],
            scope_type="unknown",
            output={"raw_payload": "<provider response 1: tool invocation>"},
            status="ok",
        ),
        # Inner callback 2 — opaque
        ScopeStartEvent(
            uuid="inner-002",
            parent_uuid="root-001",
            timestamp=_ts(1, 3),
            name="provider_callback_2",
            attributes=[],
            scope_type="unknown",
            input={"raw_payload": "<provider request 2>"},
        ),
        ScopeEndEvent(
            uuid="inner-002",
            parent_uuid="root-001",
            timestamp=_ts(1, 4),
            name="provider_callback_2",
            attributes=[],
            scope_type="unknown",
            output={"raw_payload": "<provider response 2: tool result>"},
            status="ok",
        ),
        # Inner callback 3 — opaque
        ScopeStartEvent(
            uuid="inner-003",
            parent_uuid="root-001",
            timestamp=_ts(1, 5),
            name="provider_callback_3",
            attributes=[],
            scope_type="unknown",
            input={"raw_payload": "<provider request 3>"},
        ),
        ScopeEndEvent(
            uuid="inner-003",
            parent_uuid="root-001",
            timestamp=_ts(1, 6),
            name="provider_callback_3",
            attributes=[],
            scope_type="unknown",
            output={"raw_payload": "<provider response 3: final answer>"},
            status="ok",
        ),
        # Outer wrapper ends
        ScopeEndEvent(
            uuid="root-001",
            parent_uuid=None,
            timestamp=_ts(1, 7),
            name="opaque_workflow",
            attributes=[],
            scope_type="unknown",
            output={"raw_result": "3 + 4 = 7"},
            status="ok",
        ),
    ]
    return events


# ---------------------------------------------------------------------------
# EXMP-02: Simple Tool Call — tier-2 semantic-tagged (no schema)
# ---------------------------------------------------------------------------


def generate_exmp02() -> list[Event]:
    """A single calculator tool call. Eight events plus a StreamHeader.

    Workflow: agent → llm (decides to call calculator__add) → tool runs →
    llm (formulates final answer) → agent done.
    """
    events: list[Event] = [
        StreamHeaderEvent(
            uuid="hdr-002",
            parent_uuid=None,
            timestamp=_ts(2, 0),
            name="exmp02_header",
            schemas={},  # tier-2: no schemas declared
        ),
        ScopeStartEvent(
            uuid="agent-001",
            parent_uuid=None,
            timestamp=_ts(2, 0),
            name="calculator_agent",
            attributes=[],
            scope_type="agent",
            input="What is 3 + 4?",
        ),
        ScopeStartEvent(
            uuid="llm-001",
            parent_uuid="agent-001",
            timestamp=_ts(2, 1),
            name="gpt-4.1",
            attributes=[],
            scope_type="llm",
            profile={"model_name": "gpt-4.1"},
            input={"messages": [{"role": "user", "content": "What is 3 + 4?"}]},
        ),
        ScopeEndEvent(
            uuid="llm-001",
            parent_uuid="agent-001",
            timestamp=_ts(2, 2),
            name="gpt-4.1",
            attributes=[],
            scope_type="llm",
            profile={"model_name": "gpt-4.1"},
            output={
                "content": "",
                "tool_calls": [
                    {"id": "call_abc", "name": "calculator__add", "arguments": {"a": 3, "b": 4}},
                ],
            },
            status="ok",
        ),
        ScopeStartEvent(
            uuid="tool-001",
            parent_uuid="agent-001",
            timestamp=_ts(2, 3),
            name="calculator__add",
            attributes=[],
            scope_type="tool",
            profile={"tool_call_id": "call_abc"},
            input={"a": 3, "b": 4},
        ),
        ScopeEndEvent(
            uuid="tool-001",
            parent_uuid="agent-001",
            timestamp=_ts(2, 4),
            name="calculator__add",
            attributes=[],
            scope_type="tool",
            profile={"tool_call_id": "call_abc"},
            output={"result": 7},
            status="ok",
        ),
        ScopeStartEvent(
            uuid="llm-002",
            parent_uuid="agent-001",
            timestamp=_ts(2, 5),
            name="gpt-4.1",
            attributes=[],
            scope_type="llm",
            profile={"model_name": "gpt-4.1"},
            input={
                "messages": [
                    {"role": "user", "content": "What is 3 + 4?"},
                    {"role": "assistant", "tool_calls": [{"id": "call_abc", "name": "calculator__add"}]},
                    {"role": "tool", "tool_call_id": "call_abc", "content": "7"},
                ]
            },
        ),
        ScopeEndEvent(
            uuid="llm-002",
            parent_uuid="agent-001",
            timestamp=_ts(2, 6),
            name="gpt-4.1",
            attributes=[],
            scope_type="llm",
            profile={"model_name": "gpt-4.1"},
            output={"content": "3 + 4 = 7"},
            status="ok",
        ),
        ScopeEndEvent(
            uuid="agent-001",
            parent_uuid=None,
            timestamp=_ts(2, 7),
            name="calculator_agent",
            attributes=[],
            scope_type="agent",
            output="3 + 4 = 7",
            status="ok",
        ),
    ]
    return events


# ---------------------------------------------------------------------------
# EXMP-03b: Tool Error with Parent Recovery — tier-2 (variant of EXMP-03)
# ---------------------------------------------------------------------------


def generate_exmp03b() -> list[Event]:
    """A web-search tool times out; the parent agent catches and reports OK.

    Demonstrates spec §5.2-5.3: each scope reports its own terminal status;
    parents may catch child errors and complete normally.
    """
    events: list[Event] = [
        StreamHeaderEvent(
            uuid="hdr-003b",
            parent_uuid=None,
            timestamp=_ts(4, 0),
            name="exmp03b_header",
            schemas={},
        ),
        ScopeStartEvent(
            uuid="agent-002",
            parent_uuid=None,
            timestamp=_ts(4, 0),
            name="search_agent",
            attributes=[],
            scope_type="agent",
            input="Find recent quantum-computing news.",
        ),
        ScopeStartEvent(
            uuid="llm-003",
            parent_uuid="agent-002",
            timestamp=_ts(4, 1),
            name="gpt-4.1",
            attributes=[],
            scope_type="llm",
            profile={"model_name": "gpt-4.1"},
            input={"messages": [{"role": "user", "content": "Find recent quantum-computing news."}]},
        ),
        ScopeEndEvent(
            uuid="llm-003",
            parent_uuid="agent-002",
            timestamp=_ts(4, 2),
            name="gpt-4.1",
            attributes=[],
            scope_type="llm",
            profile={"model_name": "gpt-4.1"},
            output={
                "content": "",
                "tool_calls": [
                    {"id": "call_xyz", "name": "web_search", "arguments": {"q": "quantum computing news"}},
                ],
            },
            status="ok",
        ),
        ScopeStartEvent(
            uuid="tool-002",
            parent_uuid="agent-002",
            timestamp=_ts(4, 3),
            name="web_search",
            attributes=[],
            scope_type="tool",
            profile={"tool_call_id": "call_xyz"},
            input={"q": "quantum computing news"},
        ),
        # Tool fails after 5s timeout
        ScopeEndEvent(
            uuid="tool-002",
            parent_uuid="agent-002",
            timestamp=_ts(4, 8),
            name="web_search",
            attributes=[],
            scope_type="tool",
            profile={"tool_call_id": "call_xyz"},
            output=None,
            status="error",
            error=ErrorInfo(message="request timed out after 5s", type="TimeoutError"),
        ),
        # Parent agent catches the failure and reports OK with a graceful message
        ScopeEndEvent(
            uuid="agent-002",
            parent_uuid=None,
            timestamp=_ts(4, 10),
            name="search_agent",
            attributes=[],
            scope_type="agent",
            output="Sorry — the search service is temporarily unavailable. Please try again shortly.",
            status="ok",
        ),
    ]
    return events


# ---------------------------------------------------------------------------
# EXMP-03: Tier-3 Schema-Annotated — same calculator workflow as EXMP-02
# ---------------------------------------------------------------------------


_OPENAI_SCHEMA_REF = {"name": "openai/chat-completions", "version": "v1"}

# Inline schema body for openai/chat-completions.v1 — minimal stub for the
# example. A production schema would describe the full request/response shape.
_OPENAI_CHAT_SCHEMA = {
    "$id": "openai/chat-completions.v1",
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": True,
}


def _annotated_request_calc_q1() -> dict:
    """Schema-decoded request for the first LLM turn of EXMP-03."""
    return {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "What is 3 + 4?"}],
        "temperature": 0.7,
        "max_tokens": 1024,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator__add",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    },
                },
            },
        ],
    }


def _annotated_response_calc_q1() -> dict:
    """Schema-decoded response for the first LLM turn (tool call decision)."""
    return {
        "id": "chatcmpl-exmp03-001",
        "model": "gpt-4.1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "calculator__add", "arguments": '{"a":3,"b":4}'},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            },
        ],
        "usage": {"prompt_tokens": 84, "completion_tokens": 18, "total_tokens": 102},
    }


def _annotated_request_calc_q2() -> dict:
    """Second LLM turn — formulates the final answer with the tool result."""
    return {
        "model": "gpt-4.1",
        "messages": [
            {"role": "user", "content": "What is 3 + 4?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "calculator__add", "arguments": '{"a":3,"b":4}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "7"},
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }


def _annotated_response_calc_q2() -> dict:
    """Second LLM turn response — final answer."""
    return {
        "id": "chatcmpl-exmp03-002",
        "model": "gpt-4.1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "3 + 4 = 7"},
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 124, "completion_tokens": 8, "total_tokens": 132},
    }


def generate_exmp03() -> list[Event]:
    """EXMP-02 workflow + tier-3 schema annotations on every LLM event.

    StreamHeader declares ``openai/chat-completions.v1`` with an inline
    ``$schema`` body — consumers can validate without bundling the schema
    locally (priority-2 resolution).
    """
    events: list[Event] = [
        StreamHeaderEvent(
            uuid="hdr-003",
            parent_uuid=None,
            timestamp=_ts(3, 0),
            name="exmp03_header",
            schemas={
                "openai/chat-completions.v1": {"$schema": _OPENAI_CHAT_SCHEMA},
            },
        ),
        ScopeStartEvent(
            uuid="agent-003",
            parent_uuid=None,
            timestamp=_ts(3, 0),
            name="calculator_agent",
            attributes=[],
            scope_type="agent",
            input="What is 3 + 4?",
        ),
        ScopeStartEvent(
            uuid="llm-005",
            parent_uuid="agent-003",
            timestamp=_ts(3, 1),
            name="gpt-4.1",
            attributes=[],
            scope_type="llm",
            profile={"model_name": "gpt-4.1"},
            schema=_OPENAI_SCHEMA_REF,
            input=_annotated_request_calc_q1(),
            annotated_request=_annotated_request_calc_q1(),
        ),
        ScopeEndEvent(
            uuid="llm-005",
            parent_uuid="agent-003",
            timestamp=_ts(3, 2),
            name="gpt-4.1",
            attributes=[],
            scope_type="llm",
            profile={"model_name": "gpt-4.1"},
            schema=_OPENAI_SCHEMA_REF,
            output=_annotated_response_calc_q1(),
            annotated_response=_annotated_response_calc_q1(),
            status="ok",
        ),
        ScopeStartEvent(
            uuid="tool-003",
            parent_uuid="agent-003",
            timestamp=_ts(3, 3),
            name="calculator__add",
            attributes=[],
            scope_type="tool",
            profile={"tool_call_id": "call_abc"},
            input={"a": 3, "b": 4},
        ),
        ScopeEndEvent(
            uuid="tool-003",
            parent_uuid="agent-003",
            timestamp=_ts(3, 4),
            name="calculator__add",
            attributes=[],
            scope_type="tool",
            profile={"tool_call_id": "call_abc"},
            output={"result": 7},
            status="ok",
        ),
        ScopeStartEvent(
            uuid="llm-006",
            parent_uuid="agent-003",
            timestamp=_ts(3, 5),
            name="gpt-4.1",
            attributes=[],
            scope_type="llm",
            profile={"model_name": "gpt-4.1"},
            schema=_OPENAI_SCHEMA_REF,
            input=_annotated_request_calc_q2(),
            annotated_request=_annotated_request_calc_q2(),
        ),
        ScopeEndEvent(
            uuid="llm-006",
            parent_uuid="agent-003",
            timestamp=_ts(3, 6),
            name="gpt-4.1",
            attributes=[],
            scope_type="llm",
            profile={"model_name": "gpt-4.1"},
            schema=_OPENAI_SCHEMA_REF,
            output=_annotated_response_calc_q2(),
            annotated_response=_annotated_response_calc_q2(),
            status="ok",
        ),
        ScopeEndEvent(
            uuid="agent-003",
            parent_uuid=None,
            timestamp=_ts(3, 7),
            name="calculator_agent",
            attributes=[],
            scope_type="agent",
            output="3 + 4 = 7",
            status="ok",
        ),
    ]
    return events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for the generated JSONL files (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    scenarios = [
        ("exmp01_atof.jsonl", "tier-1 raw pass-through", generate_exmp01),
        ("exmp02_atof.jsonl", "tier-2 semantic-tagged", generate_exmp02),
        ("exmp03_atof.jsonl", "tier-3 schema-annotated", generate_exmp03),
        ("exmp03b_atof.jsonl", "tier-2 with error recovery", generate_exmp03b),
    ]

    for filename, label, generator in scenarios:
        events = generator()
        path = args.output_dir / filename
        write_jsonl(events, path)
        print(f"Wrote {len(events)} events ({label}) to {path}")


if __name__ == "__main__":
    main()
