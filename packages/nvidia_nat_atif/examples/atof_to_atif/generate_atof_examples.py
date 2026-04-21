# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate ATOF v0.1 example JSONL files.

- **EXMP-01 — tier-1 raw pass-through**: A calculator-shaped workflow where the
  producer can't classify any scope. Every scope carries ``category:
  "unknown"``, ``category_profile: null``, with opaque raw JSON in ``data``.
  Demonstrates the floor — a valid ATOF stream carrying only timing + raw
  payloads. Converts to a sequence of opaque ATIF system steps (each scope-end
  becomes one system step).

- **EXMP-02 — tier-2 semantic-tagged**: Same calculator workflow as EXMP-01 but
  with every scope classified (``category: "agent"/"llm"/"tool"``) and
  ``category_profile`` populated (``category_profile.model_name`` for llm
  events, ``category_profile.tool_call_id`` for tool events). Demonstrates
  ``attributes: ["remote"]`` on the tool scope and ``data_schema`` on the llm
  scopes. Converts to a 5-step rich ATIF trajectory (user → agent → system →
  user → agent).

- **EXMP-03 — mark events**: A short chat agent bracketed by ``mark`` events
  (session-start / session-end checkpoints). Demonstrates the second event
  kind — ``mark`` — including the generic checkpoint form (``category: null``,
  just a named timestamp with associated data).

Usage:
    python generate_atof_examples.py [--output-dir DIR]

See ATOF spec §1.1 (two enrichment tiers), §3 (event kinds), §4 (category
vocabulary).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nat.atof import Event
from nat.atof import MarkEvent
from nat.atof import ScopeEvent
from nat.atof import write_jsonl

OUTPUT_DIR = Path(__file__).parent / "output"

# Schema identifier reused by both LLM turns in EXMP-02.
_OPENAI_CHAT_SCHEMA = {"name": "openai/chat-completions", "version": "1"}


# ---------------------------------------------------------------------------
# Shared timestamps (deterministic for diff-able output)
# ---------------------------------------------------------------------------


def _ts(scenario: int, second: int) -> str:
    """RFC 3339 timestamp helper. Maps scenario index to a deterministic day
    in January 2026 so each example's events are sorted and diff-able.
    """
    return f"2026-01-{scenario:02d}T00:00:{second:02d}Z"


# ---------------------------------------------------------------------------
# EXMP-01: Raw pass-through — tier-1 (all scopes opaque / category=unknown)
# ---------------------------------------------------------------------------


def generate_exmp01() -> list[Event]:
    """A calculator-shaped workflow where the producer can't classify any
    scope. Every scope carries ``category: "unknown"``, ``category_profile:
    None``, and opaque raw JSON in ``data``. Eight events total (4 paired
    scope events).

    Demonstrates the tier-1 floor: a valid ATOF stream capturing only timing +
    raw payloads. Converts to a 4-step ATIF trajectory of opaque system steps
    via the reference converter's generic scope-end fall-through
    (see README → Conversion reference).
    """
    events: list[Event] = [
        # Outer wrapper scope — opaque, no semantic class
        ScopeEvent(
            scope_category="start",
            uuid="root-001",
            parent_uuid=None,
            timestamp=_ts(1, 0),
            name="opaque_workflow",
            attributes=[],
            category="unknown",
            data={"raw_query": "What is 3 + 4?"},
        ),
        # Inner callback 1 — opaque
        ScopeEvent(
            scope_category="start",
            uuid="inner-001",
            parent_uuid="root-001",
            timestamp=_ts(1, 1),
            name="provider_callback_1",
            attributes=[],
            category="unknown",
            data={"raw_payload": "<provider request 1>"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="inner-001",
            parent_uuid="root-001",
            timestamp=_ts(1, 2),
            name="provider_callback_1",
            attributes=[],
            category="unknown",
            data={"raw_payload": "<provider response 1: tool invocation>"},
        ),
        # Inner callback 2 — opaque
        ScopeEvent(
            scope_category="start",
            uuid="inner-002",
            parent_uuid="root-001",
            timestamp=_ts(1, 3),
            name="provider_callback_2",
            attributes=[],
            category="unknown",
            data={"raw_payload": "<provider request 2>"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="inner-002",
            parent_uuid="root-001",
            timestamp=_ts(1, 4),
            name="provider_callback_2",
            attributes=[],
            category="unknown",
            data={"raw_payload": "<provider response 2: tool result>"},
        ),
        # Inner callback 3 — opaque
        ScopeEvent(
            scope_category="start",
            uuid="inner-003",
            parent_uuid="root-001",
            timestamp=_ts(1, 5),
            name="provider_callback_3",
            attributes=[],
            category="unknown",
            data={"raw_payload": "<provider request 3>"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="inner-003",
            parent_uuid="root-001",
            timestamp=_ts(1, 6),
            name="provider_callback_3",
            attributes=[],
            category="unknown",
            data={"raw_payload": "<provider response 3: final answer>"},
        ),
        # Outer wrapper ends
        ScopeEvent(
            scope_category="end",
            uuid="root-001",
            parent_uuid=None,
            timestamp=_ts(1, 7),
            name="opaque_workflow",
            attributes=[],
            category="unknown",
            data={"raw_result": "3 + 4 = 7"},
        ),
    ]
    return events


# ---------------------------------------------------------------------------
# EXMP-02: Simple Tool Call — tier-2 semantic-tagged
# ---------------------------------------------------------------------------


def generate_exmp02() -> list[Event]:
    """A single calculator tool call. Eight events (4 paired scope events).

    Workflow: agent → llm (decides to call calculator__add) → tool runs →
    llm (formulates final answer) → agent done.

    Demonstrates:
    - ``category`` + ``category_profile`` for ``agent`` / ``llm`` / ``tool``.
    - ``attributes: ["remote"]`` on the tool scope (spec §2.1) — the tool
      executes out-of-process (HTTP / MCP / subprocess).
    - ``data_schema`` on llm scopes pointing at a canonical schema identifier
      (``openai/chat-completions.v1``) that a consumer can validate ``data``
      against (spec §2, §3).
    """
    events: list[Event] = [
        ScopeEvent(
            scope_category="start",
            uuid="agent-001",
            parent_uuid=None,
            timestamp=_ts(2, 0),
            name="calculator_agent",
            attributes=[],
            category="agent",
            data={"input": "What is 3 + 4?"},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-001",
            parent_uuid="agent-001",
            timestamp=_ts(2, 1),
            name="gpt-4.1",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gpt-4.1"},
            data={"messages": [{"role": "user", "content": "What is 3 + 4?"}]},
            data_schema=_OPENAI_CHAT_SCHEMA,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-001",
            parent_uuid="agent-001",
            timestamp=_ts(2, 2),
            name="gpt-4.1",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gpt-4.1"},
            data={
                "content": "",
                "tool_calls": [
                    {"id": "call_abc", "name": "calculator__add", "arguments": {"a": 3, "b": 4}},
                ],
            },
            data_schema=_OPENAI_CHAT_SCHEMA,
        ),
        ScopeEvent(
            scope_category="start",
            uuid="tool-001",
            parent_uuid="agent-001",
            timestamp=_ts(2, 3),
            name="calculator__add",
            attributes=["remote"],
            category="tool",
            category_profile={"tool_call_id": "call_abc"},
            data={"a": 3, "b": 4},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="tool-001",
            parent_uuid="agent-001",
            timestamp=_ts(2, 4),
            name="calculator__add",
            attributes=["remote"],
            category="tool",
            category_profile={"tool_call_id": "call_abc"},
            data={"result": 7},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-002",
            parent_uuid="agent-001",
            timestamp=_ts(2, 5),
            name="gpt-4.1",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gpt-4.1"},
            data={
                "messages": [
                    {"role": "user", "content": "What is 3 + 4?"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "name": "calculator__add",
                                "arguments": {"a": 3, "b": 4},
                            },
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_abc", "content": {"result": 7}},
                ]
            },
            data_schema=_OPENAI_CHAT_SCHEMA,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-002",
            parent_uuid="agent-001",
            timestamp=_ts(2, 6),
            name="gpt-4.1",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gpt-4.1"},
            data={"content": "3 + 4 = 7"},
            data_schema=_OPENAI_CHAT_SCHEMA,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="agent-001",
            parent_uuid=None,
            timestamp=_ts(2, 7),
            name="calculator_agent",
            attributes=[],
            category="agent",
            data={"response": "3 + 4 = 7"},
        ),
    ]
    return events


# ---------------------------------------------------------------------------
# EXMP-03: Chat agent with session-boundary mark events
# ---------------------------------------------------------------------------


def generate_exmp03() -> list[Event]:
    """A short chat agent bracketed by two ``mark`` events.

    Demonstrates the ``mark`` event kind (spec §3.2):
    - Generic checkpoint form — ``category`` and ``category_profile`` are both
      absent / null; the mark is just a named timestamp with associated data.
    - Unpaired — marks are single-shot, no start/end semantics.

    Workflow:
        session_start mark → agent turn (one llm call, no tools) → session_end mark.

    Six events total (2 marks + 2 paired scope events for agent + 2 paired
    scope events for the single llm turn). Converts to a 4-step ATIF
    trajectory (system → user → agent → system) with the two marks
    materialising as ``source: "system"`` steps carrying the mark's ``data``
    as the message.
    """
    events: list[Event] = [
        MarkEvent(
            uuid="mark-start-001",
            parent_uuid=None,
            timestamp=_ts(3, 0),
            name="session_start",
            data={"session_id": "sess-003", "user_id": "user-042"},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="agent-003",
            parent_uuid=None,
            timestamp=_ts(3, 1),
            name="chat_agent",
            attributes=[],
            category="agent",
            data={"input": "What's the capital of France?"},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-003",
            parent_uuid="agent-003",
            timestamp=_ts(3, 2),
            name="gpt-4.1",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gpt-4.1"},
            data={"messages": [{"role": "user", "content": "What's the capital of France?"}]},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-003",
            parent_uuid="agent-003",
            timestamp=_ts(3, 3),
            name="gpt-4.1",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gpt-4.1"},
            data={"content": "The capital of France is Paris."},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="agent-003",
            parent_uuid=None,
            timestamp=_ts(3, 4),
            name="chat_agent",
            attributes=[],
            category="agent",
            data={"response": "The capital of France is Paris."},
        ),
        MarkEvent(
            uuid="mark-end-001",
            parent_uuid=None,
            timestamp=_ts(3, 5),
            name="session_end",
            data={"session_id": "sess-003", "message_count": 1},
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
        ("exmp03_atof.jsonl", "mark events", generate_exmp03),
    ]

    for filename, label, generator in scenarios:
        events = generator()
        path = args.output_dir / filename
        write_jsonl(events, path)
        print(f"Wrote {len(events)} events ({label}) to {path}")


if __name__ == "__main__":
    main()
