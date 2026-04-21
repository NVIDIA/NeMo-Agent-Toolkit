# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate ATOF v0.1 example JSONL files for the two enrichment tiers.

- **EXMP-01 — tier-1 raw pass-through**: A calculator-shaped workflow where the
  producer can't classify any scope. Every scope carries ``category:
  "unknown"``, ``category_profile: null``, with opaque raw JSON in ``data``.
  Demonstrates the floor — a valid ATOF stream carrying only timing + raw
  payloads. Converts to a sequence of opaque ATIF system steps (each scope-end
  becomes one system step).

- **EXMP-02 — tier-2 semantic-tagged**: Same calculator workflow as EXMP-01 but
  with every scope classified (``category: "agent"/"llm"/"tool"``) and
  ``category_profile`` populated (``category_profile.model_name`` for llm
  events, ``category_profile.tool_call_id`` for tool events). Converts to a
  5-step rich ATIF trajectory (user → agent → system → user → agent).

Usage:
    python generate_atof_examples.py [--output-dir DIR]

See ATOF spec §1.1 (two enrichment tiers), §3 (event kinds), §4 (category
vocabulary).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nat.atof import Event
from nat.atof import ScopeEvent
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
            data="What is 3 + 4?",
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
        ),
        ScopeEvent(
            scope_category="start",
            uuid="tool-001",
            parent_uuid="agent-001",
            timestamp=_ts(2, 3),
            name="calculator__add",
            attributes=[],
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
            attributes=[],
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
                    {"role": "assistant", "tool_calls": [{"id": "call_abc", "name": "calculator__add"}]},
                    {"role": "tool", "tool_call_id": "call_abc", "content": "7"},
                ]
            },
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
        ),
        ScopeEvent(
            scope_category="end",
            uuid="agent-001",
            parent_uuid=None,
            timestamp=_ts(2, 7),
            name="calculator_agent",
            attributes=[],
            category="agent",
            data="3 + 4 = 7",
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
    ]

    for filename, label, generator in scenarios:
        events = generator()
        path = args.output_dir / filename
        write_jsonl(events, path)
        print(f"Wrote {len(events)} events ({label}) to {path}")


if __name__ == "__main__":
    main()
