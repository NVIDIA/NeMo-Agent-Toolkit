# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tier-1 ATOF → ATIF conversion tests.

Verifies that a strict tier-1 ATOF stream (all ``category: "unknown"``)
produces a non-empty ATIF trajectory via the reference converter.

Tier-1 is the raw pass-through enrichment level (``atof-event-format.md``
§1.1): producers know nothing semantic, so every scope carries
``category: "unknown"``, ``category_profile: null``, and opaque raw JSON
in ``data``. The converter must still materialise each opaque scope-end
as an ATIF system step, and must fall back to the root scope's name for
``Trajectory.agent.name`` when no classified agent scope exists.

Runnable either via ``pytest`` or as a script:
    uv run pytest packages/nvidia_nat_atif/tests/test_tier1_conversion.py
    uv run python packages/nvidia_nat_atif/tests/test_tier1_conversion.py
"""

from __future__ import annotations

import json

from nat.atof import ScopeEvent
from nat.atof.scripts.atof_to_atif_converter import convert


def _tier1_stream() -> list:
    """Build an 8-event tier-1 stream: a calculator-like workflow where the
    producer cannot classify any scope (every ``category`` is ``"unknown"``).

    Structural shape mirrors EXMP-01 (outer wrapper → inner provider call →
    inner tool call → inner provider call → outer ends) but every scope is
    opaque. Used to verify the converter still produces a readable trajectory.
    """
    return [
        ScopeEvent(
            scope_category="start",
            uuid="root-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:00Z",
            name="calculator_agent",
            attributes=[],
            category="unknown",
            data={"query": "What is 3 + 4?"},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="inner-001",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:01Z",
            name="provider_call_1",
            attributes=[],
            category="unknown",
            data={"raw": "opaque request 1"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="inner-001",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:02Z",
            name="provider_call_1",
            attributes=[],
            category="unknown",
            data={"raw": "opaque response 1"},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="inner-002",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:03Z",
            name="provider_call_2",
            attributes=[],
            category="unknown",
            data={"raw": "opaque request 2"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="inner-002",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:04Z",
            name="provider_call_2",
            attributes=[],
            category="unknown",
            data={"raw": "opaque response 2"},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="inner-003",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:05Z",
            name="provider_call_3",
            attributes=[],
            category="unknown",
            data={"raw": "opaque request 3"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="inner-003",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:06Z",
            name="provider_call_3",
            attributes=[],
            category="unknown",
            data={"raw": "opaque response 3"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="root-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:07Z",
            name="calculator_agent",
            attributes=[],
            category="unknown",
            data="3 + 4 = 7",
        ),
    ]


def test_tier1_produces_nonempty_trajectory() -> None:
    """Every opaque scope-end (category='unknown') becomes a system step."""
    events = _tier1_stream()
    trajectory = convert(events)

    # Four scope-end events with category='unknown' → 4 system steps.
    assert len(trajectory.steps) == 4, f"expected 4 system steps, got {len(trajectory.steps)}"

    sources = [step.source for step in trajectory.steps]
    assert sources == ["system"] * 4, f"expected all 'system' sources, got {sources}"

    # Step IDs must be sequential from 1 (spec §7 in converter doc).
    assert [step.step_id for step in trajectory.steps] == [1, 2, 3, 4]


def test_tier1_agent_name_falls_back_to_root_scope() -> None:
    """With no category='agent' present, Trajectory.agent.name uses the
    outermost (parent_uuid=None) scope-start's name.
    """
    events = _tier1_stream()
    trajectory = convert(events)

    assert trajectory.agent.name == "calculator_agent", (
        f"expected root-scope fallback 'calculator_agent', got {trajectory.agent.name!r}"
    )
    # No LLM scope exists → no model_name resolvable.
    assert trajectory.agent.model_name is None


def test_tier1_preserves_opaque_payloads() -> None:
    """Each system step carries the scope's raw data as its message."""
    events = _tier1_stream()
    trajectory = convert(events)

    # First three steps come from inner scope-ends with dict data → JSON-encoded.
    assert trajectory.steps[0].message == json.dumps({"raw": "opaque response 1"}, separators=(",", ":"))
    assert trajectory.steps[1].message == json.dumps({"raw": "opaque response 2"}, separators=(",", ":"))
    assert trajectory.steps[2].message == json.dumps({"raw": "opaque response 3"}, separators=(",", ":"))
    # Fourth step is the root scope-end with a plain string data.
    assert trajectory.steps[3].message == "3 + 4 = 7"


def test_tier1_preserves_ancestry_and_invocation_timing() -> None:
    """Every tier-1 system step carries ancestry + invocation-timing metadata.

    ``Step.extra`` is a loosely-typed ``dict[str, Any]`` on the ATIF side, so
    accessors are dict-style rather than attribute-style.
    """
    events = _tier1_stream()
    trajectory = convert(events)

    for step in trajectory.steps:
        assert step.extra is not None, f"step {step.step_id} missing extra"
        ancestry = step.extra.get("ancestry")
        invocation = step.extra.get("invocation")
        assert ancestry is not None, f"step {step.step_id} missing ancestry"
        assert ancestry.get("function_id"), f"step {step.step_id} missing function_id"
        assert ancestry.get("function_name"), f"step {step.step_id} missing function_name"
        assert invocation is not None, f"step {step.step_id} missing invocation"
        assert invocation.get("start_timestamp") is not None
        assert invocation.get("end_timestamp") is not None
        assert invocation["start_timestamp"] < invocation["end_timestamp"]

    # Root step should have parent_id == "" (root scope has parent_uuid=None).
    root_step = trajectory.steps[3]
    assert root_step.extra["ancestry"]["parent_id"] == ""
    # Inner steps should all reference root-001 as parent.
    for inner_step in trajectory.steps[:3]:
        assert inner_step.extra["ancestry"]["parent_id"] == "root-001"
        assert inner_step.extra["ancestry"]["parent_name"] == "calculator_agent"


if __name__ == "__main__":
    test_tier1_produces_nonempty_trajectory()
    test_tier1_agent_name_falls_back_to_root_scope()
    test_tier1_preserves_opaque_payloads()
    test_tier1_preserves_ancestry_and_invocation_timing()
    print("All tier-1 conversion tests passed.")
