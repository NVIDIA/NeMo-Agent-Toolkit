# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tier-1 ATOF → ATIF conversion tests.

Verifies that a strict tier-1 ATOF stream (all ``category: "unknown"``)
produces a non-empty ATIF trajectory via the reference converter.

Tier-1 is the raw pass-through enrichment level (``atof-event-format.md``
§1.1): producers know nothing semantic, so every scope carries
``category: "unknown"``, ``category_profile: null``, and opaque raw JSON
in ``data``. The converter materialises tier-1 streams under the
**boundary-promotion default** introduced by quick task 260501-1ko:

- The root opaque scope-start lifts its ``data`` payload into a leading
  ``source: "user"`` step (Branch A).
- The root opaque scope-end lifts its ``data`` payload into a trailing
  ``source: "agent"`` step (Branch B).
- Inner (non-root) opaque scope-ends remain ``source: "system"``
  (unchanged behavior).
- ``Trajectory.agent.name`` still falls back to the outermost root
  scope's ``name`` when no ``category: "agent"`` event is present.

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
            data={"result": "3 + 4 = 7"},
        ),
    ]


def test_tier1_produces_nonempty_trajectory() -> None:
    """Tier-1 boundary-promotion default emits user + 3 system + agent shape.

    Branch A lifts the root scope-start into a leading ``user`` step.
    Three inner opaque scope-ends emit ``system`` steps (unchanged).
    Branch B lifts the root scope-end into a trailing ``agent`` step.
    """
    events = _tier1_stream()
    trajectory = convert(events)

    # Branch A user step + 3 inner system steps + Branch B agent step = 5.
    assert len(trajectory.steps) == 5, f"expected 5 steps, got {len(trajectory.steps)}"

    sources = [step.source for step in trajectory.steps]
    assert sources == ["user", "system", "system", "system", "agent"], f"expected user→3xsystem→agent, got {sources}"

    # Step IDs must be sequential from 1 (spec §7 in converter doc).
    assert [step.step_id for step in trajectory.steps] == [1, 2, 3, 4, 5]


def test_tier1_agent_name_falls_back_to_root_scope() -> None:
    """With no category='agent' present, Trajectory.agent.name uses the
    outermost (parent_uuid=None) scope-start's name.
    """
    events = _tier1_stream()
    trajectory = convert(events)

    assert trajectory.agent.name == "calculator_agent", (
        f"expected root-scope fallback 'calculator_agent', got {trajectory.agent.name!r}")
    # No LLM scope exists → no model_name resolvable.
    assert trajectory.agent.model_name is None


def test_tier1_preserves_opaque_payloads() -> None:
    """Branch A/B lift root-scope data; inner scope-ends keep raw JSON shape.

    The root scope's ``data`` is a single-key dict whose value is a string
    (``{"query": "..."}`` and ``{"result": "..."}``), so the
    ``_serialize_root_data`` heuristic lifts the bare string into the
    boundary-step ``message``. Inner scope-ends keep the existing
    JSON-serialization behavior.
    """
    events = _tier1_stream()
    trajectory = convert(events)

    # Step 1 (Branch A user step): single-key {"query": "..."} lifted to bare string.
    assert trajectory.steps[0].message == "What is 3 + 4?"

    # Steps 2-4 (inner system steps): unchanged JSON-serialized dict messages.
    assert trajectory.steps[1].message == json.dumps({"raw": "opaque response 1"}, separators=(",", ":"))
    assert trajectory.steps[2].message == json.dumps({"raw": "opaque response 2"}, separators=(",", ":"))
    assert trajectory.steps[3].message == json.dumps({"raw": "opaque response 3"}, separators=(",", ":"))

    # Step 5 (Branch B agent step): single-key {"result": "..."} lifted to bare string.
    assert trajectory.steps[4].message == "3 + 4 = 7"


def test_tier1_preserves_ancestry_and_invocation_timing() -> None:
    """Every tier-1 step (including new boundary user/agent steps) carries
    ancestry + invocation-timing metadata.

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
        # Branch A user step is emitted AT the root scope-start event itself
        # — there is no elapsed scope window yet, so start == end. Boundary
        # steps for non-start emissions still see start < end.
        if step.source == "user" and ancestry.get("function_id") == "root-001":
            assert invocation["start_timestamp"] == invocation["end_timestamp"]
        else:
            assert invocation["start_timestamp"] < invocation["end_timestamp"]

    # Branch A user step is anchored at the root scope-start: parent_id is None.
    user_step = trajectory.steps[0]
    assert user_step.source == "user"
    assert user_step.extra["ancestry"]["function_id"] == "root-001"
    assert user_step.extra["ancestry"]["parent_id"] is None

    # Inner system steps reference root-001 as parent.
    for inner_step in trajectory.steps[1:4]:
        assert inner_step.source == "system"
        assert inner_step.extra["ancestry"]["parent_id"] == "root-001"
        assert inner_step.extra["ancestry"]["parent_name"] == "calculator_agent"

    # Branch B agent step is anchored at the root scope-end: parent_id is None.
    agent_step = trajectory.steps[4]
    assert agent_step.source == "agent"
    assert agent_step.extra["ancestry"]["function_id"] == "root-001"
    assert agent_step.extra["ancestry"]["parent_id"] is None


# ---------------------------------------------------------------------------
# 260501-1ko boundary-promotion tests
# ---------------------------------------------------------------------------


def _root_only_pair(start_data, end_data) -> list:
    """Minimal root-only opaque scope pair (no inner scopes)."""
    return [
        ScopeEvent(
            scope_category="start",
            uuid="root-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:00Z",
            name="opaque_workflow",
            attributes=[],
            category="unknown",
            data=start_data,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="root-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:01Z",
            name="opaque_workflow",
            attributes=[],
            category="unknown",
            data=end_data,
        ),
    ]


def test_tier1_root_promotes_raw_query_to_user_step() -> None:
    """A root opaque scope-start with data={"query": "..."} emits a leading
    source='user' step whose message is the lifted string (single-key-dict
    lift heuristic).
    """
    events = _root_only_pair(
        start_data={"query": "What is the meaning of life?"},
        end_data=None,  # empty root-end → no agent step
    )
    trajectory = convert(events)

    assert len(trajectory.steps) >= 1
    assert trajectory.steps[0].source == "user"
    assert trajectory.steps[0].message == "What is the meaning of life?"


def test_tier1_root_promotes_raw_result_to_agent_step() -> None:
    """A root opaque scope-end with data={"result": "..."} emits a trailing
    source='agent' step whose message is the lifted string.
    """
    events = _root_only_pair(
        start_data=None,  # empty root-start → no user step
        end_data={"result": "42"},
    )
    trajectory = convert(events)

    assert len(trajectory.steps) >= 1
    last = trajectory.steps[-1]
    assert last.source == "agent"
    assert last.message == "42"


def test_tier1_root_dict_data_serializes_as_json() -> None:
    """A root scope event with multi-key dict data serializes to compact JSON
    in the boundary step (the single-key-dict lift heuristic does NOT apply).
    """
    events = _root_only_pair(
        start_data={
            "a": 1, "b": "two"
        },
        end_data={
            "x": "ok", "y": 7
        },
    )
    trajectory = convert(events)

    assert len(trajectory.steps) == 2

    # User step: multi-key dict → compact JSON.
    user_step = trajectory.steps[0]
    assert user_step.source == "user"
    user_payload = json.loads(user_step.message)
    assert user_payload == {"a": 1, "b": "two"}
    # Compact JSON has no whitespace separators.
    assert " " not in user_step.message

    # Agent step: same heuristic.
    agent_step = trajectory.steps[1]
    assert agent_step.source == "agent"
    agent_payload = json.loads(agent_step.message)
    assert agent_payload == {"x": "ok", "y": 7}
    assert " " not in agent_step.message


def test_tier1_root_empty_data_emits_no_boundary_steps() -> None:
    """A root scope-start AND scope-end with None/{} data emit NEITHER a
    user step NOR an agent step. Inner system steps (if any) still emit.
    """
    events = [
        ScopeEvent(
            scope_category="start",
            uuid="root-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:00Z",
            name="opaque_workflow",
            attributes=[],
            category="unknown",
            data=None,
        ),
        ScopeEvent(
            scope_category="start",
            uuid="inner-001",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:01Z",
            name="provider_call",
            attributes=[],
            category="unknown",
            data={"raw": "opaque request"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="inner-001",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:02Z",
            name="provider_call",
            attributes=[],
            category="unknown",
            data={"raw": "opaque response"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="root-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:03Z",
            name="opaque_workflow",
            attributes=[],
            category="unknown",
            data={},
        ),
    ]
    trajectory = convert(events)

    # No user step from Branch A (root start data was None).
    # No agent step from Branch B (root end data was {}).
    sources = [s.source for s in trajectory.steps]
    assert "user" not in sources, f"expected no user step on empty root start data, got {sources}"
    assert "agent" not in sources, f"expected no agent step on empty root end data, got {sources}"

    # Inner opaque scope-end still emits a system step.
    assert "system" in sources, f"expected inner system step to remain, got {sources}"
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].message == json.dumps({"raw": "opaque response"}, separators=(",", ":"))


def test_tier1_inner_scopes_remain_system_steps() -> None:
    """Regression guard: opaque scope-ends with parent_uuid != None still
    emit source='system' (Branch B must not affect inner non-boundary scopes).
    """
    events = [
        ScopeEvent(
            scope_category="start",
            uuid="root-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:00Z",
            name="opaque_workflow",
            attributes=[],
            category="unknown",
            data={"query": "go"},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="child-001",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:01Z",
            name="child_op",
            attributes=[],
            category="unknown",
            data={"raw": "child request"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="child-001",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:02Z",
            name="child_op",
            attributes=[],
            category="unknown",
            data={"detail": "child response"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="root-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:03Z",
            name="opaque_workflow",
            attributes=[],
            category="unknown",
            data={"result": "done"},
        ),
    ]
    trajectory = convert(events)

    # Expect: user (root start) → system (inner end) → agent (root end).
    assert len(trajectory.steps) == 3
    assert [s.source for s in trajectory.steps] == ["user", "system", "agent"]

    # The inner system step preserves the raw single-key-dict shape exactly
    # as before — Branch B uses the legacy serialization for non-root ends.
    inner = trajectory.steps[1]
    assert inner.source == "system"
    assert inner.message == json.dumps({"detail": "child response"}, separators=(",", ":"))
    assert inner.extra["ancestry"]["parent_id"] == "root-001"


def test_classified_agent_root_unchanged() -> None:
    """Regression guard for exmp02-shape: when the root scope IS an 'agent'
    (or llm/tool/context) scope, NO Branch A user step is emitted from
    Branch A and the existing handler path is taken (R1 metadata-only).

    Branch A's predicate explicitly excludes ``category in
    ("agent", "llm", "tool", "context")`` so a classified root never
    triggers boundary promotion.
    """
    events = [
        # Root is a category="agent" scope — handled by R1 (no step).
        ScopeEvent(
            scope_category="start",
            uuid="agent-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:00Z",
            name="classified_agent",
            attributes=[],
            category="agent",
            data={"query": "ignored — Branch A must not fire"},
        ),
        # An LLM scope under the agent — produces user + agent steps via R2/R4.
        ScopeEvent(
            scope_category="start",
            uuid="llm-001",
            parent_uuid="agent-001",
            timestamp="2026-01-01T00:00:01Z",
            name="provider_call",
            attributes=[],
            category="llm",
            category_profile={"model_name": "test-model"},
            data={"messages": [{
                "role": "user", "content": "hi"
            }]},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-001",
            parent_uuid="agent-001",
            timestamp="2026-01-01T00:00:02Z",
            name="provider_call",
            attributes=[],
            category="llm",
            category_profile={"model_name": "test-model"},
            data={"choices": [{
                "message": {
                    "role": "assistant", "content": "hello!"
                }
            }]},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="agent-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:03Z",
            name="classified_agent",
            attributes=[],
            category="agent",
            data={"final": "ignored — agent scope-end has no Branch B promotion"},
        ),
    ]
    trajectory = convert(events)

    sources = [s.source for s in trajectory.steps]
    # R1 root agent → no step. R2 user step (from llm input). R4 agent step (from llm output).
    # Branch A must NOT have fired (root is category="agent"); Branch B's predicate also
    # excludes category="agent" so the root scope-end produces no extra system/agent step.
    assert sources == ["user", "agent"], f"expected exmp02-shape user→agent, got {sources}"
    # The user step came from the LLM's input messages (R2), NOT from the root
    # scope-start's data. Verify the message content is the LLM input, not the
    # root data ("ignored — Branch A must not fire").
    assert trajectory.steps[0].message == "hi"
    assert "ignored" not in str(trajectory.steps[0].message)


# ---------------------------------------------------------------------------
# 260501-53t per-step model_name propagation tests
# ---------------------------------------------------------------------------


def _agent_with_single_llm_pair(
    *,
    llm_name: str,
    llm_category_profile: dict | None,
    llm_input: dict,
    llm_output: dict,
    data_schema: dict | None = None,
) -> list:
    """Minimal classified-agent root containing a single llm scope-pair.

    Mirrors the exmp02-style shape used by `test_classified_agent_root_unchanged`
    so the converter takes the standard R2/R4 path: R1 root agent → no step,
    R2 user step from llm input, R4 agent step from llm scope-end.
    """
    return [
        ScopeEvent(
            scope_category="start",
            uuid="agent-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:00Z",
            name="classified_agent",
            attributes=[],
            category="agent",
            data=None,
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-001",
            parent_uuid="agent-001",
            timestamp="2026-01-01T00:00:01Z",
            name=llm_name,
            attributes=[],
            category="llm",
            category_profile=llm_category_profile,
            data=llm_input,
            data_schema=data_schema,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-001",
            parent_uuid="agent-001",
            timestamp="2026-01-01T00:00:02Z",
            name=llm_name,
            attributes=[],
            category="llm",
            category_profile=llm_category_profile,
            data=llm_output,
            data_schema=data_schema,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="agent-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:03Z",
            name="classified_agent",
            attributes=[],
            category="agent",
            data=None,
        ),
    ]


def test_llm_step_emits_per_step_model_name_from_category_profile() -> None:
    """An LLM scope with category_profile['model_name'] populates step.model_name.

    Standard tier-2 path: producer classifies the inner scope as ``llm`` and
    declares the model identifier in ``category_profile.model_name``. The
    converter must propagate that value verbatim onto the agent step
    emitted from the LLM scope-end.
    """
    events = _agent_with_single_llm_pair(
        llm_name="some-display-name",
        llm_category_profile={"model_name": "gpt-4o-test"},
        llm_input={"messages": [{
            "role": "user", "content": "ping"
        }]},
        llm_output={"choices": [{
            "message": {
                "role": "assistant", "content": "pong"
            }
        }]},
    )
    trajectory = convert(events)

    # R2 user step + R4 agent step.
    sources = [s.source for s in trajectory.steps]
    assert sources == ["user", "agent"], f"expected user→agent, got {sources}"

    agent_step = trajectory.steps[1]
    assert agent_step.source == "agent"
    assert agent_step.llm_call_count == 1
    # category_profile.model_name wins over event.name.
    assert agent_step.model_name == "gpt-4o-test", (
        f"expected per-step model_name from category_profile, got {agent_step.model_name!r}")


def test_llm_step_falls_back_to_event_name_when_category_profile_absent() -> None:
    """When category_profile is None on an LLM scope, step.model_name falls back to event.name.

    Tier-1.5 case: producer marks the scope ``category='llm'`` but has no
    structured profile to declare. The converter still emits *something*
    identifying the call by falling back to the scope's display name.
    """
    events = _agent_with_single_llm_pair(
        llm_name="some-model",
        llm_category_profile=None,
        llm_input={"messages": [{
            "role": "user", "content": "ping"
        }]},
        llm_output={"choices": [{
            "message": {
                "role": "assistant", "content": "pong"
            }
        }]},
    )
    trajectory = convert(events)

    sources = [s.source for s in trajectory.steps]
    assert sources == ["user", "agent"], f"expected user→agent, got {sources}"

    agent_step = trajectory.steps[1]
    assert agent_step.source == "agent"
    assert agent_step.llm_call_count == 1
    # No category_profile → fall back to event.name.
    assert agent_step.model_name == "some-model", f"expected event.name fallback, got {agent_step.model_name!r}"


def test_heterogeneous_workflow_emits_distinct_model_names_per_step() -> None:
    """Three LLM scopes with three distinct category_profile.model_name values
    produce three steps each carrying their own model_name (exmp06 shape).

    This is the core motivation for per-step model_name: a router-style
    workflow where the orchestrator dispatches to specialist models. The
    root ``agent.model_name`` reflects the first/orchestrator model, but
    consumers must be able to tell which specialist actually produced
    each downstream step.
    """
    events = [
        ScopeEvent(
            scope_category="start",
            uuid="orchestrator-006",
            parent_uuid=None,
            timestamp="2026-01-06T00:00:00Z",
            name="multi_provider_router",
            attributes=[],
            category="agent",
            data=None,
        ),
        # llm-1: model-a
        ScopeEvent(
            scope_category="start",
            uuid="llm-006-a",
            parent_uuid="orchestrator-006",
            timestamp="2026-01-06T00:00:01Z",
            name="provider-a-display",
            attributes=[],
            category="llm",
            category_profile={"model_name": "model-a"},
            data={"messages": [{
                "role": "user", "content": "step A in"
            }]},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-006-a",
            parent_uuid="orchestrator-006",
            timestamp="2026-01-06T00:00:02Z",
            name="provider-a-display",
            attributes=[],
            category="llm",
            category_profile={"model_name": "model-a"},
            data={"choices": [{
                "message": {
                    "role": "assistant", "content": "step A out"
                }
            }]},
        ),
        # llm-2: model-b
        ScopeEvent(
            scope_category="start",
            uuid="llm-006-b",
            parent_uuid="orchestrator-006",
            timestamp="2026-01-06T00:00:03Z",
            name="provider-b-display",
            attributes=[],
            category="llm",
            category_profile={"model_name": "model-b"},
            data={"messages": [{
                "role": "user", "content": "step B in"
            }]},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-006-b",
            parent_uuid="orchestrator-006",
            timestamp="2026-01-06T00:00:04Z",
            name="provider-b-display",
            attributes=[],
            category="llm",
            category_profile={"model_name": "model-b"},
            data={"choices": [{
                "message": {
                    "role": "assistant", "content": "step B out"
                }
            }]},
        ),
        # llm-3: model-c
        ScopeEvent(
            scope_category="start",
            uuid="llm-006-c",
            parent_uuid="orchestrator-006",
            timestamp="2026-01-06T00:00:05Z",
            name="provider-c-display",
            attributes=[],
            category="llm",
            category_profile={"model_name": "model-c"},
            data={"messages": [{
                "role": "user", "content": "step C in"
            }]},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-006-c",
            parent_uuid="orchestrator-006",
            timestamp="2026-01-06T00:00:06Z",
            name="provider-c-display",
            attributes=[],
            category="llm",
            category_profile={"model_name": "model-c"},
            data={"choices": [{
                "message": {
                    "role": "assistant", "content": "step C out"
                }
            }]},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="orchestrator-006",
            parent_uuid=None,
            timestamp="2026-01-06T00:00:07Z",
            name="multi_provider_router",
            attributes=[],
            category="agent",
            data=None,
        ),
    ]
    trajectory = convert(events)

    # 3 LLM pairs → 3 user steps + 3 agent steps under the classified agent root.
    agent_steps = [s for s in trajectory.steps if s.source == "agent"]
    assert len(agent_steps) == 3, (
        f"expected 3 agent steps from 3 LLM scope-ends, got {len(agent_steps)}: {[s.source for s in trajectory.steps]}")

    # Each agent step carries its own per-step model_name; values are distinct.
    per_step_models = [s.model_name for s in agent_steps]
    assert set(per_step_models) == {
        "model-a", "model-b", "model-c"
    }, (f"expected {{model-a, model-b, model-c}} distinct per-step model_names, got {per_step_models}")

    # Root agent.model_name picks up the FIRST LLM scope-end's profile (R-tag
    # at converter line ~985-992): unchanged by this task. Sanity check.
    assert trajectory.agent.model_name == "model-a", (
        f"expected root agent.model_name to remain first-LLM 'model-a', got {trajectory.agent.model_name!r}")


def test_no_llm_orchestrator_step_has_no_model_name() -> None:
    """The deterministic-dispatch path (R13, llm_call_count=0) does NOT set model_name.

    A ``function`` scope that contains a ``tool`` scope-end (but no LLM
    scope) is a deterministic dispatcher per R13: the agent step is
    emitted with ``llm_call_count=0`` and synthesized tool_calls. Per spec,
    ``model_name`` on a non-LLM step is meaningless and MUST remain None.
    """
    events = [
        ScopeEvent(
            scope_category="start",
            uuid="agent-013",
            parent_uuid=None,
            timestamp="2026-01-13T00:00:00Z",
            name="dispatch_agent",
            attributes=[],
            category="agent",
            data=None,
        ),
        # function scope wrapping a tool scope — triggers R13 on the function end.
        ScopeEvent(
            scope_category="start",
            uuid="fn-013",
            parent_uuid="agent-013",
            timestamp="2026-01-13T00:00:01Z",
            name="dispatch_function",
            attributes=[],
            category="function",
            data=None,
        ),
        ScopeEvent(
            scope_category="start",
            uuid="tool-013",
            parent_uuid="fn-013",
            timestamp="2026-01-13T00:00:02Z",
            name="lookup_tool",
            attributes=[],
            category="tool",
            category_profile={"tool_call_id": "tc-013-1"},
            data={"query": "rate"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="tool-013",
            parent_uuid="fn-013",
            timestamp="2026-01-13T00:00:03Z",
            name="lookup_tool",
            attributes=[],
            category="tool",
            category_profile={"tool_call_id": "tc-013-1"},
            data={"result": "0.05"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="fn-013",
            parent_uuid="agent-013",
            timestamp="2026-01-13T00:00:04Z",
            name="dispatch_function",
            attributes=[],
            category="function",
            data=None,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="agent-013",
            parent_uuid=None,
            timestamp="2026-01-13T00:00:05Z",
            name="dispatch_agent",
            attributes=[],
            category="agent",
            data=None,
        ),
    ]
    trajectory = convert(events)

    # Locate the R13 dispatch step (source='agent', llm_call_count=0).
    dispatch_steps = [s for s in trajectory.steps if s.source == "agent" and s.llm_call_count == 0]
    assert len(dispatch_steps) == 1, (
        f"expected exactly one R13 dispatch step (llm_call_count=0), "
        f"got {len(dispatch_steps)} from sources={[s.source for s in trajectory.steps]} "
        f"llm_call_counts={[s.llm_call_count for s in trajectory.steps]}"
    )

    dispatch_step = dispatch_steps[0]
    # Per spec: no model_name on deterministic dispatch.
    assert dispatch_step.model_name is None, (
        f"expected R13 dispatch step to have model_name=None, got {dispatch_step.model_name!r}")
    # Sanity: the synthesized tool_call carries the tool_call_id so we know
    # we hit the R13 branch (not some other agent-step path).
    assert dispatch_step.tool_calls is not None
    assert len(dispatch_step.tool_calls) == 1
    assert dispatch_step.tool_calls[0].tool_call_id == "tc-013-1"


if __name__ == "__main__":
    test_tier1_produces_nonempty_trajectory()
    test_tier1_agent_name_falls_back_to_root_scope()
    test_tier1_preserves_opaque_payloads()
    test_tier1_preserves_ancestry_and_invocation_timing()
    test_tier1_root_promotes_raw_query_to_user_step()
    test_tier1_root_promotes_raw_result_to_agent_step()
    test_tier1_root_dict_data_serializes_as_json()
    test_tier1_root_empty_data_emits_no_boundary_steps()
    test_tier1_inner_scopes_remain_system_steps()
    test_classified_agent_root_unchanged()
    test_llm_step_emits_per_step_model_name_from_category_profile()
    test_llm_step_falls_back_to_event_name_when_category_profile_absent()
    test_heterogeneous_workflow_emits_distinct_model_names_per_step()
    test_no_llm_orchestrator_step_has_no_model_name()
    print("All tier-1 conversion tests passed.")
