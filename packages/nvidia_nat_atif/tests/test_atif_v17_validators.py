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
"""Direct validator tests for ATIF v1.7 model changes.

Covers the two model-level invariants introduced in v1.7 that are not
exercised by the ATOF→ATIF converter examples (none of which use
embedded subagents):

- ``SubagentTrajectoryRef`` MUST set at least one of ``trajectory_id``
  (embedded form) or ``trajectory_path`` (file-ref form). ``session_id``
  alone is informational and no longer a valid resolution key.
- Within a parent's ``Trajectory.subagent_trajectories`` array, every
  embedded subagent MUST set ``trajectory_id`` and the values MUST be
  unique.

Also pins the v1.7 type relaxation: ``Trajectory.session_id`` is now
``str | None`` with default ``None`` (was ``str`` with auto-UUID
factory), and ``SubagentTrajectoryRef.session_id`` is now
``str | None`` with default ``None`` (was required ``str``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nat.atif import Agent
from nat.atif import SubagentTrajectoryRef
from nat.atif import Trajectory

# ---------------------------------------------------------------------------
# SubagentTrajectoryRef: at-least-one-of(trajectory_id, trajectory_path)
# ---------------------------------------------------------------------------


def test_subagent_ref_accepts_trajectory_id_alone() -> None:
    """Embedded form: ``trajectory_id`` set, ``trajectory_path`` null."""
    ref = SubagentTrajectoryRef(trajectory_id="sub-001")
    assert ref.trajectory_id == "sub-001"
    assert ref.trajectory_path is None
    assert ref.session_id is None


def test_subagent_ref_accepts_trajectory_path_alone() -> None:
    """File-ref form: ``trajectory_path`` set, ``trajectory_id`` null.

    This is the pre-v1.7 back-compat path — v1.6 refs that already set
    ``trajectory_path`` continue to validate.
    """
    ref = SubagentTrajectoryRef(trajectory_path="s3://bucket/sub-001.json")
    assert ref.trajectory_id is None
    assert ref.trajectory_path == "s3://bucket/sub-001.json"


def test_subagent_ref_accepts_both_keys() -> None:
    """Setting both ``trajectory_id`` AND ``trajectory_path`` is permitted —
    e.g. an embedded ref that also records its archival path for debug."""
    ref = SubagentTrajectoryRef(
        trajectory_id="sub-001",
        trajectory_path="s3://bucket/sub-001.json",
    )
    assert ref.trajectory_id == "sub-001"
    assert ref.trajectory_path == "s3://bucket/sub-001.json"


def test_subagent_ref_accepts_session_id_as_informational() -> None:
    """``session_id`` MAY accompany an otherwise-resolvable ref as
    informational metadata (run-scoped breadcrumb)."""
    ref = SubagentTrajectoryRef(
        trajectory_id="sub-001",
        session_id="run-alpha",
    )
    assert ref.trajectory_id == "sub-001"
    assert ref.session_id == "run-alpha"


def test_subagent_ref_rejects_session_id_alone() -> None:
    """v1.7 BREAKING: a ref of shape ``{"session_id": "..."}`` (no
    ``trajectory_id`` and no ``trajectory_path``) no longer validates.
    ``session_id`` is informational, not a resolution key."""
    with pytest.raises(ValidationError) as exc_info:
        SubagentTrajectoryRef(session_id="run-alpha")
    assert "trajectory_id" in str(exc_info.value)
    assert "trajectory_path" in str(exc_info.value)


def test_subagent_ref_rejects_empty() -> None:
    """A bare ref with no fields set is unresolvable and rejected."""
    with pytest.raises(ValidationError) as exc_info:
        SubagentTrajectoryRef()
    assert "trajectory_id" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Trajectory.subagent_trajectories: trajectory_id required + unique
# ---------------------------------------------------------------------------


def _stub_trajectory(trajectory_id: str | None = None) -> Trajectory:
    """Helper: construct a minimal Trajectory with no steps."""
    return Trajectory(
        agent=Agent(name="t", version="1.0.0"),
        steps=[],
        trajectory_id=trajectory_id,
    )


def test_trajectory_standalone_omits_trajectory_id_ok() -> None:
    """``trajectory_id`` is OPTIONAL on standalone trajectories. Constructing
    one with no ``subagent_trajectories`` and no ``trajectory_id`` is fine."""
    traj = _stub_trajectory(trajectory_id=None)
    assert traj.trajectory_id is None


def test_trajectory_subagents_must_have_trajectory_id() -> None:
    """An embedded subagent (entry in parent's ``subagent_trajectories``)
    MUST have ``trajectory_id`` set."""
    parent_subagents = [_stub_trajectory(trajectory_id=None)]
    with pytest.raises(ValidationError) as exc_info:
        Trajectory(
            agent=Agent(name="parent", version="1.0.0"),
            steps=[],
            subagent_trajectories=parent_subagents,
        )
    assert "trajectory_id" in str(exc_info.value)
    assert "REQUIRED" in str(exc_info.value)


def test_trajectory_subagents_trajectory_ids_must_be_unique() -> None:
    """Within a parent's ``subagent_trajectories[]``, ``trajectory_id``s
    MUST be unique (``session_id``s, by contrast, MAY collide across
    siblings)."""
    duplicates = [
        _stub_trajectory(trajectory_id="sub-A"),
        _stub_trajectory(trajectory_id="sub-A"),
    ]
    with pytest.raises(ValidationError) as exc_info:
        Trajectory(
            agent=Agent(name="parent", version="1.0.0"),
            steps=[],
            subagent_trajectories=duplicates,
        )
    assert "duplicate" in str(exc_info.value)
    assert "sub-A" in str(exc_info.value)


def test_trajectory_subagents_unique_trajectory_ids_ok() -> None:
    """Two embedded subagents with distinct ``trajectory_id``s validate
    even when they share a ``session_id`` (run-scoped, MAY collide)."""
    siblings = [
        Trajectory(
            agent=Agent(name="A", version="1.0.0"),
            steps=[],
            trajectory_id="sub-A",
            session_id="shared-run",
        ),
        Trajectory(
            agent=Agent(name="B", version="1.0.0"),
            steps=[],
            trajectory_id="sub-B",
            session_id="shared-run",
        ),
    ]
    parent = Trajectory(
        agent=Agent(name="parent", version="1.0.0"),
        steps=[],
        subagent_trajectories=siblings,
    )
    assert len(parent.subagent_trajectories) == 2
    assert {t.trajectory_id for t in parent.subagent_trajectories} == {"sub-A", "sub-B"}


# ---------------------------------------------------------------------------
# Trajectory.session_id: type relaxation (no auto-UUID default)
# ---------------------------------------------------------------------------


def test_trajectory_session_id_defaults_to_none() -> None:
    """v1.7: ``session_id`` defaults to ``None`` (was auto-UUID factory in
    pre-v1.7 NAT). Direct Python construction without an explicit
    ``session_id`` produces ``None`` rather than a fresh random UUID."""
    traj = Trajectory(agent=Agent(name="t", version="1.0.0"), steps=[])
    assert traj.session_id is None


def test_trajectory_session_id_accepts_explicit_value() -> None:
    """Explicit ``session_id`` is preserved verbatim."""
    traj = Trajectory(
        agent=Agent(name="t", version="1.0.0"),
        steps=[],
        session_id="run-2026-04-30",
    )
    assert traj.session_id == "run-2026-04-30"


# ---------------------------------------------------------------------------
# v1.7 spec example trajectory — round-trips cleanly through the model
# ---------------------------------------------------------------------------


def test_spec_example_trajectory_validates() -> None:
    """The canonical ATIF spec example (RFC 0001 §IV — financial search) MUST
    validate against our Trajectory model with no rejections.

    This pins our model's compliance to the public spec: any v1.7 producer
    emitting a spec-conformant trajectory will be accepted by us. If the
    spec adds or relaxes a field and we miss it, this test will catch it.
    """
    spec_example = {
        "schema_version":
            "ATIF-v1.5",
        "session_id":
            "025B810F-B3A2-4C67-93C0-FE7A142A947A",
        "agent": {
            "name": "harbor-agent",
            "version": "1.0.0",
            "model_name": "gemini-2.5-flash",
            "tool_definitions": [{
                "type": "function",
                "function": {
                    "name": "financial_search",
                    "description": "Search for financial data for a given stock ticker",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string", "description": "Stock ticker symbol"
                            },
                            "metric": {
                                "type": "string",
                                "description": "The financial metric to retrieve (e.g., price, volume)",
                            },
                        },
                        "required": ["ticker", "metric"],
                    },
                },
            }, ],
            "extra": {},
        },
        "notes": ("Initial test trajectory for financial data retrieval using a single-hop ReAct pattern, "
                  "focusing on multi-tool execution in Step 2."),
        "extra": {},
        "final_metrics": {
            "total_prompt_tokens": 1120,
            "total_completion_tokens": 124,
            "total_cached_tokens": 200,
            "total_cost_usd": 0.00078,
            "total_steps": 3,
            "extra": {},
        },
        "steps": [
            {
                "step_id": 1,
                "timestamp": "2025-10-11T10:30:00Z",
                "source": "user",
                "message": "What is the current trading price of Alphabet (GOOGL)?",
                "extra": {},
            },
            {
                "step_id": 2,
                "timestamp": "2025-10-11T10:30:02Z",
                "source": "agent",
                "model_name": "gemini-2.5-flash",
                "reasoning_effort": "medium",
                "message": "I will search for the current trading price and volume for GOOGL.",
                "reasoning_content":
                    ("The request requires two data points: the current stock price and the latest volume data. "
                     "I will execute two simultaneous tool calls to retrieve this information in a single step."),
                "tool_calls": [
                    {
                        "tool_call_id": "call_price_1",
                        "function_name": "financial_search",
                        "arguments": {
                            "ticker": "GOOGL", "metric": "price"
                        },
                    },
                    {
                        "tool_call_id": "call_volume_2",
                        "function_name": "financial_search",
                        "arguments": {
                            "ticker": "GOOGL", "metric": "volume"
                        },
                    },
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "call_price_1",
                            "content": "GOOGL is currently trading at $185.35 (Close: 10/11/2025)",
                        },
                        {
                            "source_call_id": "call_volume_2",
                            "content": "GOOGL volume: 1.5M shares traded.",
                        },
                    ],
                },
                "metrics": {
                    "prompt_tokens": 520,
                    "completion_tokens": 80,
                    "cached_tokens": 200,
                    "cost_usd": 0.00045,
                },
            },
            {
                "step_id": 3,
                "timestamp": "2025-10-11T10:30:05Z",
                "source": "agent",
                "model_name": "gemini-2.5-flash",
                "reasoning_effort": "low",
                "message": ("As of October 11, 2025, Alphabet (GOOGL) is trading at $185.35 "
                            "with a volume of 1.5M shares traded."),
                "reasoning_content": ("The previous step retrieved all necessary data. I will now format this into a "
                                      "final conversational response for the user and terminate the task."),
                "metrics": {
                    "prompt_tokens": 600,
                    "completion_tokens": 44,
                    "cost_usd": 0.00033,
                    "extra": {
                        "reasoning_tokens": 12
                    },
                },
            },
        ],
    }

    traj = Trajectory.model_validate(spec_example)
    assert len(traj.steps) == 3
    assert traj.agent.name == "harbor-agent"
    assert traj.steps[1].tool_calls is not None
    assert len(traj.steps[1].tool_calls) == 2
    assert traj.steps[1].observation is not None
    assert len(traj.steps[1].observation.results) == 2


def test_observation_result_extra_field_v17() -> None:
    """v1.7 added `extra` to ObservationResult. The model must accept it
    and round-trip it cleanly.

    Pins the spec example from §ObservationResultSchema (the
    `retrieval_score` / `source_doc_id` example).
    """
    from nat.atif import ObservationResult

    result = ObservationResult.model_validate({
        "source_call_id": "call_search_001",
        "content": "NVIDIA announces new GPU architecture...",
        "extra": {
            "retrieval_score": 0.92, "source_doc_id": "doc-4821"
        },
    })
    assert result.source_call_id == "call_search_001"
    assert result.extra == {"retrieval_score": 0.92, "source_doc_id": "doc-4821"}

    # Re-dump preserves extra
    dumped = result.model_dump(exclude_none=True)
    assert dumped["extra"] == {"retrieval_score": 0.92, "source_doc_id": "doc-4821"}


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
