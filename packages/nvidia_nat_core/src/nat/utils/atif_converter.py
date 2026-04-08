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
"""Convert NAT IntermediateStep traces to the Agent Trajectory Interchange Format (ATIF).

ATIF is a standardized JSON format for logging the complete interaction history
of autonomous LLM agents. Reference: https://github.com/laude-institute/harbor

This module provides:
- Conversion helpers built on shared ATIF v1.6 models
- `IntermediateStepToATIFConverter` for batch conversion
- `ATIFStreamConverter` for incremental / streaming conversion
"""

from __future__ import annotations

__all__ = ["ATIFStreamConverter", "IntermediateStepToATIFConverter"]

import datetime
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from nat.atif import ATIFAgentConfig
from nat.atif import AtifAncestry
from nat.atif import ATIFFinalMetrics
from nat.atif import AtifInvocationInfo
from nat.atif import ATIFObservation
from nat.atif import ATIFObservationResult
from nat.atif import ATIFStep
from nat.atif import AtifStepExtra
from nat.atif import ATIFStepMetrics
from nat.atif import ATIFToolCall
from nat.atif import ATIFTrajectory
from nat.atif import SubagentTrajectoryRef
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepState
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import TraceMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _epoch_to_iso(epoch: float) -> str:
    """Convert a Unix epoch timestamp to an ISO 8601 string."""
    return datetime.datetime.fromtimestamp(epoch, tz=datetime.UTC).isoformat()


def _iso_to_epoch(timestamp: str) -> float:
    """Convert an ISO 8601 timestamp to Unix epoch seconds."""
    return datetime.datetime.fromisoformat(timestamp).timestamp()


def _extract_tool_definitions(step: IntermediateStep) -> list[dict[str, Any]] | None:
    """Extract OpenAI-style tool definitions from an IntermediateStep's metadata."""
    if not isinstance(step.metadata, TraceMetadata):
        return None
    schemas = step.metadata.tools_schema
    if not schemas:
        return None
    return [s.model_dump(by_alias=True) for s in schemas]


def _extract_metrics(step: IntermediateStep) -> ATIFStepMetrics | None:
    """Build ATIF step metrics from a NAT IntermediateStep's usage_info."""
    usage = step.usage_info
    if usage is None:
        return None
    tu = usage.token_usage
    if tu.prompt_tokens == 0 and tu.completion_tokens == 0 and tu.total_tokens == 0:
        return None
    extra: dict[str, Any] = {}
    if tu.reasoning_tokens:
        extra["reasoning_tokens"] = tu.reasoning_tokens
    return ATIFStepMetrics(
        prompt_tokens=tu.prompt_tokens or None,
        completion_tokens=tu.completion_tokens or None,
        cached_tokens=tu.cached_tokens or None,
        extra=extra or None,
    )


def _safe_str(value: Any) -> str:
    """Coerce a value to a string, returning empty string for None."""
    if value is None:
        return ""
    return str(value)


def _extract_user_input(value: Any) -> str:
    """Extract the user-facing input text from a workflow start payload.

    The ``data.input`` on a ``WORKFLOW_START`` step may be a raw string, a
    Pydantic model (for example, ``ChatRequestOrMessage``), or a dict. This helper
    tries to pull out the meaningful text.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    obj = value
    if hasattr(value, "model_dump"):
        obj = value.model_dump()
    if isinstance(obj, dict):
        if obj.get("input_message"):
            return str(obj["input_message"])
        msgs = obj.get("messages")
        if msgs and isinstance(msgs, list):
            last_user = ""
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    last_user = m.get("content", "")
            if last_user:
                return str(last_user)
    return str(value)


def _atif_ancestry_from_ist(ist: IntermediateStep) -> AtifAncestry:
    """Build typed ATIF ancestry metadata from an IntermediateStep."""
    return AtifAncestry(
        function_id=ist.function_ancestry.function_id,
        function_name=ist.function_ancestry.function_name,
        parent_id=ist.function_ancestry.parent_id,
        parent_name=ist.function_ancestry.parent_name,
    )


def _atif_invocation_from_ist(ist: IntermediateStep, *, invocation_id: str | None = None) -> AtifInvocationInfo:
    """Build typed ATIF invocation timing metadata from an IntermediateStep."""
    start_ts = ist.payload.span_event_timestamp
    end_ts = ist.event_timestamp if start_ts is not None else None
    return AtifInvocationInfo(
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        invocation_id=invocation_id,
        status="completed",
        framework=ist.payload.framework.value if ist.payload.framework is not None else None,
    )


def _atif_step_extra_model_from_ist(ist: IntermediateStep) -> AtifStepExtra:
    """Build typed ATIF step extra model from an IntermediateStep."""
    return AtifStepExtra(
        ancestry=_atif_ancestry_from_ist(ist),
        invocation=_atif_invocation_from_ist(ist),
    )


def _parse_tool_arguments(raw_input: Any) -> dict[str, Any]:
    """Best-effort extraction of tool arguments as a dict."""
    if isinstance(raw_input, dict):
        return raw_input
    if isinstance(raw_input, str):
        import ast
        import json

        try:
            parsed = json.loads(raw_input)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        try:
            parsed = ast.literal_eval(raw_input)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass

        return {"input": raw_input} if raw_input else {}
    if raw_input is not None:
        return {"input": str(raw_input)}
    return {}


def _extract_subagent_delegation_flag(metadata: Any) -> bool:
    """Extract optional subagent delegation flag from metadata payload."""
    if isinstance(metadata, TraceMetadata):
        provided = metadata.provided_metadata
        if isinstance(provided, dict):
            return bool(provided.get("is_subagent_delegation", provided.get("subagent_delegation", False)))
        return False
    if isinstance(metadata, dict):
        return bool(metadata.get("is_subagent_delegation", metadata.get("subagent_delegation", False)))
    return False


# ---------------------------------------------------------------------------
# Internal accumulator
# ---------------------------------------------------------------------------


@dataclass
class _ObservedInvocation:
    """One observed invocation within an agent turn."""

    order_key: float
    tool_call: ATIFToolCall
    tool_output: str
    ancestry: AtifAncestry
    invocation: AtifInvocationInfo
    is_subagent_delegation: bool


@dataclass
class _ExecutionStructure:
    """Pass-1 execution structure extracted from IST."""

    root_events: list[IntermediateStep]
    child_events_by_session: dict[str, list[IntermediateStep]]
    subagent_ref_by_call_id: dict[str, SubagentTrajectoryRef]


class _PendingAgentTurn:
    """Accumulator for an in-progress ATIF agent turn."""

    def __init__(self, message: str, timestamp: float, model_name: str | None, metrics: ATIFStepMetrics | None):
        self.message = message
        self.timestamp = timestamp
        self.model_name = model_name
        self.metrics = metrics
        self.ancestry: AtifAncestry | None = None
        self.invocation: AtifInvocationInfo | None = None
        self.extra: dict[str, Any] = {}
        self.observed_invocations: list[_ObservedInvocation] = []


def _record_observed_invocation(pending: _PendingAgentTurn, ist: IntermediateStep, *, start_flag: bool = False) -> None:
    """Record an observed invocation as a tool_call + observation pair."""
    tool_name = ist.name or "unknown_tool"
    if tool_name == "<workflow>":
        # Suppress synthetic workflow wrapper calls from observed tool invocations.
        return
    tool_input: dict[str, Any] = {}
    tool_output = ""
    if ist.data:
        tool_input = _parse_tool_arguments(ist.data.input)
        tool_output = _safe_str(ist.data.output)
    call_id = f"call_{ist.UUID}"
    is_subagent_delegation = _extract_subagent_delegation_flag(ist.metadata) or start_flag
    pending.observed_invocations.append(
        _ObservedInvocation(
            order_key=ist.payload.span_event_timestamp or ist.event_timestamp,
            tool_call=ATIFToolCall(tool_call_id=call_id, function_name=tool_name, arguments=tool_input),
            tool_output=tool_output,
            ancestry=_atif_ancestry_from_ist(ist),
            invocation=_atif_invocation_from_ist(ist, invocation_id=call_id),
            is_subagent_delegation=is_subagent_delegation,
        ))


def _build_flat_observation_rows(
        observed: list[_ObservedInvocation],
        *,
        subagent_ref_by_call_id: dict[str, SubagentTrajectoryRef] | None = None) -> list[ATIFObservationResult]:
    """Build observation rows and attach explicit subagent refs when available."""
    results = [
        ATIFObservationResult(source_call_id=obs.tool_call.tool_call_id, content=obs.tool_output) for obs in observed
    ]

    if not subagent_ref_by_call_id:
        return results

    for i, obs in enumerate(observed):
        ref = subagent_ref_by_call_id.get(obs.tool_call.tool_call_id)
        if ref is not None:
            results[i].subagent_trajectory_ref = [ref]

    return results


def _pass2_project_context_to_steps(
    events: list[IntermediateStep],
    *,
    step_id_start: int,
    include_workflow_start_user_step: bool,
    subagent_ref_by_call_id: dict[str, SubagentTrajectoryRef] | None = None,
) -> tuple[list[ATIFStep], int, int, int, int]:
    """Pass-2 projection for one context from IST events to ATIF steps."""
    atif_steps: list[ATIFStep] = []
    step_id = step_id_start
    pending: _PendingAgentTurn | None = None
    total_prompt = 0
    total_completion = 0
    total_cached = 0

    def _flush_pending() -> None:
        nonlocal step_id, pending
        if pending is None:
            return
        sorted_invocations = sorted(pending.observed_invocations, key=lambda i: i.order_key)
        tool_calls = [obs.tool_call for obs in sorted_invocations] or None
        observations = _build_flat_observation_rows(
            sorted_invocations,
            subagent_ref_by_call_id=subagent_ref_by_call_id,
        )
        observation = ATIFObservation(results=observations) if observations else None
        tool_ancestry = [obs.ancestry for obs in sorted_invocations]
        tool_invocations = [obs.invocation for obs in sorted_invocations] or None
        if pending.ancestry is None:
            raise ValueError("Pending ATIF step is missing required ancestry metadata")
        step_extra = AtifStepExtra(
            ancestry=pending.ancestry,
            invocation=pending.invocation,
            tool_ancestry=tool_ancestry,
            tool_invocations=tool_invocations,
            **pending.extra,
        )
        atif_steps.append(
            ATIFStep(
                step_id=step_id,
                source="agent",
                message=pending.message,
                timestamp=_epoch_to_iso(pending.timestamp),
                model_name=pending.model_name,
                tool_calls=tool_calls,
                observation=observation,
                metrics=pending.metrics,
                extra=step_extra.model_dump(exclude_none=True),
            ))
        step_id += 1
        pending = None

    for ist in events:
        event_type = ist.event_type
        state = ist.event_state

        if include_workflow_start_user_step and event_type == IntermediateStepType.WORKFLOW_START:
            user_input = ""
            if ist.data and ist.data.input is not None:
                user_input = _extract_user_input(ist.data.input)
            step_extra = _atif_step_extra_model_from_ist(ist)
            extra = step_extra.model_dump(exclude_none=True)
            atif_steps.append(
                ATIFStep(
                    step_id=step_id,
                    source="user",
                    message=user_input,
                    timestamp=_epoch_to_iso(ist.event_timestamp),
                    extra=extra or None,
                ))
            step_id += 1
            continue

        if event_type == IntermediateStepType.LLM_END:
            _flush_pending()
            llm_output = ""
            if ist.data and ist.data.output is not None:
                llm_output = _safe_str(ist.data.output)
            metrics = _extract_metrics(ist)
            if metrics:
                total_prompt += metrics.prompt_tokens or 0
                total_completion += metrics.completion_tokens or 0
                total_cached += metrics.cached_tokens or 0
            pending = _PendingAgentTurn(
                message=llm_output,
                timestamp=ist.event_timestamp,
                model_name=ist.name,
                metrics=metrics,
            )
            pending.ancestry = _atif_ancestry_from_ist(ist)
            pending.invocation = _atif_invocation_from_ist(ist)
            continue

        if event_type in (IntermediateStepType.TOOL_END, IntermediateStepType.FUNCTION_END):
            if pending is None:
                pending = _PendingAgentTurn(
                    message="",
                    timestamp=ist.event_timestamp,
                    model_name=None,
                    metrics=None,
                )
                pending.ancestry = _atif_ancestry_from_ist(ist)
                pending.invocation = _atif_invocation_from_ist(ist)
            _record_observed_invocation(pending, ist)
            continue

        if event_type == IntermediateStepType.WORKFLOW_END:
            _flush_pending()
            final_output = ""
            if ist.data and ist.data.output is not None:
                final_output = _safe_str(ist.data.output)
            last_agent_msg = ""
            last_agent_ts: float | None = None
            for s in reversed(atif_steps):
                if s.source == "agent":
                    last_agent_msg = str(s.message)
                    last_agent_ts = _iso_to_epoch(s.timestamp) if s.timestamp else None
                    break
            should_emit_terminal_step = bool(final_output) and (final_output != last_agent_msg or
                                                                (last_agent_ts is not None
                                                                 and ist.event_timestamp > last_agent_ts))
            if should_emit_terminal_step:
                step_extra = _atif_step_extra_model_from_ist(ist)
                extra = step_extra.model_dump(exclude_none=True)
                atif_steps.append(
                    ATIFStep(
                        step_id=step_id,
                        source="agent",
                        message=final_output,
                        timestamp=_epoch_to_iso(ist.event_timestamp),
                        extra=extra or None,
                    ))
                step_id += 1
            continue

        if state == IntermediateStepState.START:
            continue
        if event_type in (IntermediateStepType.LLM_NEW_TOKEN, IntermediateStepType.SPAN_CHUNK):
            continue
        if state == IntermediateStepState.END:
            continue

    _flush_pending()
    return atif_steps, step_id, total_prompt, total_completion, total_cached


def _pass1_build_execution_structure(sorted_steps: list[IntermediateStep], *, session_id: str) -> _ExecutionStructure:
    """Build root and child context ownership from IST events."""
    delegation_flags_by_uuid: dict[str, bool] = {}
    for ist in sorted_steps:
        if ist.event_state == IntermediateStepState.START and _extract_subagent_delegation_flag(ist.metadata):
            delegation_flags_by_uuid[ist.UUID] = True

    end_events = [s for s in sorted_steps if s.event_state == IntermediateStepState.END]
    children_by_parent: dict[str, list[IntermediateStep]] = {}
    for e in end_events:
        pid = e.function_ancestry.parent_id
        if pid:
            children_by_parent.setdefault(pid, []).append(e)

    wrapper_events: list[IntermediateStep] = []
    for e in end_events:
        if e.event_type not in (IntermediateStepType.TOOL_END, IntermediateStepType.FUNCTION_END):
            continue
        if _extract_subagent_delegation_flag(e.metadata) or delegation_flags_by_uuid.get(e.UUID, False):
            wrapper_events.append(e)

    child_session_by_wrapper_call_id: dict[str, str] = {}
    child_events_by_session: dict[str, list[IntermediateStep]] = {}
    delegated_function_ids: set[str] = set()

    for wrapper in wrapper_events:
        wrapper_call_id = f"call_{wrapper.UUID}"
        wrapper_fn_id = wrapper.function_ancestry.function_id
        direct_children = children_by_parent.get(wrapper_fn_id, [])
        preferred_roots = [c for c in direct_children if c.function_ancestry.function_name == (wrapper.name or "")
                           ] or direct_children
        if not preferred_roots:
            continue
        child_root = sorted(preferred_roots, key=lambda s: s.event_timestamp)[0]
        child_root_fn_id = child_root.function_ancestry.function_id

        subtree_ids: set[str] = set()
        frontier = [child_root_fn_id]
        while frontier:
            node = frontier.pop()
            if node in subtree_ids:
                continue
            subtree_ids.add(node)
            for child in children_by_parent.get(node, []):
                frontier.append(child.function_ancestry.function_id)

        child_events = [
            e for e in end_events if e.function_ancestry.function_id in subtree_ids and e.UUID != wrapper.UUID
        ]
        if not child_events:
            continue
        child_session_id = f"{session_id}:{wrapper_call_id}"
        child_session_by_wrapper_call_id[wrapper_call_id] = child_session_id
        child_events_by_session[child_session_id] = sorted(child_events, key=lambda s: s.event_timestamp)
        delegated_function_ids.update(subtree_ids)

    root_events = [
        e for e in sorted_steps
        if (e.event_type in {IntermediateStepType.WORKFLOW_START, IntermediateStepType.WORKFLOW_END}
            or e.function_ancestry.function_id not in delegated_function_ids)
    ]
    subagent_ref_by_call_id = {
        call_id: SubagentTrajectoryRef(session_id=child_sid)
        for call_id, child_sid in child_session_by_wrapper_call_id.items()
    }
    return _ExecutionStructure(
        root_events=root_events,
        child_events_by_session=child_events_by_session,
        subagent_ref_by_call_id=subagent_ref_by_call_id,
    )


# ---------------------------------------------------------------------------
# Batch converter
# ---------------------------------------------------------------------------


class IntermediateStepToATIFConverter:
    """Convert a complete list of NAT IntermediateSteps to an ATIF trajectory."""

    def __init__(self, *, allow_implicit_subagent_delegation: bool = False) -> None:
        # Legacy option retained for API compatibility; clean converter ignores implicit delegation.
        self._allow_implicit_subagent_delegation = allow_implicit_subagent_delegation

    def convert(
        self,
        steps: list[IntermediateStep],
        *,
        session_id: str | None = None,
        agent_name: str | None = None,
    ) -> ATIFTrajectory:
        """Convert a list of IntermediateSteps to an ATIF trajectory."""
        trajectory_session_id = session_id or str(uuid.uuid4())
        if not steps:
            return ATIFTrajectory(
                session_id=trajectory_session_id,
                agent=ATIFAgentConfig(name=agent_name or "nat-agent", version="0.0.0"),
            )

        sorted_steps = sorted(steps, key=lambda s: s.event_timestamp)
        agent_config = ATIFAgentConfig(name=agent_name or "nat-agent", version="0.0.0")

        if agent_name is None:
            for ist in sorted_steps:
                if ist.event_type == IntermediateStepType.WORKFLOW_START:
                    fn_name = ist.function_ancestry.function_name
                    if fn_name and fn_name != "root":
                        agent_config.name = fn_name
                        break

        for ist in sorted_steps:
            if ist.event_type == IntermediateStepType.LLM_END:
                if ist.name and not agent_config.model_name:
                    agent_config.model_name = ist.name
                defs = _extract_tool_definitions(ist)
                if defs and not agent_config.tool_definitions:
                    agent_config.tool_definitions = defs

        execution_structure = _pass1_build_execution_structure(sorted_steps, session_id=trajectory_session_id)
        atif_steps, _, total_prompt, total_completion, total_cached = _pass2_project_context_to_steps(
            execution_structure.root_events,
            step_id_start=1,
            include_workflow_start_user_step=True,
            subagent_ref_by_call_id=execution_structure.subagent_ref_by_call_id,
        )

        child_trajectories: dict[str, dict[str, Any]] = {}
        for child_session_id, child_events in execution_structure.child_events_by_session.items():
            child_steps, _, _, _, _ = _pass2_project_context_to_steps(
                child_events,
                step_id_start=1,
                include_workflow_start_user_step=False,
                subagent_ref_by_call_id=None,
            )
            child_trajectory = ATIFTrajectory(
                session_id=child_session_id,
                agent=agent_config.model_copy(deep=True),
                steps=child_steps,
            )
            child_trajectories[child_session_id] = child_trajectory.model_dump(exclude_none=True, mode="json")

        final_metrics = None
        agent_step_count = sum(1 for s in atif_steps if s.source == "agent")
        if total_prompt or total_completion or total_cached or agent_step_count:
            final_metrics = ATIFFinalMetrics(
                total_prompt_tokens=total_prompt or None,
                total_completion_tokens=total_completion or None,
                total_cached_tokens=total_cached or None,
                total_steps=agent_step_count,
            )

        trajectory_extra = {"subagent_trajectories": child_trajectories} if child_trajectories else None
        return ATIFTrajectory(
            session_id=trajectory_session_id,
            agent=agent_config,
            steps=atif_steps,
            final_metrics=final_metrics,
            extra=trajectory_extra,
        )


# ---------------------------------------------------------------------------
# Stream converter
# ---------------------------------------------------------------------------


class ATIFStreamConverter:
    """Stateful converter that reuses the same two-pass conversion model.

    This stream adapter accumulates IST events and re-projects ATIF using the
    same batch converter logic on each push. It keeps behavior consistent
    between batch and stream conversion paths.
    """

    def __init__(self, agent_name: str = "nat-agent", *, allow_implicit_subagent_delegation: bool = False):
        self._session_id: str = str(uuid.uuid4())
        self._agent_name = agent_name
        self._buffered_events: list[IntermediateStep] = []
        self._last_projected_steps: list[ATIFStep] = []
        self._returned_step_count = 0
        self._latest_trajectory = ATIFTrajectory(
            session_id=self._session_id,
            agent=ATIFAgentConfig(name=agent_name, version="0.0.0"),
            steps=[],
        )
        self._batch_converter = IntermediateStepToATIFConverter(
            allow_implicit_subagent_delegation=allow_implicit_subagent_delegation)

    @property
    def agent_config(self) -> ATIFAgentConfig:
        """Current agent configuration based on latest projection."""
        return self._latest_trajectory.agent

    def push(self, ist: IntermediateStep) -> ATIFStep | None:
        """Add one IST event and return next newly visible ATIF step."""
        self._buffered_events.append(ist)
        self._latest_trajectory = self._batch_converter.convert(
            steps=self._buffered_events,
            session_id=self._session_id,
            agent_name=self._agent_name,
        )
        self._last_projected_steps = list(self._latest_trajectory.steps)
        if self._returned_step_count < len(self._last_projected_steps):
            step = self._last_projected_steps[self._returned_step_count]
            self._returned_step_count += 1
            return step
        return None

    def finalize(self) -> list[ATIFStep]:
        """Return all projected ATIF steps not yet returned by `push`."""
        remaining = self._last_projected_steps[self._returned_step_count:]
        self._returned_step_count = len(self._last_projected_steps)
        return remaining

    def get_trajectory(self) -> ATIFTrajectory:
        """Return trajectory projected from all buffered IST events."""
        return self._latest_trajectory
