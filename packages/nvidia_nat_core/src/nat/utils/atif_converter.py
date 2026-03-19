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

from nat.data_models.atif import ATIFAgentConfig
from nat.data_models.atif import AtifAncestry
from nat.data_models.atif import ATIFFinalMetrics
from nat.data_models.atif import ATIFObservation
from nat.data_models.atif import ATIFObservationResult
from nat.data_models.atif import ATIFStep
from nat.data_models.atif import AtifStepExtra
from nat.data_models.atif import ATIFStepMetrics
from nat.data_models.atif import ATIFToolCall
from nat.data_models.atif import ATIFTrajectory
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepCategory
from nat.data_models.intermediate_step import IntermediateStepState
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import TraceMetadata
from nat.data_models.invocation_node import InvocationNode

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
        function_ancestry=ist.function_ancestry,
        span_event_timestamp=ist.payload.span_event_timestamp,
        framework=ist.payload.framework.value if ist.payload.framework is not None else None,
    )


def _atif_step_extra_model_from_ist(ist: IntermediateStep) -> AtifStepExtra:
    """Build typed ATIF step extra model from an IntermediateStep."""
    return AtifStepExtra(ancestry=_atif_ancestry_from_ist(ist))


def _index_invocation_node(index: dict[str, InvocationNode], node: InvocationNode) -> None:
    """Track the latest seen invocation node by `function_id`."""
    index[node.function_id] = node


def _build_ancestry_path(node: InvocationNode, index: dict[str, InvocationNode]) -> list[InvocationNode]:
    """Build best-effort root-to-leaf ancestry path for a node.

    The builder follows parent links through previously seen invocation nodes.
    If a parent node is not available in the index, it inserts a synthetic
    parent placeholder using `parent_id`/`parent_name` from the child.
    """
    path_reverse: list[InvocationNode] = []
    visited: set[str] = set()
    current: InvocationNode | None = node

    while current is not None and current.function_id not in visited:
        path_reverse.append(current)
        visited.add(current.function_id)

        if current.parent_id is None:
            break

        parent = index.get(current.parent_id)
        if parent is None:
            inferred_parent_name = current.parent_name or ("root" if current.parent_id == "root" else current.parent_id)
            path_reverse.append(InvocationNode(
                function_id=current.parent_id,
                function_name=inferred_parent_name,
            ))
            break

        current = parent

    return list(reversed(path_reverse))


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


# ---------------------------------------------------------------------------
# Internal accumulator
# ---------------------------------------------------------------------------


@dataclass
class _ObservedInvocation:
    """One observed invocation within an agent turn."""

    order_key: float
    tool_call: ATIFToolCall
    observation: ATIFObservationResult
    ancestry: AtifAncestry
    ancestry_path: list[InvocationNode]


class _PendingAgentTurn:
    """Accumulator for an in-progress ATIF agent turn."""

    def __init__(self, message: str, timestamp: float, model_name: str | None, metrics: ATIFStepMetrics | None):
        self.message = message
        self.timestamp = timestamp
        self.model_name = model_name
        self.metrics = metrics
        self.ancestry: AtifAncestry | None = None
        self.step_ancestry_path: list[InvocationNode] | None = None
        self.extra: dict[str, Any] = {}
        self.observed_invocations: list[_ObservedInvocation] = []


def _record_observed_invocation(pending: _PendingAgentTurn,
                                ist: IntermediateStep,
                                invocation_index: dict[str, InvocationNode]) -> None:
    """Record an observed invocation as a tool_call + observation pair."""
    tool_name = ist.name or "unknown_tool"
    tool_input: dict[str, Any] = {}
    tool_output = ""
    if ist.data:
        tool_input = _parse_tool_arguments(ist.data.input)
        tool_output = _safe_str(ist.data.output)
    call_id = f"call_{ist.UUID}"
    pending.observed_invocations.append(
        _ObservedInvocation(
            order_key=ist.payload.span_event_timestamp or ist.event_timestamp,
            tool_call=ATIFToolCall(tool_call_id=call_id, function_name=tool_name, arguments=tool_input),
            observation=ATIFObservationResult(source_call_id=call_id, content=tool_output),
            ancestry=_atif_ancestry_from_ist(ist),
            ancestry_path=_build_ancestry_path(ist.function_ancestry, invocation_index),
        ))


# ---------------------------------------------------------------------------
# Batch converter
# ---------------------------------------------------------------------------


class IntermediateStepToATIFConverter:
    """Convert a complete list of NAT IntermediateSteps to an ATIF trajectory."""

    def convert(
        self,
        steps: list[IntermediateStep],
        *,
        session_id: str | None = None,
        agent_name: str | None = None,
    ) -> ATIFTrajectory:
        """Convert a list of IntermediateSteps to an ATIF trajectory."""
        if not steps:
            return ATIFTrajectory(
                session_id=session_id or str(uuid.uuid4()),
                agent=ATIFAgentConfig(name=agent_name or "nat-agent", version="0.0.0"),
            )

        sorted_steps = sorted(steps, key=lambda s: s.event_timestamp)
        atif_steps: list[ATIFStep] = []
        step_id = 1

        agent_config = ATIFAgentConfig(name=agent_name or "nat-agent", version="0.0.0")
        tool_defs_captured = False
        pending: _PendingAgentTurn | None = None
        total_prompt = 0
        total_completion = 0
        total_cached = 0
        invocation_index: dict[str, InvocationNode] = {}

        def _flush_pending() -> None:
            nonlocal step_id, pending
            if pending is None:
                return
            sorted_invocations = sorted(pending.observed_invocations, key=lambda i: i.order_key)
            tool_calls = [obs.tool_call for obs in sorted_invocations] or None
            observations = [obs.observation for obs in sorted_invocations]
            observation = ATIFObservation(results=observations) if observations else None
            tool_ancestry = [obs.ancestry for obs in sorted_invocations]
            tool_ancestry_paths = [obs.ancestry_path for obs in sorted_invocations] or None
            if pending.ancestry is None:
                raise ValueError("Pending agent turn is missing required ATIF ancestry metadata")
            step_extra = AtifStepExtra(
                ancestry=pending.ancestry,
                tool_ancestry=tool_ancestry,
                step_ancestry_path=pending.step_ancestry_path,
                tool_ancestry_paths=tool_ancestry_paths,
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

        for ist in sorted_steps:
            _index_invocation_node(invocation_index, ist.function_ancestry)
            event_type = ist.event_type
            category = ist.event_category
            state = ist.event_state

            if event_type == IntermediateStepType.WORKFLOW_START:
                user_input = ""
                if ist.data and ist.data.input is not None:
                    user_input = _extract_user_input(ist.data.input)
                if agent_name is None:
                    fn_name = ist.function_ancestry.function_name
                    if fn_name and fn_name != "root":
                        agent_config.name = fn_name
                step_extra = _atif_step_extra_model_from_ist(ist)
                step_extra.step_ancestry_path = _build_ancestry_path(ist.function_ancestry, invocation_index)
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
                    step_extra.step_ancestry_path = _build_ancestry_path(ist.function_ancestry, invocation_index)
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
                if not tool_defs_captured:
                    defs = _extract_tool_definitions(ist)
                    if defs:
                        agent_config.tool_definitions = defs
                        tool_defs_captured = True
                if ist.name and not agent_config.model_name:
                    agent_config.model_name = ist.name
                pending = _PendingAgentTurn(
                    message=llm_output,
                    timestamp=ist.event_timestamp,
                    model_name=ist.name,
                    metrics=metrics,
                )
                pending.ancestry = _atif_ancestry_from_ist(ist)
                pending.step_ancestry_path = _build_ancestry_path(ist.function_ancestry, invocation_index)
                continue

            if event_type == IntermediateStepType.TOOL_END:
                if pending is not None:
                    _record_observed_invocation(pending, ist, invocation_index)
                else:
                    orphan_pending = _PendingAgentTurn(message="",
                                                       timestamp=ist.event_timestamp,
                                                       model_name=None,
                                                       metrics=None)
                    orphan_pending.ancestry = _atif_ancestry_from_ist(ist)
                    orphan_pending.step_ancestry_path = _build_ancestry_path(ist.function_ancestry, invocation_index)
                    _record_observed_invocation(orphan_pending, ist, invocation_index)
                    invocation = orphan_pending.observed_invocations[0]
                    step_extra = _atif_step_extra_model_from_ist(ist)
                    step_extra.step_ancestry_path = _build_ancestry_path(ist.function_ancestry, invocation_index)
                    step_extra.tool_ancestry_paths = [invocation.ancestry_path]
                    step_extra.tool_ancestry = [invocation.ancestry]
                    extra = step_extra.model_dump(exclude_none=True)
                    atif_steps.append(
                        ATIFStep(
                            step_id=step_id,
                            source="agent",
                            message="",
                            timestamp=_epoch_to_iso(ist.event_timestamp),
                            tool_calls=[invocation.tool_call],
                            observation=ATIFObservation(results=[invocation.observation]),
                            extra=extra or None,
                        ))
                    step_id += 1
                continue

            if event_type == IntermediateStepType.FUNCTION_END:
                if pending is not None:
                    _record_observed_invocation(pending, ist, invocation_index)
                continue

            if state == IntermediateStepState.START:
                continue
            if event_type == IntermediateStepType.LLM_NEW_TOKEN:
                continue
            if event_type == IntermediateStepType.SPAN_CHUNK:
                continue

            if state == IntermediateStepState.END and category not in (
                    IntermediateStepCategory.LLM,
                    IntermediateStepCategory.TOOL,
                    IntermediateStepCategory.WORKFLOW,
            ):
                continue

        _flush_pending()

        final_metrics = None
        agent_step_count = sum(1 for s in atif_steps if s.source == "agent")
        if total_prompt or total_completion or total_cached or agent_step_count:
            final_metrics = ATIFFinalMetrics(
                total_prompt_tokens=total_prompt or None,
                total_completion_tokens=total_completion or None,
                total_cached_tokens=total_cached or None,
                total_steps=agent_step_count,
            )

        return ATIFTrajectory(
            session_id=session_id or str(uuid.uuid4()),
            agent=agent_config,
            steps=atif_steps,
            final_metrics=final_metrics,
        )


# ---------------------------------------------------------------------------
# Stream converter
# ---------------------------------------------------------------------------


class ATIFStreamConverter:
    """Stateful converter that emits ATIF steps incrementally."""

    def __init__(self, agent_name: str = "nat-agent"):
        self._step_id: int = 1
        self._agent_config = ATIFAgentConfig(name=agent_name, version="0.0.0")
        self._tool_defs_captured = False
        self._pending: _PendingAgentTurn | None = None
        self._emitted_steps: list[ATIFStep] = []
        self._total_prompt = 0
        self._total_completion = 0
        self._total_cached = 0
        self._invocation_index: dict[str, InvocationNode] = {}

    @property
    def agent_config(self) -> ATIFAgentConfig:
        """Current agent configuration (populated as steps arrive)."""
        return self._agent_config

    def push(self, ist: IntermediateStep) -> ATIFStep | None:
        """Process one IntermediateStep and return a flushed ATIF step if available."""
        _index_invocation_node(self._invocation_index, ist.function_ancestry)
        event_type = ist.event_type
        category = ist.event_category
        state = ist.event_state

        if event_type == IntermediateStepType.WORKFLOW_START:
            user_input = ""
            if ist.data and ist.data.input is not None:
                user_input = _extract_user_input(ist.data.input)
            fn_name = ist.function_ancestry.function_name
            if fn_name and fn_name != "root":
                self._agent_config.name = fn_name
            step_extra = _atif_step_extra_model_from_ist(ist)
            step_extra.step_ancestry_path = _build_ancestry_path(ist.function_ancestry, self._invocation_index)
            extra = step_extra.model_dump(exclude_none=True)
            step = ATIFStep(
                step_id=self._step_id,
                source="user",
                message=user_input,
                timestamp=_epoch_to_iso(ist.event_timestamp),
                extra=extra or None,
            )
            self._step_id += 1
            self._emitted_steps.append(step)
            return step

        if event_type == IntermediateStepType.WORKFLOW_END:
            results: list[ATIFStep] = []
            flushed = self._flush_pending()
            if flushed:
                results.append(flushed)
            final_output = ""
            if ist.data and ist.data.output is not None:
                final_output = _safe_str(ist.data.output)
            last_agent_msg = ""
            last_agent_ts: float | None = None
            for s in reversed(self._emitted_steps):
                if s.source == "agent":
                    last_agent_msg = str(s.message)
                    last_agent_ts = _iso_to_epoch(s.timestamp) if s.timestamp else None
                    break
            should_emit_terminal_step = bool(final_output) and (final_output != last_agent_msg or
                                                                (last_agent_ts is not None
                                                                 and ist.event_timestamp > last_agent_ts))
            if should_emit_terminal_step:
                step_extra = _atif_step_extra_model_from_ist(ist)
                step_extra.step_ancestry_path = _build_ancestry_path(ist.function_ancestry, self._invocation_index)
                extra = step_extra.model_dump(exclude_none=True)
                final_step = ATIFStep(
                    step_id=self._step_id,
                    source="agent",
                    message=final_output,
                    timestamp=_epoch_to_iso(ist.event_timestamp),
                    extra=extra or None,
                )
                self._step_id += 1
                self._emitted_steps.append(final_step)
                results.append(final_step)
            return results[0] if results else None

        if event_type == IntermediateStepType.LLM_END:
            flushed = self._flush_pending()
            llm_output = ""
            if ist.data and ist.data.output is not None:
                llm_output = _safe_str(ist.data.output)
            metrics = _extract_metrics(ist)
            if metrics:
                self._total_prompt += metrics.prompt_tokens or 0
                self._total_completion += metrics.completion_tokens or 0
                self._total_cached += metrics.cached_tokens or 0
            if not self._tool_defs_captured:
                defs = _extract_tool_definitions(ist)
                if defs:
                    self._agent_config.tool_definitions = defs
                    self._tool_defs_captured = True
            if ist.name and not self._agent_config.model_name:
                self._agent_config.model_name = ist.name
            self._pending = _PendingAgentTurn(
                message=llm_output,
                timestamp=ist.event_timestamp,
                model_name=ist.name,
                metrics=metrics,
            )
            self._pending.ancestry = _atif_ancestry_from_ist(ist)
            self._pending.step_ancestry_path = _build_ancestry_path(ist.function_ancestry, self._invocation_index)
            return flushed

        if event_type == IntermediateStepType.TOOL_END:
            if self._pending is not None:
                _record_observed_invocation(self._pending, ist, self._invocation_index)
                return None

            orphan_pending = _PendingAgentTurn(message="", timestamp=ist.event_timestamp, model_name=None, metrics=None)
            orphan_pending.ancestry = _atif_ancestry_from_ist(ist)
            orphan_pending.step_ancestry_path = _build_ancestry_path(ist.function_ancestry, self._invocation_index)
            _record_observed_invocation(orphan_pending, ist, self._invocation_index)
            invocation = orphan_pending.observed_invocations[0]
            step_extra = _atif_step_extra_model_from_ist(ist)
            step_extra.step_ancestry_path = orphan_pending.step_ancestry_path
            step_extra.tool_ancestry = [invocation.ancestry]
            step_extra.tool_ancestry_paths = [invocation.ancestry_path]
            extra = step_extra.model_dump(exclude_none=True)
            orphan_step = ATIFStep(
                step_id=self._step_id,
                source="agent",
                message="",
                timestamp=_epoch_to_iso(ist.event_timestamp),
                tool_calls=[invocation.tool_call],
                observation=ATIFObservation(results=[invocation.observation]),
                extra=extra or None,
            )
            self._step_id += 1
            self._emitted_steps.append(orphan_step)
            return orphan_step

        if event_type == IntermediateStepType.FUNCTION_END:
            if self._pending is not None:
                _record_observed_invocation(self._pending, ist, self._invocation_index)
            return None

        if state == IntermediateStepState.END and category not in (
                IntermediateStepCategory.LLM,
                IntermediateStepCategory.TOOL,
                IntermediateStepCategory.WORKFLOW,
        ):
            return None

        return None

    def finalize(self) -> list[ATIFStep]:
        """Flush any pending agent turn and return remaining steps."""
        result: list[ATIFStep] = []
        flushed = self._flush_pending()
        if flushed:
            result.append(flushed)
        return result

    def get_trajectory(self) -> ATIFTrajectory:
        """Build the complete ATIF trajectory from all emitted steps."""
        agent_step_count = sum(1 for s in self._emitted_steps if s.source == "agent")
        final_metrics = None
        if self._total_prompt or self._total_completion or self._total_cached or agent_step_count:
            final_metrics = ATIFFinalMetrics(
                total_prompt_tokens=self._total_prompt or None,
                total_completion_tokens=self._total_completion or None,
                total_cached_tokens=self._total_cached or None,
                total_steps=agent_step_count,
            )
        return ATIFTrajectory(
            agent=self._agent_config,
            steps=list(self._emitted_steps),
            final_metrics=final_metrics,
        )

    def _flush_pending(self) -> ATIFStep | None:
        """Convert the pending turn into an ATIFStep and clear it."""
        if self._pending is None:
            return None
        pending = self._pending
        sorted_invocations = sorted(pending.observed_invocations, key=lambda i: i.order_key)
        tool_calls = [obs.tool_call for obs in sorted_invocations] or None
        observations = [obs.observation for obs in sorted_invocations]
        observation = ATIFObservation(results=observations) if observations else None
        tool_ancestry = [obs.ancestry for obs in sorted_invocations]
        tool_ancestry_paths = [obs.ancestry_path for obs in sorted_invocations] or None
        if pending.ancestry is None:
            raise ValueError("Pending agent turn is missing required ATIF ancestry metadata")
        step_extra = AtifStepExtra(
            ancestry=pending.ancestry,
            tool_ancestry=tool_ancestry,
            step_ancestry_path=pending.step_ancestry_path,
            tool_ancestry_paths=tool_ancestry_paths,
            **pending.extra,
        )
        step = ATIFStep(
            step_id=self._step_id,
            source="agent",
            message=pending.message,
            timestamp=_epoch_to_iso(pending.timestamp),
            model_name=pending.model_name,
            tool_calls=tool_calls,
            observation=observation,
            metrics=pending.metrics,
            extra=step_extra.model_dump(exclude_none=True),
        )
        self._step_id += 1
        self._emitted_steps.append(step)
        self._pending = None
        return step
