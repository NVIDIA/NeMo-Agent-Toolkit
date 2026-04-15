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
"""ATIF trajectory-to-span converter.

See ``README.md`` in this directory for usage guidance and span hierarchy details.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

from nat.atif import AtifAncestry
from nat.atif import AtifInvocationInfo
from nat.atif import AtifStepExtra
from nat.data_models.span import MimeTypes
from nat.data_models.span import Span
from nat.data_models.span import SpanAttributes
from nat.data_models.span import SpanContext
from nat.data_models.span import SpanKind
from nat.observability.mixin.serialize_mixin import SerializeMixin
from nat.observability.utils.time_utils import ns_timestamp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso_to_epoch(timestamp: str) -> float:
    """Convert an ISO 8601 timestamp to Unix epoch seconds."""
    return datetime.fromisoformat(timestamp).timestamp()


def _new_trace_id() -> int:
    """Generate a random 128-bit trace ID for a new trace."""
    return uuid.uuid4().int


def _is_terminal_agent_step(step: dict[str, Any]) -> bool:
    """True for agent steps that represent a final answer (no tool_calls)."""
    return (step.get("source") == "agent" and bool(step.get("message")) and not step.get("tool_calls"))


def _topo_sort_indices(ancestries: list[AtifAncestry]) -> list[int]:
    """Return indices in topological order (parents before children)."""
    id_to_idx = {a.function_id: i for i, a in enumerate(ancestries)}
    visited: set[int] = set()
    order: list[int] = []

    def _visit(idx: int) -> None:
        if idx in visited:
            return
        visited.add(idx)
        parent_id = ancestries[idx].parent_id
        if parent_id and parent_id in id_to_idx:
            _visit(id_to_idx[parent_id])
        order.append(idx)

    for i in range(len(ancestries)):
        _visit(i)
    return order


def _message_to_str(message: Any) -> str:
    """Normalise an ATIF message field to a plain string."""
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        return json.dumps(message, default=str)
    return str(message) if message else ""


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


class ATIFTrajectorySpanExporter(SerializeMixin):
    """Converts complete ATIF trajectories to NAT Span objects.

    Parameters
    ----------
    span_prefix : str, optional
        Prefix for span attribute keys.  Defaults to the
        ``NAT_SPAN_PREFIX`` environment variable, or ``"nat"``.
    """

    def __init__(self, span_prefix: str | None = None):
        if span_prefix is None:
            span_prefix = os.getenv("NAT_SPAN_PREFIX", "nat").strip() or "nat"
        self._span_prefix = span_prefix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(self, trajectory_data: dict[str, Any]) -> list[Span]:
        """Convert an ATIF trajectory dict to a list of Span objects.

        Parameters
        ----------
        trajectory_data : dict
            ATIF trajectory as a dict (parsed JSON).  Must contain at
            least ``session_id``, ``agent``, and ``steps``.

        Returns
        -------
        list[Span]
            Flat list of Span objects.  The first element is always
            the root WORKFLOW span.
        """
        session_id: str = trajectory_data["session_id"]
        agent_name: str = trajectory_data["agent"]["name"]
        steps: list[dict[str, Any]] = trajectory_data["steps"]
        trace_id = _new_trace_id()

        span_lookup: dict[str, Span] = {}
        delegation_refs: dict[str, str] = {}  # subagent_session_id -> tool_function_id
        spans: list[Span] = []

        # --- root WORKFLOW span ---
        first_ts, last_ts = self._trajectory_time_bounds(steps)
        root_fn_id = f"workflow_{session_id}"
        root_span = self._make_span(
            name=agent_name,
            function_id=root_fn_id,
            function_name=agent_name,
            parent_id=None,
            parent_name=None,
            event_type_str="WORKFLOW_START",
            span_kind=SpanKind.WORKFLOW,
            trace_id=trace_id,
            start_epoch=first_ts,
            end_epoch=last_ts,
            session_id=session_id,
            span_lookup=span_lookup,
        )
        span_lookup[root_fn_id] = root_span

        # --- walk steps ---
        first_user_msg: str | None = None
        last_agent_msg: str | None = None

        for step in steps:
            source = step.get("source", "")

            if source == "user":
                msg = _message_to_str(step.get("message", ""))
                if first_user_msg is None and msg:
                    first_user_msg = msg
                continue

            # Agent or system steps — need valid AtifStepExtra
            extra_raw = step.get("extra") or {}
            try:
                step_extra = AtifStepExtra.model_validate(extra_raw)
            except Exception:
                # Agent step without usable extra — capture output only
                if _is_terminal_agent_step(step):
                    last_agent_msg = _message_to_str(step.get("message", ""))
                logger.debug("Skipping step %s: no valid AtifStepExtra", step.get("step_id"))
                continue

            is_system = source == "system"
            has_tool_calls = bool(step.get("tool_calls"))

            if _is_terminal_agent_step(step):
                # Terminal agent step — final answer with no tool_calls
                last_agent_msg = _message_to_str(step.get("message", ""))
                llm_span = self._create_llm_span(
                    step,
                    step_extra,
                    trace_id,
                    session_id,
                    span_lookup,
                    root_fn_id,
                )
                spans.append(llm_span)
            elif is_system and has_tool_calls:
                # System step with tool_calls (no LLM involved)
                system_spans = self._create_system_tool_spans(
                    step,
                    step_extra,
                    trace_id,
                    session_id,
                    span_lookup,
                    root_fn_id,
                    delegation_refs,
                )
                spans.extend(system_spans)
            elif is_system and not has_tool_calls and step.get("message"):
                # Terminal system step — capture final output
                last_agent_msg = _message_to_str(step.get("message", ""))
                func_span = self._create_function_span(
                    step,
                    step_extra,
                    trace_id,
                    session_id,
                    span_lookup,
                    root_fn_id,
                )
                spans.append(func_span)
            else:
                agent_spans = self._create_agent_spans(
                    step,
                    step_extra,
                    trace_id,
                    session_id,
                    span_lookup,
                    root_fn_id,
                    delegation_refs,
                )
                spans.extend(agent_spans)

        # --- set workflow I/O ---
        if first_user_msg:
            root_span.set_attribute(SpanAttributes.INPUT_VALUE.value, first_user_msg)
            root_span.set_attribute(SpanAttributes.INPUT_MIME_TYPE.value, MimeTypes.TEXT.value)
        if last_agent_msg:
            root_span.set_attribute(SpanAttributes.OUTPUT_VALUE.value, last_agent_msg)
            root_span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE.value, MimeTypes.TEXT.value)

        spans.insert(0, root_span)

        # --- subagent trajectories ---
        for sub_traj in trajectory_data.get("subagent_trajectories", []):
            sub_spans = self._convert_subagent(
                sub_traj,
                trace_id,
                span_lookup,
                session_id,
                delegation_refs,
            )
            spans.extend(sub_spans)

        return spans

    # ------------------------------------------------------------------
    # Agent step → spans
    # ------------------------------------------------------------------

    def _create_llm_span(
        self,
        step: dict[str, Any],
        step_extra: AtifStepExtra,
        trace_id: int,
        session_id: str,
        span_lookup: dict[str, Span],
        root_fn_id: str,
    ) -> Span:
        """Create an LLM span from an agent step."""
        ancestry = step_extra.ancestry
        inv = step_extra.invocation

        start_epoch = inv.start_timestamp if inv else None
        end_epoch = inv.end_timestamp if inv else None
        if end_epoch is None and step.get("timestamp"):
            end_epoch = _iso_to_epoch(step["timestamp"])
        if start_epoch is None:
            start_epoch = end_epoch or 0.0

        # Reparent root-level agent spans under the workflow span
        parent_id = ancestry.parent_id
        if not parent_id or parent_id == "root":
            parent_id = root_fn_id

        span = self._make_span(
            name=ancestry.function_name,
            function_id=ancestry.function_id,
            function_name=ancestry.function_name,
            parent_id=parent_id,
            parent_name=ancestry.parent_name,
            event_type_str="LLM_END",
            span_kind=SpanKind.LLM,
            trace_id=trace_id,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            session_id=session_id,
            span_lookup=span_lookup,
            framework=inv.framework if inv else None,
        )

        msg = _message_to_str(step.get("message", ""))
        if msg:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE.value, msg)
            span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE.value, MimeTypes.TEXT.value)

        metrics = step.get("metrics")
        if metrics:
            prompt = metrics.get("prompt_tokens") or 0
            completion = metrics.get("completion_tokens") or 0
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT.value, prompt)
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION.value, completion)
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL.value, prompt + completion)

        if step.get("extra"):
            serialized, is_json = self._serialize_payload(step["extra"])
            span.set_attribute(f"{self._span_prefix}.metadata", serialized)
            span.set_attribute(
                f"{self._span_prefix}.metadata.mime_type",
                MimeTypes.JSON.value if is_json else MimeTypes.TEXT.value,
            )

        span_lookup[ancestry.function_id] = span
        return span

    def _create_agent_spans(
        self,
        step: dict[str, Any],
        step_extra: AtifStepExtra,
        trace_id: int,
        session_id: str,
        span_lookup: dict[str, Span],
        root_fn_id: str,
        delegation_refs: dict[str, str],
    ) -> list[Span]:
        """Create LLM span + child tool spans from an agent step with tool_calls."""
        spans: list[Span] = []

        llm_span = self._create_llm_span(
            step,
            step_extra,
            trace_id,
            session_id,
            span_lookup,
            root_fn_id,
        )
        spans.append(llm_span)

        tool_calls = step.get("tool_calls") or []
        tool_ancestry = step_extra.tool_ancestry
        tool_invocations = step_extra.tool_invocations or []
        obs_results = (step.get("observation") or {}).get("results") or []

        if tool_calls and tool_ancestry:
            sorted_indices = _topo_sort_indices(tool_ancestry)

            for idx in sorted_indices:
                if idx >= len(tool_calls):
                    continue
                tc = tool_calls[idx]
                t_anc = tool_ancestry[idx]
                t_inv = tool_invocations[idx] if idx < len(tool_invocations) else None

                obs_content: str | None = None
                if idx < len(obs_results):
                    content = obs_results[idx].get("content")
                    obs_content = (content if isinstance(content, str) else
                                   (json.dumps(content, default=str) if content else None))

                    # Track subagent delegation refs
                    for ref in obs_results[idx].get("subagent_trajectory_ref") or []:
                        sub_sid = ref.get("session_id")
                        if sub_sid:
                            delegation_refs[sub_sid] = t_anc.function_id

                tool_span = self._create_tool_span(
                    ancestry=t_anc,
                    invocation=t_inv,
                    tool_name=tc["function_name"],
                    tool_args=tc.get("arguments", {}),
                    tool_output=obs_content,
                    trace_id=trace_id,
                    session_id=session_id,
                    span_lookup=span_lookup,
                )
                spans.append(tool_span)

        return spans

    def _create_function_span(
        self,
        step: dict[str, Any],
        step_extra: AtifStepExtra,
        trace_id: int,
        session_id: str,
        span_lookup: dict[str, Span],
        root_fn_id: str,
    ) -> Span:
        """Create a FUNCTION span from a system step (no LLM, no tool_calls)."""
        ancestry = step_extra.ancestry
        inv = step_extra.invocation

        start_epoch = inv.start_timestamp if inv else None
        end_epoch = inv.end_timestamp if inv else None
        if end_epoch is None and step.get("timestamp"):
            end_epoch = _iso_to_epoch(step["timestamp"])
        if start_epoch is None:
            start_epoch = end_epoch or 0.0

        parent_id = ancestry.parent_id
        if not parent_id or parent_id == "root":
            parent_id = root_fn_id

        span = self._make_span(
            name=ancestry.function_name,
            function_id=ancestry.function_id,
            function_name=ancestry.function_name,
            parent_id=parent_id,
            parent_name=ancestry.parent_name,
            event_type_str="FUNCTION_END",
            span_kind=SpanKind.FUNCTION,
            trace_id=trace_id,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            session_id=session_id,
            span_lookup=span_lookup,
        )

        msg = _message_to_str(step.get("message", ""))
        if msg:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE.value, msg)
            span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE.value, MimeTypes.TEXT.value)

        span_lookup[ancestry.function_id] = span
        return span

    def _create_system_tool_spans(
        self,
        step: dict[str, Any],
        step_extra: AtifStepExtra,
        trace_id: int,
        session_id: str,
        span_lookup: dict[str, Span],
        root_fn_id: str,
        delegation_refs: dict[str, str],
    ) -> list[Span]:
        """Create a FUNCTION parent span + TOOL child spans from a system step."""
        spans: list[Span] = []

        # Create a FUNCTION span as the parent (not LLM since no LLM is involved)
        func_span = self._create_function_span(
            step,
            step_extra,
            trace_id,
            session_id,
            span_lookup,
            root_fn_id,
        )
        spans.append(func_span)

        tool_calls = step.get("tool_calls") or []
        tool_ancestry = step_extra.tool_ancestry
        tool_invocations = step_extra.tool_invocations or []
        obs_results = (step.get("observation") or {}).get("results") or []

        if tool_calls and tool_ancestry:
            sorted_indices = _topo_sort_indices(tool_ancestry)

            for idx in sorted_indices:
                if idx >= len(tool_calls):
                    continue
                tc = tool_calls[idx]
                t_anc = tool_ancestry[idx]
                t_inv = tool_invocations[idx] if idx < len(tool_invocations) else None

                obs_content: str | None = None
                if idx < len(obs_results):
                    content = obs_results[idx].get("content")
                    obs_content = (content if isinstance(content, str) else
                                   (json.dumps(content, default=str) if content else None))

                    for ref in obs_results[idx].get("subagent_trajectory_ref") or []:
                        sub_sid = ref.get("session_id")
                        if sub_sid:
                            delegation_refs[sub_sid] = t_anc.function_id

                tool_span = self._create_tool_span(
                    ancestry=t_anc,
                    invocation=t_inv,
                    tool_name=tc["function_name"],
                    tool_args=tc.get("arguments", {}),
                    tool_output=obs_content,
                    trace_id=trace_id,
                    session_id=session_id,
                    span_lookup=span_lookup,
                )
                spans.append(tool_span)

        return spans

    def _create_tool_span(
        self,
        ancestry: AtifAncestry,
        invocation: AtifInvocationInfo | None,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_output: str | None,
        trace_id: int,
        session_id: str,
        span_lookup: dict[str, Span],
    ) -> Span:
        """Build a TOOL span for a single tool call."""
        start_epoch = invocation.start_timestamp if invocation and invocation.start_timestamp else 0.0
        end_epoch = invocation.end_timestamp if invocation else None

        span = self._make_span(
            name=tool_name,
            function_id=ancestry.function_id,
            function_name=tool_name,
            parent_id=ancestry.parent_id,
            parent_name=ancestry.parent_name,
            event_type_str="TOOL_END",
            span_kind=SpanKind.TOOL,
            trace_id=trace_id,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            session_id=session_id,
            span_lookup=span_lookup,
        )

        if tool_args:
            serialized_input, is_json = self._serialize_payload(tool_args)
            span.set_attribute(SpanAttributes.INPUT_VALUE.value, serialized_input)
            span.set_attribute(
                SpanAttributes.INPUT_MIME_TYPE.value,
                MimeTypes.JSON.value if is_json else MimeTypes.TEXT.value,
            )

        if tool_output is not None:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE.value, tool_output)
            span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE.value, MimeTypes.TEXT.value)

        span_lookup[ancestry.function_id] = span
        return span

    # ------------------------------------------------------------------
    # Subagent handling
    # ------------------------------------------------------------------

    def _convert_subagent(
        self,
        sub_traj: dict[str, Any],
        parent_trace_id: int,
        parent_span_lookup: dict[str, Span],
        parent_session_id: str,
        delegation_refs: dict[str, str],
    ) -> list[Span]:
        """Process a subagent trajectory recursively.

        Subagent spans share the parent trace ID so they appear in the
        same Phoenix trace.  The subagent's root WORKFLOW span is linked
        as a child of the delegating tool span when the reference can be
        resolved.
        """
        sub_session_id = sub_traj.get("session_id", "")

        # Convert the subagent trajectory independently
        sub_exporter = ATIFTrajectorySpanExporter(span_prefix=self._span_prefix)
        sub_spans = sub_exporter.convert(sub_traj)

        # Override trace_id on all subagent spans to share the parent trace
        for span in sub_spans:
            if span.context:
                span.context.trace_id = parent_trace_id

        # Link the subagent's root workflow span to the delegating tool span
        delegating_fn_id = delegation_refs.get(sub_session_id)
        if delegating_fn_id and sub_spans:
            parent_tool_span = parent_span_lookup.get(delegating_fn_id)
            if parent_tool_span:
                root_sub_span = sub_spans[0]
                root_sub_span.parent = parent_tool_span.model_copy()
                if parent_tool_span.context:
                    root_sub_span.context.trace_id = parent_tool_span.context.trace_id

        return sub_spans

    # ------------------------------------------------------------------
    # Span construction
    # ------------------------------------------------------------------

    def _make_span(
        self,
        name: str,
        function_id: str,
        function_name: str,
        parent_id: str | None,
        parent_name: str | None,
        event_type_str: str,
        span_kind: SpanKind,
        trace_id: int,
        start_epoch: float,
        end_epoch: float | None,
        session_id: str,
        span_lookup: dict[str, Span],
        framework: str | None = None,
    ) -> Span:
        """Build a Span with standard NAT attributes."""
        parent_span = None
        if parent_id and parent_id != "root":
            ps = span_lookup.get(parent_id)
            if ps is not None:
                parent_span = ps.model_copy()

        span_ctx = SpanContext(trace_id=trace_id)

        p = self._span_prefix
        attributes: dict[str, Any] = {
            f"{p}.event_type": event_type_str,
            f"{p}.function.id": function_id or "unknown",
            f"{p}.function.name": function_name or "unknown",
            f"{p}.function.parent_id": parent_id or "unknown",
            f"{p}.function.parent_name": parent_name or "unknown",
            f"{p}.subspan.name": function_name or "",
            f"{p}.event_timestamp": end_epoch or start_epoch,
            f"{p}.framework": framework or "unknown",
            f"{p}.conversation.id": session_id,
            f"{p}.workflow.run_id": session_id,
            f"{p}.workflow.trace_id": f"{trace_id:032x}",
        }

        span = Span(
            name=name,
            parent=parent_span,
            context=span_ctx,
            attributes=attributes,
            start_time=ns_timestamp(start_epoch),
        )
        span.set_attribute(f"{p}.span.kind", span_kind.value)
        span.set_attribute("session.id", session_id)

        if end_epoch is not None:
            span.end(end_time=ns_timestamp(end_epoch))

        return span

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _trajectory_time_bounds(steps: list[dict[str, Any]]) -> tuple[float, float]:
        """Find the earliest and latest timestamps across all steps.

        Prefers ``extra.invocation`` epoch timestamps (authoritative)
        and only falls back to ISO ``step.timestamp`` fields when no
        invocation timestamps are available.
        """
        inv_first = float("inf")
        inv_last = 0.0
        iso_first = float("inf")
        iso_last = 0.0

        for step in steps:
            ts = step.get("timestamp")
            if ts:
                epoch = _iso_to_epoch(ts)
                iso_first = min(iso_first, epoch)
                iso_last = max(iso_last, epoch)

            extra = step.get("extra") or {}
            inv = extra.get("invocation")
            if inv:
                if inv.get("start_timestamp"):
                    inv_first = min(inv_first, inv["start_timestamp"])
                if inv.get("end_timestamp"):
                    inv_last = max(inv_last, inv["end_timestamp"])

            for ti in extra.get("tool_invocations") or []:
                if ti and ti.get("start_timestamp"):
                    inv_first = min(inv_first, ti["start_timestamp"])
                if ti and ti.get("end_timestamp"):
                    inv_last = max(inv_last, ti["end_timestamp"])

        # Prefer invocation timestamps; fall back to ISO timestamps
        has_inv = inv_first != float("inf")
        first_ts = inv_first if has_inv else iso_first
        last_ts = inv_last if has_inv else iso_last

        if first_ts == float("inf"):
            first_ts = 0.0
        if last_ts == 0.0:
            last_ts = first_ts

        return first_ts, last_ts
