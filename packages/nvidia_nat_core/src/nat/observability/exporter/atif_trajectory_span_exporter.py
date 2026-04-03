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
"""ATIF Trajectory-based span exporter.

Collects IntermediateStep events, converts them to an ATIFTrajectory via
IntermediateStepToATIFConverter, then reconstructs a span tree from the ATIF
representation and exports the spans through the processing pipeline.

This is the "Path B" alternative to SpanExporter (Path A) for A/B parity
validation before ATIF-native cutover.
"""

import datetime
import logging
import os
import typing
from abc import abstractmethod
from typing import Any
from typing import TypeVar

from nat.data_models.atif import ATIFTrajectory
from nat.data_models.atif.atif_step_extra import AtifAncestry
from nat.data_models.atif.atif_step_extra import AtifInvocationInfo
from nat.data_models.atif.atif_step_extra import AtifStepExtra
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepState
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.span import MimeTypes
from nat.data_models.span import Span
from nat.data_models.span import SpanAttributes
from nat.data_models.span import SpanContext
from nat.data_models.span import SpanKind
from nat.data_models.span import event_type_to_span_kind
from nat.observability.exporter.base_exporter import IsolatedAttribute
from nat.observability.exporter.processing_exporter import ProcessingExporter
from nat.observability.mixin.serialize_mixin import SerializeMixin
from nat.observability.utils.time_utils import ns_timestamp
from nat.utils.atif_converter import IntermediateStepToATIFConverter
from nat.utils.type_utils import override

if typing.TYPE_CHECKING:
    from nat.builder.context import ContextState
    from nat.data_models.atif import ATIFStep

logger = logging.getLogger(__name__)

OutputSpanT = TypeVar("OutputSpanT")

# Map event type strings to SpanKind for the nat.span.kind attribute.
_EVENT_TYPE_TO_SPAN_KIND: dict[str, SpanKind] = {
    "WORKFLOW_START": SpanKind.WORKFLOW,
    "WORKFLOW_END": SpanKind.WORKFLOW,
    "LLM_START": SpanKind.LLM,
    "TOOL_START": SpanKind.TOOL,
    "FUNCTION_START": SpanKind.FUNCTION,
}


def _iso_to_epoch(timestamp: str) -> float:
    """Convert an ISO 8601 timestamp to Unix epoch seconds."""
    return datetime.datetime.fromisoformat(timestamp).timestamp()


class ATIFTrajectorySpanExporter(ProcessingExporter[Span, OutputSpanT], SerializeMixin):
    """Span exporter that converts IntermediateStep events via ATIFTrajectory.

    Collects all IntermediateStep events during a workflow execution. On
    WORKFLOW_END, converts them to an ATIFTrajectory using
    IntermediateStepToATIFConverter, reconstructs a span tree from the ATIF
    step.extra ancestry/invocation metadata, and exports each span through
    the processing pipeline.

    Parameters
    ----------
    context_state : ContextState, optional
        The context state for isolation and workflow metadata.
    span_prefix : str, optional
        Prefix for span attribute keys. Defaults to ``NAT_SPAN_PREFIX`` env var
        or ``"nat"``.
    """

    _collected_steps: IsolatedAttribute[list] = IsolatedAttribute(list)

    def __init__(self, context_state: "ContextState | None" = None, span_prefix: str | None = None):
        super().__init__(context_state=context_state)
        if span_prefix is None:
            span_prefix = os.getenv("NAT_SPAN_PREFIX", "nat").strip() or "nat"
        self._span_prefix = span_prefix

    @abstractmethod
    async def export_processed(self, item: OutputSpanT) -> None:
        """Export the processed span."""
        pass

    @override
    def export(self, event: IntermediateStep) -> None:
        """Collect IntermediateStep events; trigger export on WORKFLOW_END."""
        if not isinstance(event, IntermediateStep):
            return

        self._collected_steps.append(event)

        if event.event_type == IntermediateStepType.WORKFLOW_END:
            self._process_collected_steps()

    def _process_collected_steps(self) -> None:
        """Convert collected steps to ATIFTrajectory and export as spans."""
        steps = list(self._collected_steps)
        self._collected_steps.clear()

        if not steps:
            return

        # Build timing map from raw IST START/END event pairs.
        # UUID → (start_timestamp, end_timestamp) in epoch seconds.
        # Path A gets timing from START events (span_event_timestamp or event_timestamp)
        # and END events (event_timestamp). We replicate that here.
        timing_map: dict[str, tuple[float, float]] = {}
        start_times: dict[str, float] = {}

        for ist in steps:
            uid = ist.UUID
            if ist.event_state == IntermediateStepState.START:
                start_times[uid] = ist.payload.span_event_timestamp or ist.event_timestamp
            elif ist.event_state == IntermediateStepState.END:
                start_ts = ist.payload.span_event_timestamp or start_times.get(uid, ist.event_timestamp)
                end_ts = ist.event_timestamp
                timing_map[uid] = (start_ts, end_ts)

        converter = IntermediateStepToATIFConverter()
        trajectory = converter.convert(steps)

        # DEBUG: Dump trajectory
        import json
        logger.warning("=== ATIF TRAJECTORY DEBUG (%d steps) ===", len(trajectory.steps))
        for i, step in enumerate(trajectory.steps):
            extra = step.extra or {}
            tool_ancestries = extra.get("tool_ancestry", [])
            step_info = {
                "step_id": step.step_id,
                "source": step.source,
                "model_name": step.model_name,
                "tool_calls": [tc.function_name for tc in (step.tool_calls or [])],
                "ancestry": extra.get("ancestry"),
                "tool_ancestry": [
                    {"fn": ta.get("function_name"), "parent": ta.get("parent_name")}
                    for ta in tool_ancestries
                ],
            }
            logger.warning("ATIF[%d]: %s", i, json.dumps(step_info, default=str))
        logger.warning("=== END ATIF DEBUG ===")

        spans = self._trajectory_to_spans(trajectory, timing_map)

        # DEBUG: Log final span tree
        logger.warning("=== SPAN TREE DEBUG ===")
        for i, span in enumerate(spans):
            parent_info = f"parent_span_id={span.parent.context.span_id}" if span.parent else "parent=None (ROOT)"
            duration_ms = (span.end_time - span.start_time) / 1e6 if span.end_time else 0
            logger.warning(
                "Span %d: name=%s, span_id=%s, %s, kind=%s, duration=%.2fms",
                i, span.name, span.context.span_id, parent_info,
                span.attributes.get("nat.span.kind", "?"), duration_ms,
            )
        logger.warning("=== END SPAN TREE DEBUG ===")

        for span in spans:
            self._create_export_task(self._export_with_processing(span))

    # ------------------------------------------------------------------
    # ATIFTrajectory → Span tree conversion
    # ------------------------------------------------------------------

    def _trajectory_to_spans(
        self,
        trajectory: ATIFTrajectory,
        timing_map: dict[str, tuple[float, float]],
    ) -> list[Span]:
        """Convert an ATIFTrajectory into a list of Span objects.

        Reconstructs the span parent-child hierarchy using function_id /
        parent_id from the ATIF step.extra ancestry metadata.

        Parameters
        ----------
        trajectory : ATIFTrajectory
            The converted trajectory.
        timing_map : dict
            UUID → (start_epoch, end_epoch) from raw IST event pairs.
        """
        spans: list[Span] = []

        # Shared trace_id for all spans in this trajectory
        trace_id = self._context_state.workflow_trace_id.get()
        span_ctx_root = SpanContext(trace_id=trace_id) if trace_id else SpanContext()
        shared_trace_id = span_ctx_root.trace_id

        # function_id → most-recently-created Span for parent lookup
        fn_span_map: dict[str, Span] = {}
        workflow_span: Span | None = None
        # Deferred parent resolution for spans whose parent_id wasn't in fn_span_map yet
        pending_parents: list[tuple[Span, AtifAncestry]] = []

        for step in trajectory.steps:
            extra = step.extra or {}

            if step.source == "user":
                span = self._create_workflow_span(step, shared_trace_id, extra, timing_map)
                ancestry = extra.get("ancestry")
                if ancestry:
                    fn_span_map[ancestry["function_id"]] = span
                spans.append(span)
                workflow_span = span

            elif step.source == "agent":
                new_spans = self._create_agent_step_spans(
                    step, shared_trace_id, extra, fn_span_map, workflow_span, timing_map,
                    pending_parents,
                )
                spans.extend(new_spans)

        # Second pass: resolve deferred parents now that all spans are registered
        for span, ancestry in pending_parents:
            if ancestry.parent_id:
                found = fn_span_map.get(ancestry.parent_id)
                if found is not None:
                    span.parent = found.model_copy()

        # Update workflow span end_time to cover the full trajectory
        if workflow_span and spans:
            latest_end = max(s.end_time for s in spans if s.end_time is not None)
            if latest_end and latest_end > (workflow_span.end_time or 0):
                workflow_span.end_time = latest_end

        return spans

    def _create_workflow_span(
        self,
        step: "ATIFStep",
        trace_id: int,
        extra: dict[str, Any],
        timing_map: dict[str, tuple[float, float]],
    ) -> Span:
        """Create a workflow root span from a source='user' ATIF step."""
        ancestry = self._parse_ancestry(extra.get("ancestry"))
        invocation = self._parse_invocation(extra.get("invocation"))

        start_ns, end_ns = self._resolve_timing(invocation, step.timestamp, ancestry, timing_map)

        span_ctx = SpanContext(trace_id=trace_id)
        # Use the workflow name from the trajectory agent config, or ancestry
        span_name = (ancestry.function_name if ancestry else None) or "workflow"

        span = Span(
            name=span_name,
            context=span_ctx,
            parent=None,
            start_time=start_ns,
            attributes=self._build_base_attributes(
                event_type="WORKFLOW_START",
                ancestry=ancestry,
                trace_id=trace_id,
            ),
        )

        # Set input from user message
        if step.message:
            msg = step.message if isinstance(step.message, str) else str(step.message)
            span.set_attribute(SpanAttributes.INPUT_VALUE.value, msg)

        span.set_attribute(f"{self._span_prefix}.span.kind", SpanKind.WORKFLOW.value)
        self._set_session_attribute(span)

        # End time will be updated later to cover the full trajectory
        span.end(end_time=end_ns)

        return span

    def _create_agent_step_spans(
        self,
        step: "ATIFStep",
        trace_id: int,
        extra: dict[str, Any],
        fn_span_map: dict[str, Span],
        workflow_span: Span | None,
        timing_map: dict[str, tuple[float, float]],
        pending_parents: list[tuple[Span, AtifAncestry]] | None = None,
    ) -> list[Span]:
        """Create spans for a source='agent' ATIF step.

        Handles three cases:
        1. LLM turn with tools (has model_name): LLM span + child tool spans
        2. Orphan function step (no model_name, has tool_calls): single function span
        3. Terminal step (no model_name, no tool_calls): merge output into workflow span
        """
        spans: list[Span] = []

        ancestry = self._parse_ancestry(extra.get("ancestry"))
        invocation = self._parse_invocation(extra.get("invocation"))
        tool_ancestries_raw = extra.get("tool_ancestry", [])
        tool_invocations_raw = extra.get("tool_invocations") or []
        tool_calls = step.tool_calls or []
        has_tools = len(tool_calls) > 0

        # Case 3: Terminal/final step — merge output into workflow span, don't create a span.
        # Covers two cases:
        #   a) No model_name, no tools (terminal step from WORKFLOW_END)
        #   b) Has model_name, no tools (final LLM response before terminal step)
        if not has_tools:
            if workflow_span and step.message:
                msg = step.message if isinstance(step.message, str) else str(step.message)
                serialized, is_json = self._serialize_payload(msg)
                workflow_span.set_attribute(SpanAttributes.OUTPUT_VALUE.value, serialized)
                workflow_span.set_attribute(
                    SpanAttributes.OUTPUT_MIME_TYPE.value,
                    MimeTypes.JSON.value if is_json else MimeTypes.TEXT.value,
                )
                # Preserve token metrics from final LLM response on workflow span
                if step.metrics:
                    workflow_span.set_attribute(
                        SpanAttributes.LLM_TOKEN_COUNT_PROMPT.value,
                        step.metrics.prompt_tokens or 0,
                    )
                    workflow_span.set_attribute(
                        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION.value,
                        step.metrics.completion_tokens or 0,
                    )
                    total = (step.metrics.prompt_tokens or 0) + (step.metrics.completion_tokens or 0)
                    workflow_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL.value, total)
            return spans

        # Case 2: Orphan function step (no model_name, has tool_calls)
        # The converter wraps orphan FUNCTION_END/TOOL_END events as agent steps where
        # the step-level ancestry and tool_ancestry[0] represent the SAME function.
        # Create a single function span instead of LLM + tool.
        if has_tools and not step.model_name:
            for i, tc in enumerate(tool_calls):
                t_ancestry = self._parse_ancestry(
                    tool_ancestries_raw[i] if i < len(tool_ancestries_raw) else None
                )
                t_invocation = self._parse_invocation(
                    tool_invocations_raw[i] if i < len(tool_invocations_raw) else None
                )

                fn_span = self._create_function_span(
                    tool_call=tc,
                    step=step,
                    trace_id=trace_id,
                    fn_span_map=fn_span_map,
                    workflow_span=workflow_span,
                    tool_ancestry=t_ancestry,
                    tool_invocation=t_invocation,
                    timing_map=timing_map,
                    index=i,
                )
                spans.append(fn_span)

                if t_ancestry:
                    fn_span_map[t_ancestry.function_id] = fn_span
                    # If parent_id wasn't resolved, defer for second pass
                    if (pending_parents is not None
                            and t_ancestry.parent_id
                            and t_ancestry.parent_id not in fn_span_map):
                        pending_parents.append((fn_span, t_ancestry))

            return spans

        # Case 1: LLM turn with tools (has model_name)
        event_type = "LLM_START"

        start_ns, end_ns = self._resolve_timing(invocation, step.timestamp, ancestry, timing_map)

        # Find parent: try parent_id, then function_id (same callable), then workflow
        parent_span = self._resolve_parent(ancestry, fn_span_map, workflow_span)

        span_ctx = SpanContext(trace_id=trace_id)
        span_name = step.model_name or (ancestry.function_name if ancestry else None) or "agent"

        llm_span = Span(
            name=span_name,
            context=span_ctx,
            parent=parent_span.model_copy() if parent_span else None,
            start_time=start_ns,
            attributes=self._build_base_attributes(
                event_type=event_type,
                ancestry=ancestry,
                trace_id=trace_id,
            ),
        )

        span_kind = _EVENT_TYPE_TO_SPAN_KIND.get(event_type, SpanKind.LLM)
        llm_span.set_attribute(f"{self._span_prefix}.span.kind", span_kind.value)
        self._set_session_attribute(llm_span)

        # Set output from agent message
        if step.message:
            msg = step.message if isinstance(step.message, str) else str(step.message)
            serialized, is_json = self._serialize_payload(msg)
            llm_span.set_attribute(SpanAttributes.OUTPUT_VALUE.value, serialized)
            llm_span.set_attribute(
                SpanAttributes.OUTPUT_MIME_TYPE.value,
                MimeTypes.JSON.value if is_json else MimeTypes.TEXT.value,
            )

        # Set token metrics from step.metrics
        if step.metrics:
            llm_span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT.value,
                step.metrics.prompt_tokens or 0,
            )
            llm_span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION.value,
                step.metrics.completion_tokens or 0,
            )
            total = (step.metrics.prompt_tokens or 0) + (step.metrics.completion_tokens or 0)
            llm_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL.value, total)

        llm_span.end(end_time=end_ns)
        spans.append(llm_span)

        # Register in fn_span_map for child lookups
        if ancestry:
            fn_span_map[ancestry.function_id] = llm_span

        # Create tool/function child spans.
        # The converter may produce duplicate/misaligned entries when both TOOL_END
        # and FUNCTION_END fire for the same function. The TOOL_END entry uses the
        # caller's ancestry (misaligned), while FUNCTION_END uses the function's own
        # ancestry. Skip misaligned entries to avoid duplicate spans.
        seen_tool_names: set[str] = set()
        for i, tc in enumerate(tool_calls):
            tc_name = tc.function_name if hasattr(tc, "function_name") else str(tc)
            t_ancestry = self._parse_ancestry(
                tool_ancestries_raw[i] if i < len(tool_ancestries_raw) else None
            )
            t_invocation = self._parse_invocation(
                tool_invocations_raw[i] if i < len(tool_invocations_raw) else None
            )

            # Skip misaligned entries: tool_ancestry.function_name doesn't match tool_call name
            if t_ancestry and t_ancestry.function_name != tc_name:
                continue

            # Skip duplicates (same function_name already processed)
            if tc_name in seen_tool_names:
                continue
            seen_tool_names.add(tc_name)

            tool_span = self._create_tool_span(
                tool_call=tc,
                step=step,
                trace_id=trace_id,
                llm_span=llm_span,
                fn_span_map=fn_span_map,
                tool_ancestry=t_ancestry,
                tool_invocation=t_invocation,
                timing_map=timing_map,
                index=i,
            )
            spans.append(tool_span)

            # Register for nested lookups
            if t_ancestry:
                fn_span_map[t_ancestry.function_id] = tool_span
                # If parent_id wasn't resolved (tool defaults to llm_span), defer
                if (pending_parents is not None
                        and t_ancestry.parent_id
                        and t_ancestry.parent_id not in fn_span_map):
                    pending_parents.append((tool_span, t_ancestry))

        return spans

    def _create_function_span(
        self,
        tool_call: Any,
        step: "ATIFStep",
        trace_id: int,
        fn_span_map: dict[str, Span],
        workflow_span: Span | None,
        tool_ancestry: AtifAncestry | None,
        tool_invocation: AtifInvocationInfo | None,
        timing_map: dict[str, tuple[float, float]],
        index: int,
    ) -> Span:
        """Create a single function span from an orphan function step."""
        start_ns, end_ns = self._resolve_timing(tool_invocation, step.timestamp, tool_ancestry, timing_map)

        # Resolve parent: try parent_id in fn_span_map, fall back to workflow span
        parent_span = self._resolve_parent(tool_ancestry, fn_span_map, workflow_span)

        span_ctx = SpanContext(trace_id=trace_id)
        span_name = tool_call.function_name if hasattr(tool_call, "function_name") else str(tool_call)

        event_type = "FUNCTION_START"

        fn_span = Span(
            name=span_name,
            context=span_ctx,
            parent=parent_span.model_copy() if parent_span else None,
            start_time=start_ns,
            attributes=self._build_base_attributes(
                event_type=event_type,
                ancestry=tool_ancestry,
                trace_id=trace_id,
            ),
        )

        span_kind = _EVENT_TYPE_TO_SPAN_KIND.get(event_type, SpanKind.FUNCTION)
        fn_span.set_attribute(f"{self._span_prefix}.span.kind", span_kind.value)
        self._set_session_attribute(fn_span)

        # Set input from tool arguments
        if hasattr(tool_call, "arguments") and tool_call.arguments:
            serialized, is_json = self._serialize_payload(tool_call.arguments)
            fn_span.set_attribute(SpanAttributes.INPUT_VALUE.value, serialized)
            fn_span.set_attribute(
                SpanAttributes.INPUT_MIME_TYPE.value,
                MimeTypes.JSON.value if is_json else MimeTypes.TEXT.value,
            )

        # Set output from observation
        if step.observation and step.observation.results and index < len(step.observation.results):
            obs_result = step.observation.results[index]
            if obs_result.content:
                content = obs_result.content if isinstance(obs_result.content, str) else str(obs_result.content)
                serialized, is_json = self._serialize_payload(content)
                fn_span.set_attribute(SpanAttributes.OUTPUT_VALUE.value, serialized)
                fn_span.set_attribute(
                    SpanAttributes.OUTPUT_MIME_TYPE.value,
                    MimeTypes.JSON.value if is_json else MimeTypes.TEXT.value,
                )

        fn_span.end(end_time=end_ns)
        return fn_span

    def _create_tool_span(
        self,
        tool_call: Any,
        step: "ATIFStep",
        trace_id: int,
        llm_span: Span,
        fn_span_map: dict[str, Span],
        tool_ancestry: AtifAncestry | None,
        tool_invocation: AtifInvocationInfo | None,
        timing_map: dict[str, tuple[float, float]],
        index: int,
    ) -> Span:
        """Create a single tool/function span from a tool_call entry (under an LLM span)."""
        start_ns, end_ns = self._resolve_timing(tool_invocation, step.timestamp, tool_ancestry, timing_map)

        # Determine parent: lookup by parent_id, fallback to LLM span
        parent_span = llm_span
        if tool_ancestry and tool_ancestry.parent_id:
            found = fn_span_map.get(tool_ancestry.parent_id)
            if found is not None:
                parent_span = found

        span_ctx = SpanContext(trace_id=trace_id)
        span_name = tool_call.function_name if hasattr(tool_call, "function_name") else str(tool_call)

        # Determine event type based on ancestry or default to TOOL_START
        event_type = "TOOL_START"
        if tool_invocation and tool_invocation.framework:
            event_type = "FUNCTION_START"

        tool_span = Span(
            name=span_name,
            context=span_ctx,
            parent=parent_span.model_copy() if parent_span else None,
            start_time=start_ns,
            attributes=self._build_base_attributes(
                event_type=event_type,
                ancestry=tool_ancestry,
                trace_id=trace_id,
            ),
        )

        span_kind = _EVENT_TYPE_TO_SPAN_KIND.get(event_type, SpanKind.TOOL)
        tool_span.set_attribute(f"{self._span_prefix}.span.kind", span_kind.value)
        self._set_session_attribute(tool_span)

        # Set input from tool arguments
        if hasattr(tool_call, "arguments") and tool_call.arguments:
            serialized, is_json = self._serialize_payload(tool_call.arguments)
            tool_span.set_attribute(SpanAttributes.INPUT_VALUE.value, serialized)
            tool_span.set_attribute(
                SpanAttributes.INPUT_MIME_TYPE.value,
                MimeTypes.JSON.value if is_json else MimeTypes.TEXT.value,
            )

        # Set output from observation
        if step.observation and step.observation.results and index < len(step.observation.results):
            obs_result = step.observation.results[index]
            if obs_result.content:
                content = obs_result.content if isinstance(obs_result.content, str) else str(obs_result.content)
                serialized, is_json = self._serialize_payload(content)
                tool_span.set_attribute(SpanAttributes.OUTPUT_VALUE.value, serialized)
                tool_span.set_attribute(
                    SpanAttributes.OUTPUT_MIME_TYPE.value,
                    MimeTypes.JSON.value if is_json else MimeTypes.TEXT.value,
                )

        tool_span.end(end_time=end_ns)
        return tool_span

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_parent(
        self,
        ancestry: AtifAncestry | None,
        fn_span_map: dict[str, Span],
        workflow_span: Span | None,
    ) -> Span | None:
        """Resolve the parent span for a given ancestry.

        Lookup order:
        1. ancestry.parent_id in fn_span_map (direct parent callable)
        2. ancestry.function_id in fn_span_map (same callable, e.g. workflow → LLM)
        3. workflow_span (fall back to root)
        """
        if ancestry:
            if ancestry.parent_id:
                found = fn_span_map.get(ancestry.parent_id)
                if found is not None:
                    return found
            # Same callable: e.g., an LLM span inside a workflow that shares function_id
            found = fn_span_map.get(ancestry.function_id)
            if found is not None:
                return found
        return workflow_span

    def _parse_ancestry(self, raw: dict[str, Any] | AtifAncestry | None) -> AtifAncestry | None:
        """Parse ancestry data from an ATIF extra dict."""
        if raw is None:
            return None
        if isinstance(raw, AtifAncestry):
            return raw
        return AtifAncestry(**raw)

    def _parse_invocation(self, raw: dict[str, Any] | AtifInvocationInfo | None) -> AtifInvocationInfo | None:
        """Parse invocation timing data from an ATIF extra dict."""
        if raw is None:
            return None
        if isinstance(raw, AtifInvocationInfo):
            return raw
        return AtifInvocationInfo(**raw)

    def _resolve_timing(
        self,
        invocation: AtifInvocationInfo | None,
        step_timestamp: str | None,
        ancestry: AtifAncestry | None = None,
        timing_map: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[int, int]:
        """Resolve start/end nanosecond timestamps.

        Priority order:
        1. invocation.start_timestamp / end_timestamp (from ATIF extra)
        2. timing_map lookup by ancestry.function_id (from raw IST START/END pairs)
        3. step ISO timestamp with zero duration (fallback)
        """
        # Priority 1: ATIF invocation timestamps
        if invocation and invocation.start_timestamp is not None and invocation.end_timestamp is not None:
            return ns_timestamp(invocation.start_timestamp), ns_timestamp(invocation.end_timestamp)

        # Priority 2: timing map from raw IST events
        if timing_map and ancestry:
            timing = timing_map.get(ancestry.function_id)
            if timing is not None:
                return ns_timestamp(timing[0]), ns_timestamp(timing[1])

        # Priority 3: Fallback to step ISO timestamp, zero duration
        if step_timestamp:
            epoch = _iso_to_epoch(step_timestamp)
            ts_ns = ns_timestamp(epoch)
            return ts_ns, ts_ns

        # Last resort: current time
        import time
        now_ns = int(time.time() * 1e9)
        return now_ns, now_ns

    def _build_base_attributes(
        self,
        event_type: str,
        ancestry: AtifAncestry | None,
        trace_id: int,
    ) -> dict[str, Any]:
        """Build the common set of span attributes matching Path A parity."""
        attrs: dict[str, Any] = {
            f"{self._span_prefix}.event_type": event_type,
            f"{self._span_prefix}.function.id": ancestry.function_id if ancestry else "unknown",
            f"{self._span_prefix}.function.name": ancestry.function_name if ancestry else "unknown",
            f"{self._span_prefix}.function.parent_id": (
                ancestry.parent_id if ancestry and ancestry.parent_id else "unknown"
            ),
            f"{self._span_prefix}.function.parent_name": (
                ancestry.parent_name if ancestry and ancestry.parent_name else "unknown"
            ),
            f"{self._span_prefix}.workflow.trace_id": f"{trace_id:032x}" if trace_id else "unknown",
            f"{self._span_prefix}.conversation.id": self._context_state.conversation_id.get() or "unknown",
            f"{self._span_prefix}.workflow.run_id": self._context_state.workflow_run_id.get() or "unknown",
        }
        return attrs

    def _set_session_attribute(self, span: Span) -> None:
        """Set session.id from conversation_id for Phoenix session grouping."""
        try:
            conversation_id = self._context_state.conversation_id.get()
            if conversation_id:
                span.set_attribute("session.id", conversation_id)
        except (AttributeError, LookupError):
            pass

    @override
    async def _cleanup(self):
        """Clean up any remaining collected steps."""
        if self._collected_steps:
            logger.warning(
                "ATIFTrajectorySpanExporter: %d collected steps were not processed (no WORKFLOW_END received)",
                len(self._collected_steps),
            )
        self._collected_steps.clear()
        await super()._cleanup()
