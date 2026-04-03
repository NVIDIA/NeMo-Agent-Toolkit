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

        converter = IntermediateStepToATIFConverter()
        trajectory = converter.convert(steps)

        spans = self._trajectory_to_spans(trajectory)

        for span in spans:
            self._create_export_task(self._export_with_processing(span))

    # ------------------------------------------------------------------
    # ATIFTrajectory → Span tree conversion
    # ------------------------------------------------------------------

    def _trajectory_to_spans(self, trajectory: ATIFTrajectory) -> list[Span]:
        """Convert an ATIFTrajectory into a list of Span objects.

        Reconstructs the span parent-child hierarchy using function_id /
        parent_id from the ATIF step.extra ancestry metadata.
        """
        spans: list[Span] = []

        # Shared trace_id for all spans in this trajectory
        trace_id = self._context_state.workflow_trace_id.get()
        span_ctx_root = SpanContext(trace_id=trace_id) if trace_id else SpanContext()
        shared_trace_id = span_ctx_root.trace_id

        # function_id → most-recently-created Span for parent lookup
        fn_span_map: dict[str, Span] = {}

        for step in trajectory.steps:
            extra = step.extra or {}

            if step.source == "user":
                span = self._create_workflow_span(step, shared_trace_id, extra)
                ancestry = extra.get("ancestry")
                if ancestry:
                    fn_span_map[ancestry["function_id"]] = span
                spans.append(span)

            elif step.source == "agent":
                new_spans = self._create_agent_step_spans(step, shared_trace_id, extra, fn_span_map)
                spans.extend(new_spans)

        return spans

    def _create_workflow_span(
        self,
        step: "ATIFStep",
        trace_id: int,
        extra: dict[str, Any],
    ) -> Span:
        """Create a workflow root span from a source='user' ATIF step."""
        ancestry = self._parse_ancestry(extra.get("ancestry"))
        invocation = self._parse_invocation(extra.get("invocation"))

        start_ns, end_ns = self._resolve_timing(invocation, step.timestamp)

        span_ctx = SpanContext(trace_id=trace_id)
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

        # End the span (workflow spans span the full trajectory)
        span.end(end_time=end_ns)

        return span

    def _create_agent_step_spans(
        self,
        step: "ATIFStep",
        trace_id: int,
        extra: dict[str, Any],
        fn_span_map: dict[str, Span],
    ) -> list[Span]:
        """Create spans for a source='agent' ATIF step (LLM turn + tool calls)."""
        spans: list[Span] = []

        ancestry = self._parse_ancestry(extra.get("ancestry"))
        invocation = self._parse_invocation(extra.get("invocation"))
        tool_ancestries_raw = extra.get("tool_ancestry", [])
        tool_invocations_raw = extra.get("tool_invocations") or []
        tool_calls = step.tool_calls or []

        # Determine if this is a final agent response or an LLM turn with tools
        has_tools = len(tool_calls) > 0
        event_type = "LLM_START" if has_tools or step.model_name else "WORKFLOW_END"

        start_ns, end_ns = self._resolve_timing(invocation, step.timestamp)

        # Find parent span from ancestry
        parent_span = None
        if ancestry and ancestry.parent_id:
            parent_span = fn_span_map.get(ancestry.parent_id)

        span_ctx = SpanContext(trace_id=trace_id)
        span_name = (ancestry.function_name if ancestry else None) or step.model_name or "agent"

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

        # Create tool/function child spans
        for i, tc in enumerate(tool_calls):
            t_ancestry = self._parse_ancestry(
                tool_ancestries_raw[i] if i < len(tool_ancestries_raw) else None
            )
            t_invocation = self._parse_invocation(
                tool_invocations_raw[i] if i < len(tool_invocations_raw) else None
            )

            tool_span = self._create_tool_span(
                tool_call=tc,
                step=step,
                trace_id=trace_id,
                llm_span=llm_span,
                fn_span_map=fn_span_map,
                tool_ancestry=t_ancestry,
                tool_invocation=t_invocation,
                index=i,
            )
            spans.append(tool_span)

            # Register for nested lookups
            if t_ancestry:
                fn_span_map[t_ancestry.function_id] = tool_span

        return spans

    def _create_tool_span(
        self,
        tool_call: Any,
        step: "ATIFStep",
        trace_id: int,
        llm_span: Span,
        fn_span_map: dict[str, Span],
        tool_ancestry: AtifAncestry | None,
        tool_invocation: AtifInvocationInfo | None,
        index: int,
    ) -> Span:
        """Create a single tool/function span from a tool_call entry."""
        start_ns, end_ns = self._resolve_timing(tool_invocation, step.timestamp)

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
            # Functions (non-LLM callables) typically use FUNCTION_START
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
    ) -> tuple[int, int]:
        """Resolve start/end nanosecond timestamps from invocation or step timestamp.

        Falls back to the step ISO timestamp with zero duration when invocation
        timing is unavailable.
        """
        if invocation and invocation.start_timestamp is not None and invocation.end_timestamp is not None:
            return ns_timestamp(invocation.start_timestamp), ns_timestamp(invocation.end_timestamp)

        # Fallback: parse step ISO timestamp, zero duration
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
