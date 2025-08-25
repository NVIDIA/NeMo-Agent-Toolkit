# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import time

from openinference.semconv.trace import OpenInferenceSpanKindValues
from openinference.semconv.trace import SpanAttributes

from nat.data_models.span import Span
from nat.data_models.span import SpanStatusCode
from nat.plugins.opentelemetry.otel_span import OtelSpan
from opentelemetry.trace import SpanContext
from opentelemetry.trace import SpanKind
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode
from opentelemetry.trace import TraceFlags

logger = logging.getLogger(__name__)

SPAN_EVENT_TYPE_TO_SPAN_KIND_MAP = {
    "LLM_START": OpenInferenceSpanKindValues.LLM,
    "LLM_END": OpenInferenceSpanKindValues.LLM,
    "LLM_NEW_TOKEN": OpenInferenceSpanKindValues.LLM,
    "TOOL_START": OpenInferenceSpanKindValues.TOOL,
    "TOOL_END": OpenInferenceSpanKindValues.TOOL,
    "FUNCTION_START": OpenInferenceSpanKindValues.CHAIN,
    "FUNCTION_END": OpenInferenceSpanKindValues.CHAIN,
    "WORKFLOW_START": OpenInferenceSpanKindValues.CHAIN,
    "WORKFLOW_END": OpenInferenceSpanKindValues.CHAIN,
    "TASK_START": OpenInferenceSpanKindValues.CHAIN,
    "TASK_END": OpenInferenceSpanKindValues.CHAIN,
    "CUSTOM_START": OpenInferenceSpanKindValues.CHAIN,
    "CUSTOM_END": OpenInferenceSpanKindValues.CHAIN,
    "EMBEDDER_START": OpenInferenceSpanKindValues.EMBEDDING,
    "EMBEDDER_END": OpenInferenceSpanKindValues.EMBEDDING,
    "RETRIEVER_START": OpenInferenceSpanKindValues.RETRIEVER,
    "RETRIEVER_END": OpenInferenceSpanKindValues.RETRIEVER,
    "AGENT_START": OpenInferenceSpanKindValues.AGENT,
    "AGENT_END": OpenInferenceSpanKindValues.AGENT,
    "RERANKER_START": OpenInferenceSpanKindValues.RERANKER,
    "RERANKER_END": OpenInferenceSpanKindValues.RERANKER,
    "GUARDRAIL_START": OpenInferenceSpanKindValues.GUARDRAIL,
    "GUARDRAIL_END": OpenInferenceSpanKindValues.GUARDRAIL,
    "EVALUATOR_START": OpenInferenceSpanKindValues.EVALUATOR,
    "EVALUATOR_END": OpenInferenceSpanKindValues.EVALUATOR,
}


# Reuse expensive objects to avoid repeated creation
class _SharedObjects:

    def __init__(self):
        self.resource = None  # type: ignore
        self.instrumentation_scope = None  # type: ignore


_shared = _SharedObjects()
_SAMPLED_TRACE_FLAGS = TraceFlags(1)


def _get_shared_resource():
    """Get shared resource object to avoid repeated creation."""
    if _shared.resource is None:
        from opentelemetry.sdk.resources import Resource
        _shared.resource = Resource.create()  # type: ignore
    return _shared.resource


def _get_shared_instrumentation_scope():
    """Get shared instrumentation scope to avoid repeated creation."""
    if _shared.instrumentation_scope is None:
        from opentelemetry.sdk.trace import InstrumentationScope
        _shared.instrumentation_scope = InstrumentationScope("nat", "1.0.0")  # type: ignore
    return _shared.instrumentation_scope


def convert_event_type_to_span_kind(event_type: str) -> OpenInferenceSpanKindValues:
    """Convert an event type to a span kind.

    Args:
        event_type (str): The event type to convert

    Returns:
        OpenInferenceSpanKindValues: The corresponding span kind
    """
    return SPAN_EVENT_TYPE_TO_SPAN_KIND_MAP.get(event_type, OpenInferenceSpanKindValues.UNKNOWN)


def convert_span_status_code(nat_status_code: SpanStatusCode) -> StatusCode:
    """Convert NAT SpanStatusCode to OpenTelemetry StatusCode.

    Args:
        nat_status_code (SpanStatusCode): The NAT span status code to convert

    Returns:
        StatusCode: The corresponding OpenTelemetry StatusCode
    """
    status_map = {
        SpanStatusCode.OK: StatusCode.OK,
        SpanStatusCode.ERROR: StatusCode.ERROR,
        SpanStatusCode.UNSET: StatusCode.UNSET,
    }
    return status_map.get(nat_status_code, StatusCode.UNSET)


def convert_span_to_otel(nat_span: Span) -> OtelSpan:
    """Convert a NAT Span to an OtelSpan using ultra-fast conversion.

    Args:
        nat_span (Span): The NAT span to convert

    Returns:
        OtelSpan: The converted OtelSpan with proper parent hierarchy.
    """
    # Fast path for spans without context
    if not nat_span.context:
        # Create minimal OtelSpan bypassing expensive constructor
        otel_span = object.__new__(OtelSpan)  # Bypass __init__
        otel_span._name = nat_span.name
        otel_span._context = None  # type: ignore
        otel_span._parent = None
        otel_span._attributes = nat_span.attributes.copy()
        otel_span._events = []
        otel_span._links = []
        otel_span._kind = SpanKind.INTERNAL
        otel_span._start_time = nat_span.start_time
        otel_span._end_time = nat_span.end_time
        otel_span._status = Status(StatusCode.UNSET)
        otel_span._ended = False
        otel_span._resource = _get_shared_resource()  # type: ignore
        otel_span._instrumentation_scope = _get_shared_instrumentation_scope()  # type: ignore
        otel_span._dropped_attributes = 0
        otel_span._dropped_events = 0
        otel_span._dropped_links = 0
        otel_span._status_description = None
        return otel_span

    # Process parent efficiently (if needed)
    parent_otel_span = None
    trace_id = nat_span.context.trace_id

    if nat_span.parent:
        parent_otel_span = convert_span_to_otel(nat_span.parent)
        parent_context = parent_otel_span.get_span_context()
        trace_id = parent_context.trace_id

    # Create SpanContext efficiently
    otel_span_context = SpanContext(
        trace_id=trace_id,
        span_id=nat_span.context.span_id,
        is_remote=False,
        trace_flags=_SAMPLED_TRACE_FLAGS,  # Reuse flags object
    )

    # Create OtelSpan bypassing expensive constructor
    otel_span = object.__new__(OtelSpan)  # Bypass __init__
    otel_span._name = nat_span.name
    otel_span._context = otel_span_context
    otel_span._parent = parent_otel_span
    otel_span._attributes = nat_span.attributes.copy()
    otel_span._events = []
    otel_span._links = []
    otel_span._kind = SpanKind.INTERNAL
    otel_span._start_time = nat_span.start_time
    otel_span._end_time = nat_span.end_time

    # Reuse status conversion
    status_code = convert_span_status_code(nat_span.status.code)
    otel_span._status = Status(status_code, nat_span.status.message)

    otel_span._ended = False
    otel_span._resource = _get_shared_resource()  # type: ignore
    otel_span._instrumentation_scope = _get_shared_instrumentation_scope()  # type: ignore
    otel_span._dropped_attributes = 0
    otel_span._dropped_events = 0
    otel_span._dropped_links = 0
    otel_span._status_description = None

    # Set span kind efficiently (direct attribute modification)
    event_type = nat_span.attributes.get("nat.event_type", "UNKNOWN")
    span_kind = SPAN_EVENT_TYPE_TO_SPAN_KIND_MAP.get(event_type, OpenInferenceSpanKindValues.UNKNOWN)
    otel_span._attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] = span_kind.value

    # Process events (only if they exist)
    if nat_span.events:
        for nat_event in nat_span.events:
            # Optimize timestamp handling
            if isinstance(nat_event.timestamp, int):
                event_timestamp_ns = nat_event.timestamp
            elif nat_event.timestamp:
                event_timestamp_ns = int(nat_event.timestamp)
            else:
                event_timestamp_ns = int(time.time() * 1e9)

            # Add event directly to internal list (bypass add_event method)
            otel_span._events.append({
                "name": nat_event.name, "attributes": nat_event.attributes, "timestamp": event_timestamp_ns
            })

    return otel_span


def convert_spans_to_otel_batch(spans: list[Span]) -> list[OtelSpan]:
    """Convert a list of NAT spans to OtelSpans using stateless conversion.

    This is useful for batch processing or demos. Each span is converted
    independently using the stateless approach.

    Args:
        spans (list[Span]): List of NAT spans to convert

    Returns:
        list[OtelSpan]: List of converted OtelSpans with proper parent-child relationships
    """
    return [convert_span_to_otel(span) for span in spans]
