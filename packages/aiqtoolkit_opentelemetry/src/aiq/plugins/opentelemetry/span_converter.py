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

from openinference.semconv.trace import OpenInferenceSpanKindValues
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import SpanContext
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode
from opentelemetry.trace import TraceFlags

from aiq.data_models.span import Span
from aiq.data_models.span import SpanStatusCode
from aiq.plugins.opentelemetry.otel_span import OtelSpan

logger = logging.getLogger(__name__)

SPAN_EVENT_TYPE_TO_SPAN_KIND_MAP = {
    "LLM_START": OpenInferenceSpanKindValues.LLM,
    "LLM_END": OpenInferenceSpanKindValues.LLM,
    "LLM_NEW_TOKEN": OpenInferenceSpanKindValues.LLM,
    "TOOL_START": OpenInferenceSpanKindValues.TOOL,
    "TOOL_END": OpenInferenceSpanKindValues.TOOL,
    "FUNCTION_START": OpenInferenceSpanKindValues.CHAIN,
    "FUNCTION_END": OpenInferenceSpanKindValues.CHAIN,
}


def convert_event_type_to_span_kind(event_type: str) -> OpenInferenceSpanKindValues:
    """Convert an event type to a span kind.

    Args:
        event_type (str): The event type to convert

    Returns:
        OpenInferenceSpanKindValues: The corresponding span kind
    """
    return SPAN_EVENT_TYPE_TO_SPAN_KIND_MAP.get(event_type, OpenInferenceSpanKindValues.UNKNOWN)


def convert_span_status_code(aiq_status_code: SpanStatusCode) -> StatusCode:
    """Convert AIQ SpanStatusCode to OpenTelemetry StatusCode.

    Args:
        aiq_status_code (SpanStatusCode): The AIQ span status code to convert

    Returns:
        StatusCode: The corresponding OpenTelemetry StatusCode
    """
    status_map = {
        SpanStatusCode.OK: StatusCode.OK,
        SpanStatusCode.ERROR: StatusCode.ERROR,
        SpanStatusCode.UNSET: StatusCode.UNSET,
    }
    return status_map.get(aiq_status_code, StatusCode.UNSET)


def convert_span_to_otel(aiq_span: Span) -> OtelSpan:
    """Convert an AIQ Span to an OtelSpan using stateless recursive conversion.

    This is completely stateless - uses parent info directly from Span objects.
    Each conversion operation is self-contained with only local caching to avoid
    duplicate work within the same operation.

    Args:
        aiq_span (Span): The AIQ span to convert

    Returns:
        OtelSpan: The converted OtelSpan with proper parent hierarchy.
    """
    local_cache: dict[int, OtelSpan] = {}  # Local cache just for this operation
    return _convert_span_with_parents(aiq_span, local_cache)


def _convert_span_with_parents(aiq_span: Span, local_cache: dict[int, OtelSpan]) -> OtelSpan:
    """Convert a span ensuring its parent chain is converted first (stateless).

    This uses only a local cache for this single conversion operation.
    No state is maintained between different span conversion operations.

    Args:
        aiq_span (Span): The span to convert along with its parent chain
        local_cache (dict[int, OtelSpan]): Local cache only for this conversion operation

    Returns:
        OtelSpan: The converted OtelSpan with proper parent relationships
    """
    if not aiq_span.context:
        # Create a span without context - fallback case
        return OtelSpan(
            name=aiq_span.name,
            context=None,
            attributes=aiq_span.attributes.copy(),
            start_time=aiq_span.start_time,
            end_time=aiq_span.end_time,
        )

    span_id = aiq_span.context.span_id

    # Check local cache (only for this conversion operation)
    if span_id in local_cache:
        return local_cache[span_id]

    # Convert parent first (if exists) using parent info from Span object
    parent_otel_span = None
    if aiq_span.parent:
        parent_otel_span = _convert_span_with_parents(aiq_span.parent, local_cache)

    # Create OpenTelemetry SpanContext - inherit trace_id from parent if available
    trace_id = aiq_span.context.trace_id
    if parent_otel_span and parent_otel_span.get_span_context():
        trace_id = parent_otel_span.get_span_context().trace_id

    otel_span_context = SpanContext(
        trace_id=trace_id,
        span_id=aiq_span.context.span_id,
        is_remote=False,
        trace_flags=TraceFlags(1),  # SAMPLED
    )

    # Convert status
    otel_status_code = convert_span_status_code(aiq_span.status.code)
    otel_status = Status(otel_status_code, aiq_span.status.message)

    # Create the OtelSpan with parent reference
    otel_span = OtelSpan(
        name=aiq_span.name,
        context=otel_span_context,
        parent=parent_otel_span,
        attributes=aiq_span.attributes.copy(),
        start_time=aiq_span.start_time,
        end_time=aiq_span.end_time,
        status=otel_status,
    )
    # Set span kind
    span_kind = convert_event_type_to_span_kind(aiq_span.attributes.get("aiq.event_type", "UNKNOWN"))
    otel_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, span_kind.value)

    # Convert and add events
    for aiq_event in aiq_span.events:
        event_timestamp_ns = int(aiq_event.timestamp) if aiq_event.timestamp else None
        otel_span.add_event(name=aiq_event.name, attributes=aiq_event.attributes, timestamp=event_timestamp_ns)

    # Store in local cache (only for this operation)
    local_cache[span_id] = otel_span

    return otel_span


def convert_spans_to_otel_batch(spans: list[Span]) -> list[OtelSpan]:
    """Convert a list of AIQ spans to OtelSpans using stateless conversion.

    This is useful for batch processing or demos. Each span is converted
    independently using the stateless approach.

    Args:
        spans (list[Span]): List of AIQ spans to convert

    Returns:
        list[OtelSpan]: List of converted OtelSpans with proper parent-child relationships
    """
    return [convert_span_to_otel(span) for span in spans]
