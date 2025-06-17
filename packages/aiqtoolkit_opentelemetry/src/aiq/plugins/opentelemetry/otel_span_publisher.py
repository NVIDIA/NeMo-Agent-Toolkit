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
import re
import uuid
from typing import Any

from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import SpanContext

from aiq.builder.context import AIQContextState
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import TraceMetadata
from aiq.observability.span_publisher import AbstractSpanPublisher
from aiq.observability.utils import merge_dicts
from aiq.observability.utils import ns_timestamp
from aiq.plugins.opentelemetry.otel_span import MimeTypes
from aiq.plugins.opentelemetry.otel_span import OtelSpan
from aiq.plugins.opentelemetry.otel_span import event_type_to_span_kind

logger = logging.getLogger(__name__)


class OtelSpanPublisher(AbstractSpanPublisher):
    """Publisher for OtelSpan instances.

    This class is responsible for processing intermediate steps and creating OtelSpan instances and publishing them to
    the unified event stream.

    Args:
        context_state (AIQContextState | None): The context state to use for the publisher.
    """

    def __init__(self, context_state: AIQContextState | None = None):
        """Initialize the OtelSpanPublisher with the specified context state."""
        super().__init__(context_state)
        self._outstanding_spans: dict[str, OtelSpan] = {}
        self._span_stack: dict[str, OtelSpan] = {}
        self._metadata_stack: dict[str, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        """The name of the publisher."""
        return "otel_span_publisher"

    def _process_start_event(self, event: IntermediateStep):
        """Process the start event of an intermediate step.

        Args:
            event (IntermediateStep): The intermediate step to process.
        """

        parent_span = None
        span_ctx = None

        if (len(self._span_stack) > 0):
            parent_span = self._span_stack.get(event.function_ancestry.parent_id, None)
            if parent_span is None:
                logger.warning("No parent span found for step %s", event.UUID)
                return

            parent_ctx = parent_span.get_span_context()
            span_ctx = SpanContext(
                trace_id=parent_ctx.trace_id,
                span_id=uuid.uuid4().int & ((1 << 64) - 1),
                is_remote=False,
                trace_flags=parent_ctx.trace_flags,
                trace_state=parent_ctx.trace_state,
            )

        # Extract start/end times from the step
        # By convention, `span_event_timestamp` is the time we started, `event_timestamp` is the time we ended.
        # If span_event_timestamp is missing, we default to event_timestamp (meaning zero-length).
        s_ts = event.payload.span_event_timestamp or event.payload.event_timestamp
        start_ns = ns_timestamp(s_ts)

        # Optional: embed the LLM/tool name if present
        if event.payload.name:
            sub_span_name = f"{event.payload.name}"
        else:
            sub_span_name = f"{event.payload.event_type}"

        sub_span = OtelSpan(
            name=sub_span_name,
            context=span_ctx,
            parent=parent_span,
            attributes={
                "aiq.event_type": event.payload.event_type.value,
                "aiq.function.id": event.function_ancestry.function_id if event.function_ancestry else "unknown",
                "aiq.function.name": event.function_ancestry.function_name if event.function_ancestry else "unknown",
                "aiq.subspan.name": event.payload.name or "",
                "aiq.event_timestamp": event.event_timestamp,
                "aiq.framework": event.payload.framework.value if event.payload.framework else "unknown",
            },
            start_time=start_ns)

        span_kind = event_type_to_span_kind(event.event_type)
        sub_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, span_kind.value)

        if event.payload.data and event.payload.data.input:
            # optional parse
            match = re.search(r"Human:\s*Question:\s*(.*)", str(event.payload.data.input))
            if match:
                human_question = match.group(1).strip()
                sub_span.set_attribute(SpanAttributes.INPUT_VALUE, human_question)
            else:
                serialized_input, is_json = self._serialize_payload(event.payload.data.input)
                sub_span.set_attribute(SpanAttributes.INPUT_VALUE, serialized_input)
                sub_span.set_attribute(SpanAttributes.INPUT_MIME_TYPE,
                                       MimeTypes.JSON.value if is_json else MimeTypes.TEXT.value)

        # Add metadata to the metadata stack
        start_metadata = event.payload.metadata or {}

        if isinstance(start_metadata, dict):
            self._metadata_stack[event.UUID] = start_metadata
        elif isinstance(start_metadata, TraceMetadata):
            self._metadata_stack[event.UUID] = start_metadata.model_dump()
        else:
            logger.warning("Invalid metadata type for step %s", event.UUID)
            return

        # Store the span in both stacks
        self._span_stack[event.UUID] = sub_span
        self._outstanding_spans[event.UUID] = sub_span

        logger.debug("Added span to tracking (outstanding: %d, stack: %d, trace_id: %s)",
                     len(self._outstanding_spans),
                     len(self._span_stack),
                     sub_span.trace_id)

    def _process_end_event(self, event: IntermediateStep):
        """Process the end event of an intermediate step.

        Args:
            event (IntermediateStep): The intermediate step to process.
        """

        # Find the subspan that was created in the start event
        sub_span = self._outstanding_spans.pop(event.UUID, None)

        if sub_span is None:
            logger.warning("No subspan found for step %s (outstanding spans: %s, stack: %s)",
                           event.UUID,
                           list(self._outstanding_spans.keys()),
                           list(self._span_stack.keys()))
            return

        self._span_stack.pop(event.UUID, None)
        logger.debug("Processing end event for span %s (name: %s)", event.UUID, sub_span.name)

        # Optionally add more attributes from usage_info or data
        usage_info = event.payload.usage_info
        if usage_info:
            sub_span.set_attribute("aiq.usage.num_llm_calls",
                                   usage_info.num_llm_calls if usage_info.num_llm_calls else 0)
            sub_span.set_attribute("aiq.usage.seconds_between_calls",
                                   usage_info.seconds_between_calls if usage_info.seconds_between_calls else 0)
            sub_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                                   usage_info.token_usage.prompt_tokens if usage_info.token_usage else 0)
            sub_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                                   usage_info.token_usage.completion_tokens if usage_info.token_usage else 0)
            sub_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                                   usage_info.token_usage.total_tokens if usage_info.token_usage else 0)

        if event.payload.data and event.payload.data.output is not None:
            serialized_output, is_json = self._serialize_payload(event.payload.data.output)
            sub_span.set_attribute(SpanAttributes.OUTPUT_VALUE, serialized_output)
            sub_span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE,
                                   MimeTypes.JSON.value if is_json else MimeTypes.TEXT.value)

        # Merge metadata from start event with end event metadata
        start_metadata = self._metadata_stack.pop(event.UUID)

        if start_metadata is None:
            logger.warning("No metadata found for step %s", event.UUID)
            return

        end_metadata = event.payload.metadata or {}

        if not isinstance(end_metadata, (dict, TraceMetadata)):
            logger.warning("Invalid metadata type for step %s", event.UUID)
            return

        if isinstance(end_metadata, TraceMetadata):
            end_metadata = end_metadata.model_dump()

        merged_metadata = merge_dicts(start_metadata, end_metadata)
        serialized_metadata, is_json = self._serialize_payload(merged_metadata)
        sub_span.set_attribute("aiq.metadata", serialized_metadata)
        sub_span.set_attribute("aiq.metadata.mime_type", MimeTypes.JSON.value if is_json else MimeTypes.TEXT.value)

        end_ns = ns_timestamp(event.payload.event_timestamp)

        # End the subspan
        sub_span.end(end_time=end_ns)
        logger.debug("Ended span %s (name: %s)", event.UUID, sub_span.name)

        # Export the span with its parent context
        self._create_export_task(sub_span)
