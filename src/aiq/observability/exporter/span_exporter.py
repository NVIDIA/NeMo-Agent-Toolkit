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
from abc import abstractmethod
from typing import TypeVar

from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepState
from aiq.data_models.intermediate_step import TraceMetadata
from aiq.data_models.span import MimeTypes
from aiq.data_models.span import Span
from aiq.data_models.span import SpanAttributes
from aiq.data_models.span import SpanContext
from aiq.data_models.span import event_type_to_span_kind
from aiq.observability.exporter.base_exporter import IsolatedAttribute
from aiq.observability.exporter.processing_exporter import ProcessingExporter
from aiq.observability.mixin.serialize_mixin import SerializeMixin
from aiq.observability.utils.dict_utils import merge_dicts
from aiq.observability.utils.time_utils import ns_timestamp
from aiq.utils.type_utils import override

logger = logging.getLogger(__name__)

InputSpanT = TypeVar("InputSpanT")
OutputSpanT = TypeVar("OutputSpanT")


class SpanExporter(ProcessingExporter[InputSpanT, OutputSpanT], SerializeMixin):
    """Abstract base class for span exporters with processing pipeline support.

    This class specializes ProcessingExporter for span-based telemetry export. It converts
    IntermediateStep events into Span objects and supports processing pipelines for
    span transformation before export.

    The generic types work as follows:
    - InputSpanT: The type of spans that enter the processing pipeline (typically Span)
    - OutputSpanT: The type of spans after processing through the pipeline (e.g., OtelSpan)

    Key Features:
    - Automatic span creation from IntermediateStep events
    - Span lifecycle management (start/end event tracking)
    - Processing pipeline support via ProcessingExporter
    - Metadata and attribute handling
    - Usage information tracking
    - Automatic isolation of mutable state for concurrent execution using descriptors

    Inheritance Hierarchy:
    - BaseExporter: Core event subscription and lifecycle management + DescriptorIsolationMixin
    - ProcessingExporter: Adds processor pipeline functionality
    - SpanExporter: Specializes for span creation and export

    Event Processing Flow:
    1. IntermediateStep (START) → Create Span → Add to tracking
    2. IntermediateStep (END) → Complete Span → Process through pipeline → Export

    Args:
        context_state (AIQContextState, optional): The context state to use for the exporter. Defaults to None.
    """

    # Use descriptors for automatic isolation of span-specific state
    _outstanding_spans: IsolatedAttribute[dict] = IsolatedAttribute(dict)
    _span_stack: IsolatedAttribute[dict] = IsolatedAttribute(dict)
    _metadata_stack: IsolatedAttribute[dict] = IsolatedAttribute(dict)

    @abstractmethod
    async def export_processed(self, item: OutputSpanT) -> None:
        """Export the processed span.

        Args:
            item (OutputSpanT): The processed span to export.
        """
        pass

    @override
    def export(self, event: IntermediateStep) -> None:
        """The main logic that reacts to each IntermediateStep.

        Args:
            event (IntermediateStep): The event to process.
        """
        if not isinstance(event, IntermediateStep):
            return

        if (event.event_state == IntermediateStepState.START):
            self._process_start_event(event)
        elif (event.event_state == IntermediateStepState.END):
            self._process_end_event(event)

    def _process_start_event(self, event: IntermediateStep):
        """Process the start event of an intermediate step.

        Args:
            event (IntermediateStep): The event to process.
        """

        parent_span = None
        span_ctx = None

        # Look up the parent span to establish hierarchy
        # event.parent_id is the UUID of the last START step with a different UUID from current step
        # This maintains proper parent-child relationships in the span tree
        # Skip lookup if parent_id is "root" (indicates this is a top-level span)
        if len(self._span_stack) > 0 and event.parent_id and event.parent_id != "root":

            parent_span = self._span_stack.get(event.parent_id, None)
            if parent_span is None:
                logger.warning("No parent span found for step %s", event.UUID)
                return

            parent_span = parent_span.model_copy() if isinstance(parent_span, Span) else None
            if parent_span and parent_span.context:
                span_ctx = SpanContext(trace_id=parent_span.context.trace_id)

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

        sub_span = Span(
            name=sub_span_name,
            parent=parent_span,
            context=span_ctx,
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
        sub_span.set_attribute("aiq.span.kind", span_kind.value)

        if event.payload.data and event.payload.data.input:
            match = re.search(r"Human:\s*Question:\s*(.*)", str(event.payload.data.input))
            if match:
                human_question = match.group(1).strip()
                sub_span.set_attribute(SpanAttributes.INPUT_VALUE.value, human_question)
            else:
                serialized_input, is_json = self._serialize_payload(event.payload.data.input)
                sub_span.set_attribute(SpanAttributes.INPUT_VALUE.value, serialized_input)
                sub_span.set_attribute(SpanAttributes.INPUT_MIME_TYPE.value,
                                       MimeTypes.JSON.value if is_json else MimeTypes.TEXT.value)

        # Add metadata to the metadata stack
        start_metadata = event.payload.metadata or {}

        if isinstance(start_metadata, dict):
            self._metadata_stack[event.UUID] = start_metadata  # type: ignore
        elif isinstance(start_metadata, TraceMetadata):
            self._metadata_stack[event.UUID] = start_metadata.model_dump()  # type: ignore
        else:
            logger.warning("Invalid metadata type for step %s", event.UUID)
            return

        self._span_stack[event.UUID] = sub_span  # type: ignore
        self._outstanding_spans[event.UUID] = sub_span  # type: ignore

        logger.debug(
            "Added span to tracking (outstanding: %d, stack: %d, event_id: %s)",
            len(self._outstanding_spans),  # type: ignore
            len(self._span_stack),  # type: ignore
            event.UUID)

    def _process_end_event(self, event: IntermediateStep):
        """Process the end event of an intermediate step.

        Args:
            event (IntermediateStep): The event to process.
        """

        # Find the subspan that was created in the start event
        sub_span: Span | None = self._outstanding_spans.pop(event.UUID, None)  # type: ignore

        if sub_span is None:
            logger.warning("No subspan found for step %s", event.UUID)
            return

        self._span_stack.pop(event.UUID, None)  # type: ignore

        # Optionally add more attributes from usage_info or data
        usage_info = event.payload.usage_info
        if usage_info:
            sub_span.set_attribute(SpanAttributes.AIQ_USAGE_NUM_LLM_CALLS.value,
                                   usage_info.num_llm_calls if usage_info.num_llm_calls else 0)
            sub_span.set_attribute(SpanAttributes.AIQ_USAGE_SECONDS_BETWEEN_CALLS.value,
                                   usage_info.seconds_between_calls if usage_info.seconds_between_calls else 0)
            sub_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT.value,
                                   usage_info.token_usage.prompt_tokens if usage_info.token_usage else 0)
            sub_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION.value,
                                   usage_info.token_usage.completion_tokens if usage_info.token_usage else 0)
            sub_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL.value,
                                   usage_info.token_usage.total_tokens if usage_info.token_usage else 0)

        if event.payload.data and event.payload.data.output is not None:
            serialized_output, is_json = self._serialize_payload(event.payload.data.output)
            sub_span.set_attribute(SpanAttributes.OUTPUT_VALUE.value, serialized_output)
            sub_span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE.value,
                                   MimeTypes.JSON.value if is_json else MimeTypes.TEXT.value)

        # Merge metadata from start event with end event metadata
        start_metadata = self._metadata_stack.pop(event.UUID)  # type: ignore

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

        # Export the span with processing pipeline
        self._create_export_task(self._export_with_processing(sub_span))  # type: ignore

    @override
    async def _cleanup(self):
        """Clean up any remaining spans."""
        if self._outstanding_spans:  # type: ignore
            logger.warning("Not all spans were closed. Remaining: %s", self._outstanding_spans)  # type: ignore

            for span_info in self._outstanding_spans.values():  # type: ignore
                span_info.end()

        self._outstanding_spans.clear()  # type: ignore
        self._span_stack.clear()  # type: ignore
        self._metadata_stack.clear()  # type: ignore
