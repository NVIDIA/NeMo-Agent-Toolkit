# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from contextlib import asynccontextmanager
from typing import Any

from openinference.semconv.trace import OpenInferenceSpanKindValues
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span
from opentelemetry.trace.propagation import set_span_in_context
from pydantic import TypeAdapter

from aiq.builder.context import AIQContextState
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepState

logger = logging.getLogger(__name__)

OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND


def _ns_timestamp(seconds_float: float) -> int:
    """
    Convert AgentIQ’s float `event_timestamp` (in seconds) into an integer number
    of nanoseconds, as OpenTelemetry expects.
    """
    return int(seconds_float * 1e9)


class AsyncOtelSpanListener:
    """
    A separate, async class that listens to the AgentIQ intermediate step
    event stream and creates proper Otel spans:

    - On FUNCTION_START => open a new top-level span
    - On any other intermediate step => open a child subspan (immediate open/close)
    - On FUNCTION_END => close the function’s top-level span

    This runs fully independently from the normal AgentIQ workflow, so that
    the workflow is not blocking or entangled by OTel calls.
    """

    def __init__(self, context_state: AIQContextState | None = None):
        """
        :param context_state: Optionally supply a specific AIQContextState.
                              If None, uses the global singleton.
        """
        self._context_state = context_state or AIQContextState.get()

        # Maintain a subscription so we can unsubscribe on shutdown
        self._subscription = None

        # Outstanding spans which have been opened but not yet closed
        self._outstanding_spans: dict[str, Span] = {}

        # Stack of spans, for when we need to create a child span
        self._span_stack: list[Span] = []

        self._running = False

        # Prepare the tracer (optionally you might already have done this)
        if trace.get_tracer_provider() is None or not isinstance(trace.get_tracer_provider(), TracerProvider):
            tracer_provider = TracerProvider()
            trace.set_tracer_provider(tracer_provider)

        # We’ll optionally attach exporters if you want (out of scope to do it here).
        # Example: tracer_provider.add_span_processor(BatchSpanProcessor(your_exporter))

        self._tracer = trace.get_tracer("aiq-async-otel-listener")

    def _on_next(self, step: IntermediateStep) -> None:
        """
        The main logic that reacts to each IntermediateStep.
        """
        if (step.event_state == IntermediateStepState.START):

            self._process_start_event(step)

        elif (step.event_state == IntermediateStepState.END):

            self._process_end_event(step)

    def _on_error(self, exc: Exception) -> None:
        logger.error("Error in intermediate step subscription: %s", exc, exc_info=True)

    def _on_complete(self) -> None:
        logger.debug("Intermediate step stream completed. No more events will arrive.")

    @asynccontextmanager
    async def start(self):
        """
        Usage::

            otel_listener = AsyncOtelSpanListener()
            async with otel_listener.start():
                # run your AgentIQ workflow
                ...
            # cleans up

        This sets up the subscription to the AgentIQ event stream and starts the background loop.
        """
        try:
            # Subscribe to the event stream
            subject = self._context_state.event_stream.get()
            self._subscription = subject.subscribe(
                on_next=self._on_next,
                on_error=self._on_error,
                on_complete=self._on_complete,
            )

            self._running = True

            yield  # let the caller do their workflow

        finally:
            # Cleanup
            self._running = False
            # Close out any running spans
            await self._cleanup()

            if self._subscription:
                self._subscription.unsubscribe()
            self._subscription = None

    async def _cleanup(self):
        """
        Close any remaining open spans.
        """
        if self._outstanding_spans:
            logger.warning(
                "Not all spans were closed. Ensure all start events have a corresponding end event. Remaining: %s",
                self._outstanding_spans)

        for span_info in self._outstanding_spans.values():
            span_info.end()

        self._outstanding_spans.clear()

        if self._span_stack:
            logger.error(
                "Not all spans were closed. Ensure all start events have a corresponding end event. Remaining: %s",
                self._span_stack)

        self._span_stack.clear()

    def _serialize_payload(self, input_value: Any) -> tuple[str, bool]:
        """
        Serialize the input value to a string. Returns a tuple with the serialized value and a boolean indicating if the
        serialization is JSON or a string
        """
        try:
            return TypeAdapter(type(input_value)).dump_json(input_value).decode('utf-8'), True
        except Exception:
            # Fallback to string representation if we can't serialize using pydantic
            return str(input_value), False

    def _process_start_event(self, step: IntermediateStep):

        parent_ctx = None

        if (len(self._span_stack) > 0):
            parent_span = self._span_stack[-1]

            parent_ctx = set_span_in_context(parent_span)

        # Extract start/end times from the step
        # By convention, `span_event_timestamp` is the time we started, `event_timestamp` is the time we ended.
        # If span_event_timestamp is missing, we default to event_timestamp (meaning zero-length).
        s_ts = step.payload.span_event_timestamp or step.payload.event_timestamp
        start_ns = _ns_timestamp(s_ts)

        # Optional: embed the LLM/tool name if present
        if step.payload.name:
            sub_span_name = f"{step.payload.name}"
        else:
            sub_span_name = f"{step.payload.event_type}"

        # Start the subspan
        sub_span = self._tracer.start_span(
            name=sub_span_name,
            context=parent_ctx,
            attributes={
                "aiq.event_type": step.payload.event_type.value,
                "aiq.function.id": step.function_ancestry.function_id,
                "aiq.function.name": step.function_ancestry.function_name,
                "aiq.subspan.name": step.payload.name or "",
                "aiq.event_timestamp": step.event_timestamp,
                "aiq.framework": step.payload.framework.value if step.payload.framework else "unknown",
            },
            start_time=start_ns,
        )

        event_type_to_span_kind = {
            "LLM_START": OpenInferenceSpanKindValues.LLM,
            "LLM_END": OpenInferenceSpanKindValues.LLM,
            "LLM_NEW_TOKEN": OpenInferenceSpanKindValues.LLM,
            "TOOL_START": OpenInferenceSpanKindValues.TOOL,
            "TOOL_END": OpenInferenceSpanKindValues.TOOL,
            "FUNCTION_START": OpenInferenceSpanKindValues.CHAIN,
            "FUNCTION_END": OpenInferenceSpanKindValues.CHAIN,
        }

        span_kind = event_type_to_span_kind.get(step.event_type, OpenInferenceSpanKindValues.UNKNOWN)
        sub_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, span_kind.value)

        if step.payload.data and step.payload.data.input:
            # optional parse
            match = re.search(r"Human:\s*Question:\s*(.*)", str(step.payload.data.input))
            if match:
                human_question = match.group(1).strip()
                sub_span.set_attribute(SpanAttributes.INPUT_VALUE, human_question)
            else:
                serialized_input, is_json = self._serialize_payload(step.payload.data.input)
                sub_span.set_attribute(SpanAttributes.INPUT_VALUE, serialized_input)
                sub_span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json" if is_json else "text/plain")

        self._span_stack.append(sub_span)

        self._outstanding_spans[step.UUID] = sub_span

    def _process_end_event(self, step: IntermediateStep):

        # Find the subspan that was created in the start event
        sub_span = self._outstanding_spans.pop(step.UUID, None)

        if sub_span is None:
            logger.warning("No subspan found for step %s", step.UUID)
            return

        self._span_stack.pop()

        # Optionally add more attributes from usage_info or data
        usage_info = step.payload.usage_info
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

        if step.payload.data and step.payload.data.output is not None:
            serialized_output, is_json = self._serialize_payload(step.payload.data.output)
            sub_span.set_attribute(SpanAttributes.OUTPUT_VALUE, serialized_output)
            sub_span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json" if is_json else "text/plain")

        end_ns = _ns_timestamp(step.payload.event_timestamp)

        # End the subspan
        sub_span.end(end_time=end_ns)
