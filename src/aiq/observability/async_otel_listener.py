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

import json
import logging
import re
import warnings
from contextlib import asynccontextmanager
from contextlib import contextmanager
from typing import Any

from openinference.semconv.trace import OpenInferenceSpanKindValues
from openinference.semconv.trace import SpanAttributes
from pydantic import TypeAdapter

from aiq.builder.context import AIQContextState
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepState
from aiq.utils.optional_imports import TelemetryOptionalImportError
from aiq.utils.optional_imports import try_import_opentelemetry

try:
    with warnings.catch_warnings():
        # Ignore deprecation warnings being triggered by weave. https://github.com/wandb/weave/issues/3666
        # and https://github.com/wandb/weave/issues/4533
        warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"^`sentry_sdk\.Hub` is deprecated")
        warnings.filterwarnings("ignore",
                                category=DeprecationWarning,
                                message=r"^Using extra keyword arguments on `Field` is deprecated")
        warnings.filterwarnings("ignore",
                                category=DeprecationWarning,
                                message=r"^`include` is deprecated and does nothing")
        from weave.trace.context import weave_client_context
        from weave.trace.context.call_context import get_current_call
        from weave.trace.context.call_context import set_call_stack
        from weave.trace.weave_client import Call
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    # we simply don't do anything if weave is not available
    pass

logger = logging.getLogger(__name__)

OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND

# Try to import OpenTelemetry modules
# If the dependencies are not installed, use dummy objects here
try:
    opentelemetry = try_import_opentelemetry()
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Span
    from opentelemetry.trace.propagation import set_span_in_context
except TelemetryOptionalImportError:
    from aiq.utils.optional_imports import DummySpan  # pylint: disable=ungrouped-imports
    from aiq.utils.optional_imports import DummyTrace  # pylint: disable=ungrouped-imports
    from aiq.utils.optional_imports import DummyTracerProvider  # pylint: disable=ungrouped-imports
    from aiq.utils.optional_imports import dummy_set_span_in_context  # pylint: disable=ungrouped-imports

    trace = DummyTrace  # pylint: disable=invalid-name
    TracerProvider = DummyTracerProvider
    Span = DummySpan
    set_span_in_context = dummy_set_span_in_context


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Merge two dictionaries, prioritizing non-null values from the first dictionary.

    Args:
        dict1 (dict): First dictionary (higher priority)
        dict2 (dict): Second dictionary (lower priority)

    Returns:
        dict: Merged dictionary with non-null values from dict1 taking precedence
    """
    result = dict2.copy()  # Start with a copy of the second dictionary
    for key, value in dict1.items():
        if value is not None:  # Only update if value is not None
            result[key] = value
    return result


def _ns_timestamp(seconds_float: float) -> int:
    """
    Convert AIQ Toolkit's float `event_timestamp` (in seconds) into an integer number
    of nanoseconds, as OpenTelemetry expects.
    """
    return int(seconds_float * 1e9)


class AsyncOtelSpanListener:
    """
    A separate, async class that listens to the AIQ Toolkit intermediate step
    event stream and creates proper Otel spans:

    - On FUNCTION_START => open a new top-level span
    - On any other intermediate step => open a child subspan (immediate open/close)
    - On FUNCTION_END => close the function's top-level span

    This runs fully independently from the normal AIQ Toolkit workflow, so that
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
        self._span_stack: dict[str, Span] = {}

        self._running = False

        # Prepare the tracer (optionally you might already have done this)
        if trace.get_tracer_provider() is None or not isinstance(trace.get_tracer_provider(), TracerProvider):
            tracer_provider = TracerProvider()
            trace.set_tracer_provider(tracer_provider)

        # We'll optionally attach exporters if you want (out of scope to do it here).
        # Example: tracer_provider.add_span_processor(BatchSpanProcessor(your_exporter))

        self._tracer = trace.get_tracer("aiq-async-otel-listener")

        # Initialize Weave-specific components if available
        self.gc = None
        self._weave_calls = {}
        if WEAVE_AVAILABLE:
            try:
                # Try to get the weave client, but don't fail if Weave isn't initialized
                self.gc = weave_client_context.require_weave_client()
            except Exception:
                # Weave is not initialized, so we don't do anything
                pass

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
                # run your AIQ Toolkit workflow
                ...
            # cleans up

        This sets up the subscription to the AIQ Toolkit event stream and starts the background loop.
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

        self._span_stack.clear()

        # Clean up any lingering Weave calls if Weave is available and initialized
        if self.gc is not None and self._weave_calls:
            for _, call in list(self._weave_calls.items()):
                self.gc.finish_call(call, {"status": "incomplete"})
            self._weave_calls.clear()

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
            parent_span = self._span_stack.get(step.function_ancestry.parent_id, None)
            if parent_span is None:
                logger.warning("No parent span found for step %s", step.UUID)
                return

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

        # Optional: add metadata to the span from TraceMetadata
        if step.payload.metadata:
            sub_span.set_attribute("aiq.metadata", step.payload.metadata.model_dump_json())

        self._span_stack[step.UUID] = sub_span

        self._outstanding_spans[step.UUID] = sub_span

        # Create corresponding Weave call if Weave is available and initialized
        if self.gc is not None:
            self._create_weave_call(step, sub_span)

    def _process_end_event(self, step: IntermediateStep):

        # Find the subspan that was created in the start event
        sub_span = self._outstanding_spans.pop(step.UUID, None)

        if sub_span is None:
            logger.warning("No subspan found for step %s", step.UUID)
            return

        self._span_stack.pop(step.UUID, None)

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

        # # Optional: add metadata to the span from TraceMetadata
        if step.payload.metadata:
            start_event_metadata = json.loads(sub_span.attributes.get("aiq.metadata", {}))
            end_event_metadata = json.loads(step.payload.metadata.model_dump_json())
            merged_event_metadata = merge_dicts(start_event_metadata, end_event_metadata)
            sub_span.set_attribute("aiq.metadata", json.dumps(merged_event_metadata))

        end_ns = _ns_timestamp(step.payload.event_timestamp)

        # End the subspan
        sub_span.end(end_time=end_ns)

        # Finish corresponding Weave call if Weave is available and initialized
        if self.gc is not None:
            self._finish_weave_call(step)

    @contextmanager
    def parent_call(self, trace_id: str, parent_call_id: str):
        """Context manager to set a parent call context for Weave.
        This allows connecting AIQ spans to existing traces from other frameworks.
        """
        dummy_call = Call(trace_id=trace_id, id=parent_call_id, _op_name="", project_id="", parent_id=None, inputs={})
        with set_call_stack([dummy_call]):
            yield

    def _create_weave_call(self, step: IntermediateStep, span: Span) -> None:
        """
        Create a Weave call directly from the span and step data,
        connecting to existing framework traces if available.
        """
        # Check for existing Weave trace/call
        existing_call = get_current_call()

        # Extract parent call if applicable
        parent_call = None

        # If we have an existing Weave call from another framework (e.g., LangChain),
        # use it as the parent
        if existing_call is not None:
            parent_call = existing_call
            logger.debug("Found existing Weave call: %s from trace: %s", existing_call.id, existing_call.trace_id)
        # Otherwise, check our internal stack for parent relationships
        elif len(self._weave_calls) > 0 and len(self._span_stack) > 1:
            # Get the parent span using stack position (one level up)
            parent_span_id = self._span_stack[-2].get_span_context().span_id
            # Find the corresponding weave call for this parent span
            for call in self._weave_calls.values():
                if getattr(call, "span_id", None) == parent_span_id:
                    parent_call = call
                    break

        # Generate a meaningful operation name based on event type
        event_type = step.payload.event_type.split(".")[-1]
        if step.payload.name:
            op_name = f"aiq.{event_type}.{step.payload.name}"
        else:
            op_name = f"aiq.{event_type}"

        # Create input dictionary
        inputs = {}
        if step.payload.data and step.payload.data.input is not None:
            try:
                # Add the input to the Weave call
                inputs["input"] = step.payload.data.input
            except Exception:
                # If serialization fails, use string representation
                inputs["input"] = str(step.payload.data.input)

        # Create the Weave call
        call = self.gc.create_call(
            op_name,
            inputs=inputs,
            parent=parent_call,
            attributes=span.attributes,
            display_name=op_name,
        )

        # Store the call with step UUID as key
        self._weave_calls[step.UUID] = call

        # Store span ID for parent reference
        setattr(call, "span_id", span.get_span_context().span_id)

        return call

    def _finish_weave_call(self, step: IntermediateStep) -> None:
        """
        Finish a previously created Weave call
        """
        # Find the call for this step
        call = self._weave_calls.pop(step.UUID, None)

        if call is None:
            logger.warning("No Weave call found for step %s", step.UUID)
            return

        # Create output dictionary
        outputs = {}
        if step.payload.data and step.payload.data.output is not None:
            try:
                # Add the output to the Weave call
                outputs["output"] = step.payload.data.output
            except Exception:
                # If serialization fails, use string representation
                outputs["output"] = str(step.payload.data.output)

        # Add usage information if available
        usage_info = step.payload.usage_info
        if usage_info:
            if usage_info.token_usage:
                outputs["prompt_tokens"] = usage_info.token_usage.prompt_tokens
                outputs["completion_tokens"] = usage_info.token_usage.completion_tokens
                outputs["total_tokens"] = usage_info.token_usage.total_tokens

            if usage_info.num_llm_calls:
                outputs["num_llm_calls"] = usage_info.num_llm_calls

            if usage_info.seconds_between_calls:
                outputs["seconds_between_calls"] = usage_info.seconds_between_calls

        # Finish the call with outputs
        self.gc.finish_call(call, outputs)
