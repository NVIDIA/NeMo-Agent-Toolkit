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
from collections.abc import Generator
from contextlib import contextmanager

from weave.trace.context import weave_client_context
from weave.trace.context.call_context import get_current_call
from weave.trace.context.call_context import set_call_stack
from weave.trace.weave_client import Call

from aiq.data_models.span import Span
from aiq.data_models.span import SpanAttributes
from aiq.observability.exporter.base_exporter import IsolatedAttribute
from aiq.utils.log_utils import LogFilter

logger = logging.getLogger(__name__)

# Alternative: Use LogFilter to filter specific message patterns
presidio_filter = LogFilter([
    "nlp_engine not provided",
    "Created NLP engine",
    "registry not provided",
    "Loaded recognizer",
    "Recognizer not added to registry"
])


class WeaveMixin:
    """Mixin for Weave exporters.

    This mixin provides a default implementation of the export method for Weave exporters.
    It uses the weave_client_context to create and finish Weave calls.

    Args:
        project (str): The project name to group the telemetry traces.
        entity (str | None): The entity name to group the telemetry traces.
    """

    _weave_calls: IsolatedAttribute[dict[int, Call]] = IsolatedAttribute(dict)
    _in_flight_calls: IsolatedAttribute[set[int]] = IsolatedAttribute(set)

    def __init__(self, *args, project: str, entity: str | None = None, verbose: bool = False, **kwargs):
        """Initialize the Weave exporter with the specified project and entity.

        Args:
            project (str): The project name to group the telemetry traces.
            entity (str | None): The entity name to group the telemetry traces.
        """
        self._gc = weave_client_context.require_weave_client()
        self._project = project
        self._entity = entity

        # Optionally, set log filtering for presidio-analyzer to reduce verbosity
        if not verbose:
            presidio_logger = logging.getLogger('presidio-analyzer')
            presidio_logger.addFilter(presidio_filter)

        super().__init__(*args, **kwargs)

    async def export_processed(self, item: Span | list[Span]) -> None:
        """Export a batch of spans.

        Args:
            item (Span | list[Span]): The span or list of spans to export.
        """
        if not isinstance(item, list):
            spans = [item]
        else:
            spans = item

        for span in spans:
            self._export_processed(span)

    def _export_processed(self, span: Span) -> None:
        """Export a single span.

        Args:
            span (Span): The span to export.
        """
        span_id = span.context.span_id if span.context else None  # type: ignore
        if span_id is None:
            logger.warning("Span has no context or span_id, skipping export")
            return

        try:
            call = self._create_weave_call(span)
            self._finish_weave_call(call, span)
        except Exception as e:
            logger.error("Error exporting spans: %s", e, exc_info=True)
            # Clean up in-flight tracking if call creation/finishing failed
            self._in_flight_calls.discard(span_id)

    @contextmanager
    def parent_call(self, trace_id: str, parent_call_id: str) -> Generator[None]:
        """Create a dummy Weave call for the parent span.

        Args:
            trace_id (str): The trace ID of the parent span.
            parent_call_id (str): The ID of the parent call.

        Yields:
            None: The dummy Weave call.
        """
        dummy_call = Call(trace_id=trace_id, id=parent_call_id, _op_name="", project_id="", parent_id=None, inputs={})
        with set_call_stack([dummy_call]):
            yield

    def _create_weave_call(self, span: Span) -> Call:
        """
        Create a Weave call directly from the span and step data, connecting to existing framework traces if available.

        Args:
            span (Span): The span to create a Weave call for.

        Returns:
            Call: The Weave call.
        """
        span_id = span.context.span_id if span.context else None  # type: ignore
        if span_id is None:
            raise ValueError("Span has no context or span_id")

        # Mark this call as in-flight to prevent premature cleanup
        self._in_flight_calls.add(span_id)

        # Check for existing Weave trace/call
        existing_call = get_current_call()

        # Extract parent call if applicable
        parent_call = None

        # If we have an existing Weave call from another framework (e.g., LangChain),
        # use it as the parent (but only if it's actually a Call object)
        if existing_call is not None and isinstance(existing_call, Call):
            parent_call = existing_call
            logger.debug("Found existing Weave call: %s from trace: %s", existing_call.id, existing_call.trace_id)
        # Otherwise, check our internal stack for parent relationships
        elif len(self._weave_calls) > 0:
            # Get the parent span using stack position (one level up)
            parent_span_id = (span.parent.context.span_id if span.parent and span.parent.context else None
                              )  # type: ignore
            if parent_span_id:
                # Find the corresponding weave call for this parent span
                for call in self._weave_calls.values():
                    if getattr(call, "span_id", None) == parent_span_id:
                        parent_call = call
                        break

        # Generate a meaningful operation name based on event type
        span_event_type = span.attributes.get(SpanAttributes.AIQ_EVENT_TYPE.value, "unknown")
        event_type = span_event_type.split(".")[-1]
        if span.name:
            op_name = f"aiq.{event_type}.{span.name}"
        else:
            op_name = f"aiq.{event_type}"

        # Create input dictionary
        inputs = {}
        input_value = span.attributes.get(SpanAttributes.INPUT_VALUE.value)
        if input_value is not None:
            try:
                # Add the input to the Weave call
                inputs["input"] = input_value
            except Exception:
                # If serialization fails, use string representation
                inputs["input"] = str(input_value)

        # Create the Weave call
        call = self._gc.create_call(
            op_name,
            inputs=inputs,
            parent=parent_call,
            attributes=span.attributes,
            display_name=op_name,
        )

        # Store the call with span span ID as key
        self._weave_calls[span_id] = call

        # Store span ID for parent reference
        setattr(call, "span_id", span_id)

        return call

    def _finish_weave_call(self, call: Call, span: Span):
        """Finish a previously created Weave call.

        Args:
            call (Call): The Weave call to finish.
            span (Span): The span to finish the call for.
        """
        span_id = span.context.span_id if span.context else None  # type: ignore
        if span_id is None:
            logger.warning("Span has no context or span_id")
            return

        if call is None:
            logger.warning("No Weave call found for span %s", span_id)
            # Still remove from in-flight tracking
            self._in_flight_calls.discard(span_id)
            return

        # Check if this call was already finished by cleanup (race condition protection)
        if span_id not in self._weave_calls:
            logger.debug("Call for span %s was already finished (likely by cleanup)", span_id)
            self._in_flight_calls.discard(span_id)
            return

        # Create output dictionary
        outputs = {}
        output = span.attributes.get(SpanAttributes.OUTPUT_VALUE.value)
        if output is not None:
            try:
                # Add the output to the Weave call
                outputs["output"] = output
            except Exception:
                # If serialization fails, use string representation
                outputs["output"] = str(output)

        # Add usage information
        outputs["prompt_tokens"] = span.attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT.value)
        outputs["completion_tokens"] = span.attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION.value)
        outputs["total_tokens"] = span.attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL.value)
        outputs["num_llm_calls"] = span.attributes.get(SpanAttributes.AIQ_USAGE_NUM_LLM_CALLS.value)
        outputs["seconds_between_calls"] = span.attributes.get(SpanAttributes.AIQ_USAGE_SECONDS_BETWEEN_CALLS.value)

        try:
            # Finish the call with outputs
            self._gc.finish_call(call, outputs)
            logger.debug("Successfully finished call for span %s", span_id)
        except Exception as e:
            logger.warning("Error finishing call for span %s: %s", span_id, e)
        finally:
            # Always clean up tracking regardless of finish success/failure
            self._weave_calls.pop(span_id, None)
            self._in_flight_calls.discard(span_id)

    async def _cleanup_weave_calls(self) -> None:
        """Clean up any lingering unfinished Weave calls.

        This method should only be called during exporter shutdown to handle
        calls that weren't properly finished during normal operation.

        CRITICAL: Only cleans up calls that are NOT currently in-flight to prevent
        race conditions with background export tasks.
        """
        if self._gc is not None and self._weave_calls:
            # Only clean up calls that are not currently being processed
            abandoned_calls = {}
            for span_id, call in self._weave_calls.items():
                if span_id not in self._in_flight_calls:
                    abandoned_calls[span_id] = call

            if abandoned_calls:
                logger.debug("Cleaning up %d truly abandoned Weave calls (out of %d total)",
                             len(abandoned_calls),
                             len(self._weave_calls))

                for span_id, call in abandoned_calls.items():
                    try:
                        # Finish any remaining calls with incomplete status
                        self._gc.finish_call(call, {"status": "incomplete"})
                        logger.debug("Finished abandoned call for span %s", span_id)
                    except Exception as e:
                        logger.warning("Error finishing abandoned call for span %s: %s", span_id, e)
                    finally:
                        # Remove from tracking
                        self._weave_calls.pop(span_id, None)
            else:
                logger.debug("No abandoned calls to clean up (%d calls still in-flight)", len(self._in_flight_calls))
