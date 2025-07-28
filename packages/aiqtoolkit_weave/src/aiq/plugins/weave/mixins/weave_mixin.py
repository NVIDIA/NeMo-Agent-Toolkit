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

logger = logging.getLogger(__name__)


class WeaveMixin:
    """Mixin for Weave exporters.

    This mixin provides a default implementation of the export method for Weave exporters.
    It uses the weave_client_context to create and finish Weave calls.

    Args:
        project (str): The project name to group the telemetry traces.
        entity (str | None): The entity name to group the telemetry traces.
    """

    def __init__(self, *args, project: str, entity: str | None = None, **kwargs):
        """Initialize the Weave exporter with the specified project and entity.

        Args:
            project (str): The project name to group the telemetry traces.
            entity (str | None): The entity name to group the telemetry traces.
        """
        self._gc = weave_client_context.require_weave_client()
        self._project = project
        self._entity = entity
        self._weave_calls = {}
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
        try:
            call = self._create_weave_call(span)
            self._finish_weave_call(call, span)
        except Exception as e:
            logger.error("Error exporting spans: %s", e, exc_info=True)

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
        elif len(self._weave_calls) > 0:
            # Get the parent span using stack position (one level up)
            parent_span_id = span.parent.context.span_id  # type: ignore
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
        self._weave_calls[span.context.span_id] = call  # type: ignore

        # Store span ID for parent reference
        setattr(call, "span_id", span.context.span_id)  # type: ignore

        return call

    def _finish_weave_call(self, call: Call, span: Span):
        """Finish a previously created Weave call.

        Args:
            call (Call): The Weave call to finish.
            span (Span): The span to finish the call for.
        """

        if call is None:
            logger.warning("No Weave call found for span %s", span.context.span_id)
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

        # Finish the call with outputs
        self._gc.finish_call(call, outputs)

    async def _cleanup_weave_calls(self) -> None:
        # Clean up any lingering Weave calls if Weave is available and initialized
        if self._gc is not None and self._weave_calls:
            for _, call in list(self._weave_calls.items()):
                self._gc.finish_call(call, {"status": "incomplete"})
            self._weave_calls.clear()
