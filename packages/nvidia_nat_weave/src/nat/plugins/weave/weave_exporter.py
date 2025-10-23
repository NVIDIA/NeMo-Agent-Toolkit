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
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.span import Span
from nat.observability.exporter.base_exporter import IsolatedAttribute
from nat.observability.exporter.span_exporter import SpanExporter
from nat.utils.log_utils import LogFilter
from nat.utils.string_utils import truncate_string
from nat.utils.type_utils import override
from weave.trace.context import weave_client_context
from weave.trace.context.call_context import get_current_call
from weave.trace.context.call_context import set_call_stack
from weave.trace.weave_client import Call

logger = logging.getLogger(__name__)

# Use LogFilter to filter out specific message patterns
presidio_filter = LogFilter([
    "nlp_engine not provided",
    "Created NLP engine",
    "registry not provided",
    "Loaded recognizer",
    "Recognizer not added to registry"
])


class WeaveExporter(SpanExporter[Span, Span]):
    """A Weave exporter that exports telemetry traces to Weights & Biases Weave using OpenTelemetry."""

    _weave_calls: IsolatedAttribute[dict[str, Call]] = IsolatedAttribute(dict)

    def __init__(self,
                 context_state=None,
                 entity: str | None = None,
                 project: str | None = None,
                 verbose: bool = False,
                 attributes: dict[str, Any] | None = None):
        super().__init__(context_state=context_state)
        self._entity = entity
        self._project = project
        self._attributes = attributes or {}
        self._gc = weave_client_context.require_weave_client()

        # Optionally, set log filtering for presidio-analyzer to reduce verbosity
        if not verbose:
            presidio_logger = logging.getLogger('presidio-analyzer')
            presidio_logger.addFilter(presidio_filter)

    @override
    async def export_processed(self, item: Span | list[Span]) -> None:
        """Dummy implementation of export_processed.

        Args:
            item (Span | list[Span]): The span or list of spans to export.
        """
        pass

    def _process_start_event(self, event: IntermediateStep):
        """Process the start event for a Weave call.

        Args:
            event (IntermediateStep): The intermediate step event.
        """
        super()._process_start_event(event)
        span = self._span_stack.get(event.UUID, None)
        if span is None:
            logger.warning("No span found for event %s", event.UUID)
            return
        self._create_weave_call(event, span)

    def _process_end_event(self, event: IntermediateStep):
        """Process the end event for a Weave call.

        Args:
            event (IntermediateStep): The intermediate step event.
        """
        super()._process_end_event(event)
        self._finish_weave_call(event)

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

    def _create_weave_call(self, step: IntermediateStep, span: Span) -> Call:
        """
        Create a Weave call directly from the span and step data,
        connecting to existing framework traces if available.

        Args:
            step (IntermediateStep): The intermediate step event.
            span (Span): The span associated with the intermediate step.

        Returns:
            Call: The Weave call created from the span and step data.
        """
        # Check for existing Weave trace/call
        existing_call = get_current_call()

        # Extract parent call if applicable
        parent_call = None

        # If we have an existing Weave call from another framework (e.g., LangChain/LangGraph),
        # use it as the parent
        if existing_call is not None:
            parent_call = existing_call
            logger.debug("Found existing Weave call: %s from trace: %s", existing_call.id, existing_call.trace_id)
        # Otherwise, check our internal stack for parent relationships
        elif len(self._weave_calls) > 0 and len(self._span_stack) > 1:
            # Get the parent span using stack position (one level up)
            parent_span_id = self._span_stack[-2].context.span_id
            # Find the corresponding weave call for this parent span
            for call in self._weave_calls.values():
                if getattr(call, "span_id", None) == parent_span_id:
                    parent_call = call
                    break

        # Generate a meaningful operation name based on event type
        event_type = step.payload.event_type.split(".")[-1]
        if step.payload.name:
            op_name = f"nat.{event_type}.{step.payload.name}"
        else:
            op_name = f"nat.{event_type}"

        # Create input dictionary
        inputs = {}
        if step.payload.data and step.payload.data.input is not None:
            try:
                # Add the input to the Weave call
                inputs["input"] = step.payload.data.input
                self._extract_input_message(step.payload.data.input, inputs)
            except Exception:
                # If serialization fails, use string representation
                inputs["input"] = str(step.payload.data.input)

        # Create the Weave call
        attributes = span.attributes.copy()
        attributes.update(self._attributes)

        call = self._gc.create_call(
            op_name,
            inputs=inputs,
            parent=parent_call,
            attributes=attributes,
            display_name=op_name,
        )

        # Store the call with step UUID as key
        self._weave_calls[step.UUID] = call

        # Store span ID for parent reference
        if span.context is not None:
            setattr(call, "span_id", span.context.span_id)
        else:
            logger.warning("Span has no context, skipping span_id setting")

        return call

    def _extract_input_message(self, input_data: Any, inputs: dict[str, Any]) -> None:
        """
        Extract message content from input data and add to inputs dictionary.
        Also handles websocket mode where message is located at messages[0].content[0].text.

        Args:
            input_data: The raw input data from the request
            inputs: Dictionary to populate with extracted message content
        """
        # Extract message content if input has messages attribute
        messages = getattr(input_data, 'messages', [])
        if messages:
            content = messages[0].content
            if isinstance(content, list) and content:
                inputs["input_message"] = getattr(content[0], 'text', content[0])
            else:
                inputs["input_message"] = content

    def _extract_output_message(self, output_data: Any, outputs: dict[str, Any]) -> None:
        """
        Extract message content from various response formats and add a preview to the outputs dictionary.
        No data is added to the outputs dictionary if the output format is not supported.

        Supported output formats for message content include:

        - output.choices[0].message.content     /chat endpoint
        - output.value                          /generate endpoint
        - output[0].choices[0].message.content  chat WS schema
        - output[0].choices[0].delta.content    chat_stream WS schema, /chat/stream endpoint
        - output[0].value                       generate & generate_stream WS schema, /generate/stream endpoint

        Args:
            output_data: The raw output data from the response
            outputs: Dictionary to populate with extracted message content.
        """
        # Handle choices-keyed output object for /chat completion endpoint
        choices = getattr(output_data, 'choices', None)
        if choices:
            outputs["output_message"] = truncate_string(choices[0].message.content)
            return

        # Handle value-keyed output object for union types common for /generate completion endpoint
        value = getattr(output_data, 'value', None)
        if value:
            outputs["output_message"] = truncate_string(value)
            return

        # Handle list-based outputs (streaming or websocket)
        if not isinstance(output_data, list) or not output_data:
            return

        choices = getattr(output_data[0], 'choices', None)
        if choices:
            # chat websocket schema
            message = getattr(choices[0], 'message', None)
            if message:
                outputs["output_message"] = truncate_string(getattr(message, 'content', None))
                return

            # chat_stream websocket schema and /chat/stream completion endpoint
            delta = getattr(choices[0], 'delta', None)
            if delta:
                outputs["output_preview"] = truncate_string(getattr(delta, 'content', None))
                return

        # generate & generate_stream websocket schema, and /generate/stream completion endpoint
        value = getattr(output_data[0], 'value', None)
        if value:
            outputs["output_preview"] = truncate_string(str(value))

    def _finish_weave_call(self, step: IntermediateStep) -> None:
        """
        Finish a previously created Weave call.

        Args:
            step (IntermediateStep): The intermediate step event.
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
                self._extract_output_message(step.payload.data.output, outputs)
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
        self._gc.finish_call(call, outputs)

    async def _cleanup_weave_calls(self) -> None:
        """
        Clean up any lingering Weave calls.
        """
        if self._weave_calls:
            for _, call in list(self._weave_calls.items()):
                self._gc.finish_call(call, {"status": "incomplete"})
            self._weave_calls.clear()

    async def _cleanup(self) -> None:
        """Perform cleanup once the exporter is stopped."""
        await self._cleanup_weave_calls()
        await super()._cleanup()
