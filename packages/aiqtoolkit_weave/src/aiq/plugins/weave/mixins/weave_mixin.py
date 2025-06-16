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
from contextlib import contextmanager

from weave.trace.context import weave_client_context
from weave.trace.context.call_context import get_current_call
from weave.trace.context.call_context import set_call_stack
from weave.trace.weave_client import Call

from aiq.observability.span.span import Span
from aiq.observability.span.span import SpanAttributes

logger = logging.getLogger(__name__)


class WeaveMixin:

    def __init__(self, *args, project: str, entity: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._gc = weave_client_context.require_weave_client()
        self._project = project
        self._entity = entity

    async def export(self, span: Span) -> None:
        """Export a batch of spans."""

        try:
            call = self._create_weave_call(span)
            self._finish_weave_call(call, span)
        except Exception as e:
            logger.error("Error exporting spans: %s", e, exc_info=True)

    @contextmanager
    def parent_call(self, trace_id: str, parent_call_id: str):
        dummy_call = Call(trace_id=trace_id, id=parent_call_id, _op_name="", project_id="", parent_id=None, inputs={})
        with set_call_stack([dummy_call]):
            yield

    def _create_weave_call(self, span: Span):
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
            parent_span_id = span.context.parent.span_id
            # Find the corresponding weave call for this parent span
            for call in self._weave_calls.values():
                if getattr(call, "span_id", None) == parent_span_id:
                    parent_call = call
                    break

        # Generate a meaningful operation name based on event type
        event_type = span.attributes.get("aiq.event_type").split(".")[-1]
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

        # Store span ID for parent reference
        setattr(call, "span_id", span.context.span_id)

        return call

    def _finish_weave_call(self, call: Call, span: Span):
        """
        Finish a previously created Weave call
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
