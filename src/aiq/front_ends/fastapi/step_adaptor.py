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

import html
import logging
from functools import reduce
from textwrap import dedent

from aiq.data_models.api_server import AIQResponseIntermediateStep
from aiq.data_models.api_server import AIQResponseSerializable
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepCategory
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepType
from aiq.data_models.invocation_node import InvocationNode
from aiq.data_models.step_adaptor import StepAdaptorConfig
from aiq.data_models.step_adaptor import StepAdaptorMode
from aiq.utils.type_utils import is_valid_json

logger = logging.getLogger(__name__)


class StepAdaptor:

    def __init__(self, config: StepAdaptorConfig):

        self._history: list[IntermediateStep] = []
        self.config = config

    def _step_matches_filter(self, step: IntermediateStep, config: StepAdaptorConfig) -> bool:
        """
        Returns True if this intermediate step should be included (based on the config.mode).
        """

        if config.mode == StepAdaptorMode.OFF:
            return False

        if config.mode == StepAdaptorMode.DEFAULT:
            # default existing behavior: show LLM events + TOOL_END + FUNCTION events
            if step.event_category == IntermediateStepCategory.LLM:
                return True
            if step.event_category == IntermediateStepCategory.TOOL:
                return True
            if step.event_category == IntermediateStepCategory.FUNCTION:
                return True
            return False

        if config.mode == StepAdaptorMode.CUSTOM:
            # pass only what the user explicitly listed
            return step.event_type in config.custom_event_types

        return False

    def _handle_llm(self, step: IntermediateStepPayload, ancestry: InvocationNode) -> AIQResponseSerializable | None:
        input_str: str | None = None
        output_str: str | None = None

        # Find the start in the history with matching run_id
        start_step = next(
            (x for x in self._history if x.event_type == IntermediateStepType.LLM_START and x.UUID == step.UUID), None)

        if not start_step:
            # If we don't have a start step, we can't do anything
            return None

        input_str = str(start_step.data.input)

        if step.event_type == IntermediateStepType.LLM_NEW_TOKEN:

            # Find all of the previous LLM chunks and concatenate them
            output_str = reduce(
                lambda x, y: x + y,
                (str(x.data.chunk)
                 for x in self._history if x.event_type == IntermediateStepType.LLM_NEW_TOKEN and x.UUID == step.UUID),
                "")

        elif step.event_type == IntermediateStepType.LLM_END:
            output_str = str(step.data.output)

        if not input_str and not output_str:
            return None

        escaped_input = html.escape(input_str, quote=False)

        # Dont use f-strings here because the payload is markdown and screws up the dedent
        payload = dedent("""
        **Input:**
        ```python
        {input_value}
        ```
        """).strip("\n").format(input_value=escaped_input)

        if (output_str):
            escaped_output = html.escape(output_str, quote=False) if output_str else ""

            # Dont use f-strings here because the payload is markdown and screws up the dedent
            payload = dedent("""
            {payload}

            **Output:**
            {output_value}
            """).strip("\n").format(payload=payload, output_value=escaped_output)

        event = AIQResponseIntermediateStep(id=step.UUID,
                                            name=step.name or "",
                                            payload=payload,
                                            parent_id=ancestry.function_id)

        return event

    def _handle_tool(self, step: IntermediateStepPayload, ancestry: InvocationNode) -> AIQResponseSerializable | None:
        """
        Handles both TOOL_START and TOOL_END events
        """
        input_str: str | None = None
        output_str: str | None = None

        # Find the start in the history with matching run_id
        start_step = next(
            (x for x in self._history if x.event_type == IntermediateStepType.TOOL_START and x.UUID == step.UUID), None)

        if not start_step:
            # If we don't have a start step, we can't do anything
            return None

        input_str = str(start_step.data.input)

        if step.event_type == IntermediateStepType.TOOL_END:
            output_str = str(step.data.output)

        if not input_str and not output_str:
            return None

        escaped_input = html.escape(input_str, quote=False)
        format_input_type = "json" if is_valid_json(escaped_input) else "python"

        # Dont use f-strings here because the payload is markdown and screws up the dedent
        payload = dedent("""
        **Input:**
        ```{format_input_type}
        {input_value}
        ```
        """).strip("\n").format(input_value=escaped_input, format_input_type=format_input_type)

        if output_str:
            escaped_output = html.escape(output_str, quote=False)
            format_output_type = "json" if is_valid_json(escaped_output) else "python"

            # Dont use f-strings here because the payload is markdown and screws up the dedent
            payload = dedent("""
            {payload}

            **Output:**
            ```{format_output_type}
            {output_value}
            ```
            """).strip("\n").format(payload=payload, output_value=escaped_output, format_output_type=format_output_type)

        event = AIQResponseIntermediateStep(id=step.UUID,
                                            name=f"Tool: {step.name}",
                                            payload=payload,
                                            parent_id=ancestry.function_id)

        return event

    def _handle_function(self, step: IntermediateStepPayload,
                         ancestry: InvocationNode) -> AIQResponseSerializable | None:
        """
        Handles the FUNCTION_START and FUNCTION_END events
        """
        input_str: str | None = None
        output_str: str | None = None

        if step.event_type == IntermediateStepType.FUNCTION_START:
            # For function start events, display input data
            if step.data and hasattr(step.data, 'input'):
                input_str = str(step.data.input)
            elif step.data:
                input_str = str(step.data)

            if not input_str:
                return None

            escaped_input = html.escape(input_str, quote=False)
            format_input_type = "json" if is_valid_json(escaped_input) else "python"

            # Create payload for function start
            payload_str = dedent("""
            **Function Input:**
            ```{format_input_type}
            {input_value}
            ```
            """).strip("\n").format(input_value=escaped_input, format_input_type=format_input_type)

            event = AIQResponseIntermediateStep(id=step.UUID,
                                                name=f"Function Start: {step.name}",
                                                payload=payload_str,
                                                parent_id=ancestry.parent_id)
            return event

        if step.event_type == IntermediateStepType.FUNCTION_END:
            # Find the start event with matching UUID
            start_step = next(
                (x
                 for x in self._history if x.event_type == IntermediateStepType.FUNCTION_START and x.UUID == step.UUID),
                None)

            # For function end events, display output data
            if step.data and hasattr(step.data, 'output'):
                output_str = str(step.data.output)
            elif step.data:
                output_str = str(step.data)

            if not output_str:
                return None

            escaped_output = html.escape(output_str, quote=False)
            format_output_type = "json" if is_valid_json(escaped_output) else "python"

            # Get input from start step if available
            input_payload = ""
            if start_step and start_step.data:
                if hasattr(start_step.data, 'input'):
                    input_str = str(start_step.data.input)
                else:
                    input_str = str(start_step.data)

                if input_str:
                    escaped_input = html.escape(input_str, quote=False)
                    format_input_type = "json" if is_valid_json(escaped_input) else "python"
                    input_payload = dedent("""
                    **Function Input:**
                    ```{format_input_type}
                    {input_value}
                    ```
                    """).strip("\n").format(input_value=escaped_input, format_input_type=format_input_type)

            # Create payload for function end
            payload_str = dedent("""
            {input_payload}**Function Output:**
            ```{format_output_type}
            {output_value}
            ```
            """).strip("\n").format(input_payload=input_payload,
                                    output_value=escaped_output,
                                    format_output_type=format_output_type)

            event = AIQResponseIntermediateStep(id=step.UUID,
                                                name=f"Function Complete: {step.name}",
                                                payload=payload_str,
                                                parent_id=ancestry.parent_id)
            return event

        return None

    def _handle_custom(self, payload: IntermediateStepPayload,
                       ancestry: InvocationNode) -> AIQResponseSerializable | None:
        """
        Handles the CUSTOM event
        """
        escaped_payload = html.escape(str(payload), quote=False)
        escaped_payload = escaped_payload.replace("\n", "")

        # Attempt to determine type
        format_type = "json" if is_valid_json(escaped_payload) else "python"

        # Don't use f-strings here because the payload is markdown and screws up the dedent
        payload_str = dedent("""
        ```{format_type}
        {payload}
        ```
        """).strip("\n").format(payload=escaped_payload, format_type=format_type)

        # Return the event
        event = AIQResponseIntermediateStep(id=payload.UUID,
                                            name=f"{payload.event_type}",
                                            payload=payload_str,
                                            parent_id=ancestry.function_id)

        return event

    def process(self, step: IntermediateStep) -> AIQResponseSerializable | None:  # pylint: disable=R1710

        # Track the chunk
        self._history.append(step)
        payload = step.payload
        ancestry = step.function_ancestry

        if not self._step_matches_filter(step, self.config):
            return None

        try:

            if step.event_category == IntermediateStepCategory.LLM:
                return self._handle_llm(payload, ancestry)

            if step.event_category == IntermediateStepCategory.TOOL:
                return self._handle_tool(payload, ancestry)

            if step.event_category == IntermediateStepCategory.FUNCTION:
                return self._handle_function(payload, ancestry)

            if step.event_category == IntermediateStepCategory.CUSTOM:
                return self._handle_custom(payload, ancestry)

        except Exception as e:
            logger.error("Error processing intermediate step: %s", e, exc_info=True)

        return None
