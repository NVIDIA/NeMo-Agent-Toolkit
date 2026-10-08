# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from textwrap import dedent

from nat.data_models.api_server import ResponseIntermediateStep
from nat.data_models.api_server import ResponseSerializable
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepCategory
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.step_adaptor import StepAdaptorConfig
from nat.data_models.step_adaptor import StepAdaptorMode
from nat.utils.type_utils import is_valid_json

logger = logging.getLogger(__name__)


class StepAdaptor:

    def __init__(self, config: StepAdaptorConfig):
        """
        Initializes the ``StepAdaptor`` with configuration settings.

        Args:
            config (StepAdaptorConfig): The configuration governing event filtering and payload truncation.
        """
        self._history: list[IntermediateStep] = []
        self._llm_tokens: dict[str, str] = {}
        self._llm_token_counts: dict[str, int] = {}
        self._llm_inputs: dict[str, str] = {}
        self.config = config

    def _clear_llm_state(self, uuid: str) -> None:
        """
        Removes buffered streaming state for a completed LLM run.

        Args:
            uuid (str): The intermediate step UUID to clear state for.
        """
        self._llm_tokens.pop(uuid, None)
        self._llm_token_counts.pop(uuid, None)
        self._llm_inputs.pop(uuid, None)

    def _truncate_text(self, text: str | None, max_len: int) -> str:
        """
        Truncates text if it exceeds ``max_len``, appending a truncation notice.

        Args:
            text (str | None): The text to truncate.
            max_len (int): The maximum character length allowed.

        Returns:
            str: The truncated text or empty string if input is None or empty.
        """
        if not text:
            return ""
        if max_len <= 0:
            return ""
        if len(text) <= max_len:
            return text

        placeholder = f"\n... [truncated {len(text)} characters]"
        if max_len > len(placeholder):
            slice_len = max_len - len(placeholder)
            return f"{text[:slice_len]}\n... [truncated {len(text) - slice_len} characters]"
        return text[:max_len]

    def _format_streamed_output(self, uuid: str) -> str:
        """
        Formats the accumulated streaming tokens for a step UUID with truncation.

        Args:
            uuid (str): The intermediate step UUID.

        Returns:
            str: The truncated streaming output string.
        """
        buf = self._llm_tokens.get(uuid, "")
        total_chars = self._llm_token_counts.get(uuid, 0)
        max_len = self.config.max_output_length

        if max_len <= 0:
            return ""
        if total_chars <= max_len:
            return buf

        placeholder = f"\n... [truncated {total_chars} characters]"
        if max_len > len(placeholder):
            slice_len = max_len - len(placeholder)
            return f"{buf[:slice_len]}\n... [truncated {total_chars - slice_len} characters]"
        return buf[:max_len]

    def _step_matches_filter(self, step: IntermediateStep, config: StepAdaptorConfig) -> bool:
        """
        Determines if an intermediate step should be included based on ``config.mode``.

        Args:
            step (IntermediateStep): The intermediate step to evaluate.
            config (StepAdaptorConfig): The current adaptor configuration.

        Returns:
            bool: ``True`` if the step should be processed, ``False`` otherwise.
        """
        if config.mode == StepAdaptorMode.OFF:
            return False

        if step.event_type == IntermediateStepType.LLM_NEW_TOKEN and not config.stream_llm_tokens:
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

    def _handle_llm(self, step: IntermediateStepPayload, ancestry: InvocationNode) -> ResponseSerializable | None:
        """
        Handles ``LLM_START``, ``LLM_NEW_TOKEN``, and ``LLM_END`` events.

        Args:
            step (IntermediateStepPayload): The intermediate step payload.
            ancestry (InvocationNode): The invocation node representing the ancestry hierarchy.

        Returns:
            ResponseSerializable | None: The formatted intermediate step response, or ``None`` if skipped.
        """
        input_str: str | None = None
        output_str: str | None = None

        if step.event_type == IntermediateStepType.LLM_START:
            if hasattr(step.data, "input") and step.data.input is not None:
                input_str = str(step.data.input)
            elif step.data is not None:
                input_str = str(step.data)
            else:
                input_str = ""

            if input_str:
                input_str = self._truncate_text(input_str, self.config.max_input_length)
            self._llm_inputs[step.UUID] = input_str
        else:
            input_str = self._llm_inputs.get(step.UUID, "")

        if step.event_type == IntermediateStepType.LLM_NEW_TOKEN:
            output_str = self._format_streamed_output(step.UUID)

        elif step.event_type == IntermediateStepType.LLM_END:
            if hasattr(step.data, "output") and step.data.output is not None and str(step.data.output).strip() != "":
                output_str = self._truncate_text(str(step.data.output), self.config.max_output_length)
            else:
                output_str = self._format_streamed_output(step.UUID)
            self._clear_llm_state(step.UUID)

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

        event = ResponseIntermediateStep(id=step.UUID,
                                         name=step.name or "",
                                         payload=payload,
                                         parent_id=ancestry.function_id)

        return event

    def _handle_tool(self, step: IntermediateStepPayload, ancestry: InvocationNode) -> ResponseSerializable | None:
        """
        Handles both ``TOOL_START`` and ``TOOL_END`` events.

        Args:
            step (IntermediateStepPayload): The intermediate step payload.
            ancestry (InvocationNode): The invocation node representing the ancestry hierarchy.

        Returns:
            ResponseSerializable | None: The formatted intermediate step response, or ``None`` if skipped.
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
        if input_str:
            input_str = self._truncate_text(input_str, self.config.max_input_length)

        if step.event_type == IntermediateStepType.TOOL_END:
            output_str = str(step.data.output)
            if output_str:
                output_str = self._truncate_text(output_str, self.config.max_output_length)

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

        event = ResponseIntermediateStep(id=step.UUID,
                                         name=f"Tool: {step.name}",
                                         payload=payload,
                                         parent_id=ancestry.function_id)

        return event

    def _handle_function(self, step: IntermediateStepPayload, ancestry: InvocationNode) -> ResponseSerializable | None:
        """
        Handles the ``FUNCTION_START`` and ``FUNCTION_END`` events.

        Args:
            step (IntermediateStepPayload): The intermediate step payload.
            ancestry (InvocationNode): The invocation node representing the ancestry hierarchy.

        Returns:
            ResponseSerializable | None: The formatted intermediate step response, or ``None`` if skipped.
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

            input_str = self._truncate_text(input_str, self.config.max_input_length)
            escaped_input = html.escape(input_str, quote=False)
            format_input_type = "json" if is_valid_json(escaped_input) else "python"

            # Create payload for function start
            payload_str = dedent("""
            **Function Input:**
            ```{format_input_type}
            {input_value}
            ```
            """).strip("\n").format(input_value=escaped_input, format_input_type=format_input_type)

            event = ResponseIntermediateStep(id=step.UUID,
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

            output_str = self._truncate_text(output_str, self.config.max_output_length)
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
                    input_str = self._truncate_text(input_str, self.config.max_input_length)
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

            event = ResponseIntermediateStep(id=step.UUID,
                                             name=f"Function Complete: {step.name}",
                                             payload=payload_str,
                                             parent_id=ancestry.parent_id)
            return event

        return None

    def _handle_custom(self, payload: IntermediateStepPayload, ancestry: InvocationNode) -> ResponseSerializable | None:
        """
        Handles the ``CUSTOM`` event.

        Args:
            payload (IntermediateStepPayload): The intermediate step payload.
            ancestry (InvocationNode): The invocation node representing the ancestry hierarchy.

        Returns:
            ResponseSerializable | None: The formatted intermediate step response, or ``None`` if skipped.
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
        event = ResponseIntermediateStep(id=payload.UUID,
                                         name=f"{payload.event_type}",
                                         payload=payload_str,
                                         parent_id=ancestry.function_id)

        return event

    def process(self, step: IntermediateStep) -> ResponseSerializable | None:
        """
        Processes an intermediate step and returns a serialized response if matched.

        Args:
            step (IntermediateStep): The intermediate step event to process.

        Returns:
            ResponseSerializable | None: The adapted response model if matched and processed,
                or ``None`` if filtered out or an error occurred.
        """
        # Track the chunk if not a streaming token event
        if step.event_type != IntermediateStepType.LLM_NEW_TOKEN:
            self._history.append(step)
        payload = step.payload
        ancestry = step.function_ancestry

        if step.event_type == IntermediateStepType.LLM_NEW_TOKEN:
            chunk_str: str | None = None
            if step.data:
                if hasattr(step.data, "chunk") and step.data.chunk is not None:
                    chunk_str = str(step.data.chunk)
                elif hasattr(step.data, "output") and step.data.output is not None:
                    chunk_str = str(step.data.output)
                elif hasattr(step.data, "payload") and step.data.payload is not None:
                    chunk_str = str(step.data.payload)
                elif not hasattr(step.data, "__dict__") and step.data is not None:
                    chunk_str = str(step.data)

            if chunk_str:
                self._llm_token_counts[step.UUID] = self._llm_token_counts.get(step.UUID, 0) + len(chunk_str)
                current_buf = self._llm_tokens.get(step.UUID, "")
                max_len = self.config.max_output_length
                if len(current_buf) < max_len:
                    remaining = max_len - len(current_buf)
                    self._llm_tokens[step.UUID] = current_buf + chunk_str[:remaining]

        if not self._step_matches_filter(step, self.config):
            if step.event_type == IntermediateStepType.LLM_END:
                self._clear_llm_state(step.UUID)
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
            logger.exception("Error processing intermediate step: %s", e)
        finally:
            if step.event_type == IntermediateStepType.LLM_END:
                self._clear_llm_state(step.UUID)

        return None
