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

import asyncio
import copy
import importlib
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from collections.abc import Callable
from typing import Any

from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import TraceMetadata
from nat.data_models.intermediate_step import UsageInfo
from nat.profiler.callbacks.base_callback_class import BaseProfilerCallback
from nat.profiler.callbacks.token_usage_base_model import TokenUsageBaseModel

logger = logging.getLogger(__name__)


class StrandsToolInstrumentationHook:
    """Hook callbacks for instrumenting Strands tool invocations.

    This class provides callbacks for Strands' hooks API to
    capture tool execution events and emit proper TOOL_START/END spans.
    """

    def __init__(self, handler: 'StrandsProfilerHandler'):
        """Initialize the hook with a reference to the profiler handler.

        Args:
            handler: StrandsProfilerHandler instance that manages this hook
        """
        self.handler = handler
        self._tool_start_times: dict[str, float] = {}
        self._step_manager = Context.get().intermediate_step_manager

    def on_before_tool_invocation(self, event: Any) -> None:
        """Handle tool invocation start.

        Called by Strands before a tool is executed.
        Emits a TOOL_START span.

        Args:
            event: BeforeToolInvocationEvent from Strands
        """
        try:
            tool_use = event.tool_use
            selected_tool = event.selected_tool

            if not selected_tool:
                logger.debug("Tool hook: no selected_tool, skipping")
                return

            # Extract tool information
            tool_name, tool_use_id, tool_input = self._extract_tool_info(selected_tool, tool_use)

            # Store start time for duration calculation
            self._tool_start_times[tool_use_id] = time.time()

            step_manager = self._step_manager

            start_payload = IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_START,
                framework=LLMFrameworkEnum.STRANDS,
                name=tool_name,
                UUID=tool_use_id,
                data=StreamEventData(input=str(tool_input), output=""),
                metadata=TraceMetadata(
                    tool_inputs=copy.deepcopy(tool_input),
                    tool_info=copy.deepcopy(getattr(selected_tool, 'tool_spec', {})),
                ),
                usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
            )

            step_manager.push_intermediate_step(start_payload)

            logger.debug("TOOL_START: %s (ID: %s)", tool_name, tool_use_id)
        except Exception:  # noqa: BLE001
            logger.error("Error in before_tool_invocation")
            raise

    def on_after_tool_invocation(self, event: Any) -> None:
        """Handle tool invocation end.

        Called by Strands after a tool execution completes.
        Emits a TOOL_END span.

        Args:
            event: AfterToolInvocationEvent from Strands
        """
        try:
            tool_use = event.tool_use
            selected_tool = event.selected_tool
            result = event.result
            exception = event.exception

            if not selected_tool:
                logger.debug("Tool hook: no selected_tool, skipping")
                return

            # Extract tool information
            tool_name, tool_use_id, tool_input = self._extract_tool_info(selected_tool, tool_use)
            start_time = self._tool_start_times.pop(tool_use_id, time.time())

            # Extract output from result
            tool_output = ""
            if isinstance(result, dict):
                content = result.get('content', [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            tool_output += item['text']

            # Handle errors
            if exception:
                tool_output = f"Error: {exception}"

            # Use stored step_manager to avoid context isolation issues
            step_manager = self._step_manager

            end_payload = IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_END,
                span_event_timestamp=start_time,
                framework=LLMFrameworkEnum.STRANDS,
                name=tool_name,
                UUID=tool_use_id,
                metadata=TraceMetadata(tool_outputs=tool_output),
                usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                data=StreamEventData(input=str(tool_input), output=tool_output),
            )
            step_manager.push_intermediate_step(end_payload)

            logger.debug("TOOL_END: %s (ID: %s)", tool_name, tool_use_id)

        except Exception:  # noqa: BLE001
            logger.error("Failed to handle after_tool_invocation")
            raise

    def _extract_tool_info(self, selected_tool: Any, tool_use: dict) -> tuple[str, str, dict]:
        """Extract tool name, ID, and input from event.

        Args:
            selected_tool: The tool being invoked
            tool_use: Tool use dictionary from Strands event

        Returns:
            Tuple of (tool_name, tool_use_id, tool_input)
        """
        tool_name = getattr(selected_tool, 'tool_name', tool_use.get('name', 'unknown_tool'))
        tool_use_id = tool_use.get('toolUseId')
        if tool_use_id is None:
            logger.warning("Missing toolUseId in tool_use event, using 'unknown' fallback")
            tool_use_id = "unknown"
        tool_input = tool_use.get('input', {}) or {}
        return tool_name, tool_use_id, tool_input


class StrandsProfilerHandler(BaseProfilerCallback):

    def __init__(self) -> None:
        super().__init__()
        self._patched: bool = False
        self.last_call_ts = time.time()

        # Note: tool hooks are now created per-agent-instance in wrapped_init
        # to avoid shared state in concurrent execution

    def instrument(self) -> None:
        """
        Instrument Strands for telemetry capture.

        This patches:
        1. Model streaming methods (OpenAI/Bedrock) for LLM spans
        2. Agent.__init__ to auto-register tool hooks on Agent creation

        Tool instrumentation uses Strands' hooks API,
        which is automatically registered when an Agent is instantiated.
        """
        if self._patched:
            return

        try:
            # Patch LLM streaming methods
            OpenAIModel = None
            BedrockModel = None
            try:
                openai_mod = importlib.import_module("strands.models.openai")
                OpenAIModel = getattr(openai_mod, "OpenAIModel", None)
            except Exception:  # noqa: BLE001
                OpenAIModel = None

            try:
                bedrock_mod = importlib.import_module("strands.models.bedrock")
                BedrockModel = getattr(bedrock_mod, "BedrockModel", None)
            except Exception:  # noqa: BLE001
                BedrockModel = None

            to_patch: list[tuple[type, str]] = []
            if OpenAIModel is not None:
                for name in ("stream", "structured_output"):
                    if hasattr(OpenAIModel, name):
                        to_patch.append((OpenAIModel, name))
            if BedrockModel is not None:
                for name in ("stream", "structured_output"):
                    if hasattr(BedrockModel, name):
                        to_patch.append((BedrockModel, name))

            for cls, method_name in to_patch:
                original = getattr(cls, method_name)
                wrapped = self._wrap_stream_method(original)
                setattr(cls, method_name, wrapped)

            debug_targets = [f"{c.__name__}.{m}" for c, m in to_patch]
            logger.info(
                "StrandsProfilerHandler LLM instrumentation: %s",
                debug_targets,
            )

            # Patch Agent.__init__ to auto-register hooks
            self._instrument_agent_init()

            self._patched = True

        except Exception:  # noqa: BLE001
            logger.error("Failed to instrument Strands models")
            raise

    def _instrument_agent_init(self) -> None:
        """Patch Agent.__init__ to auto-register hooks on instantiation.

        This ensures that whenever a Strands Agent is created, our tool
        instrumentation hooks are automatically registered without requiring
        any user code changes.
        """
        try:
            # Import Agent class
            agent_mod = importlib.import_module("strands.agent.agent")
            Agent = getattr(agent_mod, "Agent", None)

            if Agent is None:
                logger.warning("Agent class not found in strands.agent.agent")
                return

            # Save reference to handler in closure
            handler = self

            # Save original __init__
            original_init = Agent.__init__

            def wrapped_init(agent_self, *args, **kwargs):
                """Wrapped Agent.__init__ that auto-registers hooks."""
                # Call original init
                original_init(agent_self, *args, **kwargs)

                # Auto-register tool hooks on this agent instance
                try:
                    # Import hook event types
                    # pylint: disable=import-outside-toplevel
                    from strands.hooks import AfterToolCallEvent
                    from strands.hooks import BeforeToolCallEvent

                    # Create a dedicated hook instance for this agent
                    agent_tool_hook = StrandsToolInstrumentationHook(handler)

                    # Register tool hooks on this agent instance
                    agent_self.hooks.add_callback(BeforeToolCallEvent, agent_tool_hook.on_before_tool_invocation)
                    agent_self.hooks.add_callback(AfterToolCallEvent, agent_tool_hook.on_after_tool_invocation)

                    logger.debug("Strands tool hooks registered on Agent instance")

                except Exception:  # noqa: BLE001
                    logger.exception("Failed to auto-register hooks")

            # Replace Agent.__init__ with wrapped version
            Agent.__init__ = wrapped_init

            logger.info("Strands Agent.__init__ instrumentation applied")

        except Exception:  # noqa: BLE001
            logger.exception("Failed to instrument Agent.__init__")

    def _extract_model_info(self, model_instance: Any) -> tuple[str, dict[str, Any]]:
        """Extract model name from Strands model instance."""
        model_name = ""

        for attr_name in ['config', 'client_args']:
            if hasattr(model_instance, attr_name):
                attr_value = getattr(model_instance, attr_name, None)
                if isinstance(attr_value, dict):
                    for key, val in attr_value.items():
                        if 'model' in key.lower() and val:
                            model_name = str(val)
                            break
                if model_name:
                    break

        return str(model_name), {}

    def _wrap_stream_method(self, original: Callable[..., Any]) -> Callable[..., Any]:
        # Capture handler reference in closure
        handler = self

        async def wrapped(model_self, *args, **kwargs) -> AsyncGenerator[Any, None]:  # type: ignore[override]
            """
            Wrapper for Strands model streaming that emits paired
            LLM_START/END spans with usage and metrics.
            """
            context = Context.get()
            step_manager = context.intermediate_step_manager

            event_uuid = str(uuid.uuid4())
            start_time = time.time()

            # Extract model info and parameters
            model_name, _ = handler._extract_model_info(model_self)

            # Extract messages from args (Strands passes as positional args)
            # Signature: stream(self, messages, tool_specs=None,
            #                   system_prompt=None, **kwargs)
            raw_messages = args[0] if args else []
            tool_specs = args[1] if len(args) > 1 else kwargs.get("tool_specs")
            system_prompt = (args[2] if len(args) > 2 else kwargs.get("system_prompt"))

            # Build chat_inputs with system prompt and messages
            all_messages = []
            if system_prompt:
                all_messages.append({"text": system_prompt, "role": "system"})
            if isinstance(raw_messages, list):
                all_messages.extend(copy.deepcopy(raw_messages))

            # Extract tools schema for metadata
            tools_schema = []
            if tool_specs and isinstance(tool_specs, list):
                try:
                    tools_schema = [{
                        "type": "function",
                        "function": {
                            "name": tool_spec.get("name", "unknown"),
                            "description": tool_spec.get("description", ""),
                            "parameters": tool_spec.get("inputSchema", {}).get("json", {})
                        }
                    } for tool_spec in tool_specs]
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to extract tools schema", exc_info=True)
                    tools_schema = []

            # Extract string representation of last user message for data.input
            # (full message history is in metadata.chat_inputs)
            llm_input_str = ""
            if all_messages:
                last_msg = all_messages[-1]
                if isinstance(last_msg, dict) and 'text' in last_msg:
                    llm_input_str = last_msg['text']
                elif isinstance(last_msg, dict):
                    llm_input_str = str(last_msg)
                else:
                    llm_input_str = str(last_msg)

            # Always emit START first (before streaming begins)
            start_payload = IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                framework=LLMFrameworkEnum.STRANDS,
                name=str(model_name),
                UUID=event_uuid,
                data=StreamEventData(input=llm_input_str, output=""),
                metadata=TraceMetadata(
                    chat_inputs=copy.deepcopy(all_messages),
                    tools_schema=copy.deepcopy(tools_schema),
                ),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(),
                    num_llm_calls=1,
                    seconds_between_calls=int(time.time() - self.last_call_ts),
                ),
            )
            step_manager.push_intermediate_step(start_payload)
            self.last_call_ts = time.time()

            # Collect output text, tool calls, and token usage while streaming
            output_text = ""
            tool_calls = []  # List of tool calls made by the LLM
            current_tool_call = None  # Currently accumulating tool call
            token_usage = TokenUsageBaseModel()
            ended: bool = False

            def _push_end_if_needed() -> None:
                nonlocal ended
                if ended:
                    return

                # Determine the output to show in the span
                # If there are tool calls, format them as the output
                # Otherwise, use the text response
                if tool_calls:
                    # Format tool calls as readable output
                    tool_call_strs = []
                    for tc in tool_calls:
                        tool_name = tc.get('name', 'unknown')
                        tool_input = tc.get('input', {})
                        tool_call_strs.append(f"Tool: {tool_name}\nInput: {tool_input}")
                    output_content = "\n\n".join(tool_call_strs)
                else:
                    output_content = output_text

                chat_responses_list = []
                if output_content:
                    chat_responses_list = [output_content]

                # Build metadata with standard NAT structure
                metadata = TraceMetadata(
                    chat_responses=chat_responses_list,
                    chat_inputs=all_messages,
                    tools_schema=copy.deepcopy(tools_schema),
                )

                # Push END with input/output and token usage
                end_payload = IntermediateStepPayload(
                    event_type=IntermediateStepType.LLM_END,
                    span_event_timestamp=start_time,
                    framework=LLMFrameworkEnum.STRANDS,
                    name=str(model_name),
                    UUID=event_uuid,
                    data=StreamEventData(input=llm_input_str, output=output_content),
                    usage_info=UsageInfo(token_usage=token_usage, num_llm_calls=1),
                    metadata=metadata,
                )
                step_manager.push_intermediate_step(end_payload)
                ended = True

            try:
                agen = original(model_self, *args, **kwargs)
                if hasattr(agen, "__aiter__"):
                    async for ev in agen:  # type: ignore
                        try:
                            # Extract text content
                            text_content = self._extract_text_from_event(ev)
                            if text_content:
                                output_text += text_content

                            # Extract tool call information
                            tool_call_info = self._extract_tool_call_from_event(ev)
                            if tool_call_info:
                                if "name" in tool_call_info:
                                    # New tool call starting
                                    if current_tool_call:
                                        # Finalize and save previous tool call
                                        self._finalize_tool_call(current_tool_call)
                                        tool_calls.append(current_tool_call)
                                    current_tool_call = tool_call_info
                                elif "input_chunk" in tool_call_info and current_tool_call:
                                    # Accumulate input JSON string chunks
                                    current_tool_call["input_str"] += tool_call_info["input_chunk"]

                            # Check for contentBlockStop to finalize current tool call
                            if "contentBlockStop" in ev and current_tool_call:
                                self._finalize_tool_call(current_tool_call)
                                tool_calls.append(current_tool_call)
                                current_tool_call = None

                            # Extract usage information (but don't push END yet - wait for all text)
                            usage_info = self._extract_usage_from_event(ev)
                            if usage_info:
                                token_usage = TokenUsageBaseModel(**usage_info)

                        except Exception:  # noqa: BLE001
                            logger.debug("Failed to extract streaming fields from event", exc_info=True)
                        yield ev
                else:
                    # Non-async generator fallback
                    res = agen
                    if asyncio.iscoroutine(res):
                        res = await res  # type: ignore[func-returns-value]
                    yield res
            finally:
                # Ensure END is always pushed
                _push_end_if_needed()

        return wrapped

    def _extract_text_from_event(self, ev: dict) -> str:
        """Extract text content from a Strands event.

        Args:
            ev: Event dictionary from Strands stream

        Returns:
            Extracted text content or empty string
        """
        if not isinstance(ev, dict):
            return ""

        # Try multiple possible locations for text content
        if "data" in ev:
            return str(ev["data"])

        # Check for Strands contentBlockDelta structure (for streaming text responses)
        if "contentBlockDelta" in ev and isinstance(ev["contentBlockDelta"], dict):
            delta = ev["contentBlockDelta"].get("delta", {})
            if isinstance(delta, dict) and "text" in delta:
                return str(delta["text"])

        # Check for other common text fields
        if "content" in ev:
            return str(ev["content"])

        if "text" in ev:
            return str(ev["text"])

        # Check for nested content
        if "message" in ev and isinstance(ev["message"], dict):
            if "content" in ev["message"]:
                return str(ev["message"]["content"])

        return ""

    def _finalize_tool_call(self, tool_call: dict[str, Any]) -> None:
        """Parse the accumulated input_str JSON and store in the input field.

        Args:
            tool_call: Tool call dictionary with input_str to parse
        """
        input_str = tool_call.get("input_str", "")
        if input_str:
            try:
                tool_call["input"] = json.loads(input_str)
            except (json.JSONDecodeError, ValueError):
                logger.debug("Failed to parse tool input JSON: %s", input_str)
                tool_call["input"] = {"raw": input_str}
        # Remove the temporary input_str field
        tool_call.pop("input_str", None)

    def _extract_tool_call_from_event(self, ev: dict) -> dict[str, Any] | None:
        """Extract tool call information from a Strands event.

        Args:
            ev: Event dictionary from Strands stream

        Returns:
            Dictionary with tool call info (name, input_chunk) or None if not a tool call
        """
        if not isinstance(ev, dict):
            return None

        # Check for contentBlockStart with toolUse
        if "contentBlockStart" in ev:
            start = ev["contentBlockStart"].get("start", {})
            if isinstance(start, dict) and "toolUse" in start:
                tool_use = start["toolUse"]
                return {
                    "name": tool_use.get("name", "unknown"),
                    "id": tool_use.get("toolUseId", "unknown"),
                    "input_str": "",  # Will accumulate JSON string chunks
                    "input": {}  # Will be parsed at the end
                }

        # Check for contentBlockDelta with toolUse input (streaming chunks)
        if "contentBlockDelta" in ev:
            delta = ev["contentBlockDelta"].get("delta", {})
            if isinstance(delta, dict) and "toolUse" in delta:
                tool_use_delta = delta["toolUse"]
                input_chunk = tool_use_delta.get("input", "")
                if input_chunk:
                    # Return the chunk to be accumulated
                    return {"input_chunk": input_chunk}

        return None

    def _extract_usage_from_event(self, ev: dict) -> dict[str, int] | None:
        """Extract usage information from a Strands event.

        Args:
            ev: Event dictionary from Strands stream

        Returns:
            Dictionary with token usage info or None if not found
        """
        if not isinstance(ev, dict):
            return None

        md = ev.get("metadata")
        if not isinstance(md, dict):
            return None

        usage = md.get("usage")
        if not isinstance(usage, dict):
            return None

        try:
            return {
                "prompt_tokens": int(usage.get("inputTokens") or 0),
                "completion_tokens": int(usage.get("outputTokens") or 0),
                "total_tokens": int(usage.get("totalTokens") or 0),
            }
        except (ValueError, TypeError):
            return None
