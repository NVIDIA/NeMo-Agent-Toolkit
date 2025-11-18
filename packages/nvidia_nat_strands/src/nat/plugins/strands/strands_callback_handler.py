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

    This class provides callbacks for Strands' experimental hooks API to
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
        except Exception as exc:  # noqa: BLE001
            logger.error("Error in before_tool_invocation: %s", exc, exc_info=True)

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

        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to handle after_tool_invocation: %s", exc, exc_info=True)

    def _extract_tool_info(self, selected_tool: Any, tool_use: dict) -> tuple[str, str, dict]:
        """Extract tool name, ID, and input from event.

        Args:
            selected_tool: The tool being invoked
            tool_use: Tool use dictionary from Strands event

        Returns:
            Tuple of (tool_name, tool_use_id, tool_input)
        """
        tool_name = getattr(selected_tool, 'tool_name', tool_use.get('name', 'unknown_tool'))
        tool_use_id = tool_use.get('toolUseId', str(uuid.uuid4()))
        tool_input = tool_use.get('input', {}) or {}
        return tool_name, tool_use_id, tool_input


class StrandsProfilerHandler(BaseProfilerCallback):

    def __init__(self) -> None:
        super().__init__()
        self._patched: bool = False
        self._hooks_registered: bool = False
        self.last_call_ts = time.time()
        self._run_id_to_start_time = {}

        # Create the tool instrumentation hook
        self.tool_hook = StrandsToolInstrumentationHook(self)

    def instrument(self) -> None:
        """
        Instrument Strands for telemetry capture.

        This patches:
        1. Model streaming methods (OpenAI/Bedrock) for LLM spans
        2. Agent.__init__ to auto-register tool hooks on Agent creation

        Tool instrumentation uses Strands' experimental hooks API,
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

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to instrument Strands models: %s",
                exc,
                exc_info=True,
            )

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
                # Note: Each agent instance needs its own hook registration
                try:
                    # Import hook event types
                    # pylint: disable=import-outside-toplevel
                    from strands.experimental.hooks import AfterToolInvocationEvent
                    from strands.experimental.hooks import BeforeToolInvocationEvent

                    # Register tool hooks on this agent instance
                    agent_self.hooks.add_callback(BeforeToolInvocationEvent,
                                                  handler.tool_hook.on_before_tool_invocation)
                    agent_self.hooks.add_callback(AfterToolInvocationEvent, handler.tool_hook.on_after_tool_invocation)

                    logger.debug("Strands tool hooks registered on Agent instance")

                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to auto-register hooks: %s", exc, exc_info=True)

            # Replace Agent.__init__ with wrapped version
            Agent.__init__ = wrapped_init

            logger.info("Strands Agent.__init__ instrumentation applied")

        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to instrument Agent.__init__: %s", exc, exc_info=True)

    def _extract_model_info(self, model_instance: Any) -> tuple[str, dict[str, Any]]:
        """Extract model name and parameters from Strands model instance."""
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

        # Extract model parameters
        model_params = {}
        try:
            params = getattr(model_instance, "params", {})
            if isinstance(params, dict):
                model_params = {
                    "temperature": params.get("temperature"),
                    "max_tokens": params.get("max_tokens"),
                    "top_p": params.get("top_p"),
                    "stream": params.get("stream", True),
                }
                # Remove None values
                model_params = {k: v for k, v in model_params.items() if v is not None}
        except Exception:  # noqa: BLE001
            pass

        return str(model_name), model_params

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
            system_prompt = (args[2] if len(args) > 2 else kwargs.get("system_prompt"))

            # Build chat_inputs with system prompt and messages
            all_messages = []
            if system_prompt:
                all_messages.append({"text": system_prompt, "role": "system"})
            if isinstance(raw_messages, list):
                all_messages.extend(copy.deepcopy(raw_messages))

            # Always emit START first (before streaming begins)
            start_payload = IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                framework=LLMFrameworkEnum.STRANDS,
                name=str(model_name),
                UUID=event_uuid,
                data=StreamEventData(input=copy.deepcopy(all_messages), output=""),
                metadata=TraceMetadata(chat_inputs=copy.deepcopy(all_messages), ),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(),
                    num_llm_calls=1,
                    seconds_between_calls=int(time.time() - self.last_call_ts),
                ),
            )
            step_manager.push_intermediate_step(start_payload)
            self.last_call_ts = time.time()
            self._run_id_to_start_time[event_uuid] = time.time()

            # Collect output text and token usage while streaming
            output_text = ""
            token_usage = TokenUsageBaseModel()
            ended: bool = False

            def _push_end_if_needed() -> None:
                nonlocal ended
                if ended:
                    return

                chat_responses_list = []
                if output_text:
                    chat_responses_list = [output_text]

                # Build metadata with standard NAT structure
                metadata = TraceMetadata(
                    chat_responses=chat_responses_list,
                    chat_inputs=all_messages,
                )

                # Push END with input/output text and token usage
                end_payload = IntermediateStepPayload(
                    event_type=IntermediateStepType.LLM_END,
                    span_event_timestamp=start_time,
                    framework=LLMFrameworkEnum.STRANDS,
                    name=str(model_name),
                    UUID=event_uuid,
                    data=StreamEventData(input=copy.deepcopy(all_messages), output=output_text),
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

                            # Extract usage information
                            usage_info = self._extract_usage_from_event(ev)
                            if usage_info and not ended:
                                token_usage = TokenUsageBaseModel(**usage_info)
                                _push_end_if_needed()

                        except Exception:  # noqa: BLE001
                            pass
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

        if "data" in ev:
            return str(ev["data"])

        return ""

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
