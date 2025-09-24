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

import copy
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import litellm

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


class ADKProfilerHandler(BaseProfilerCallback):
    """
    A callback manager/handler for Google ADK that intercepts calls to:
      - Tools
      - LLMs
    to collect usage statistics (tokens, inputs, outputs, time intervals, etc.)
    and store them in NeMo Agent Toolkit's usage_stats queue for subsequent analysis.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.last_call_ts = time.time()
        self.step_manager = Context.get().intermediate_step_manager

        # Original references to Google ADK Tool and LLM methods (for uninstrumenting if needed)
        self._original_tool_call = None
        self._original_llm_call = None
        self._instrumented = False

    def instrument(self) -> None:
        """
        Monkey-patch the relevant Google ADK methods with usage-stat collection logic.
        Assumes the 'google-adk' library is installed.
        """

        if getattr(self, "_instrumented", False):
            logger.debug("ADKProfilerHandler already instrumented; skipping.")
            return
        try:
            from google.adk.tools.function_tool import FunctionTool
        except Exception as _e:
            logger.exception("ADK import failed; skipping instrumentation")
            return

        # Save the originals
        self._original_tool_call = getattr(FunctionTool, "run_async", None)
        self._original_llm_call = getattr(litellm, "acompletion", None)

        # Patch if available
        if self._original_tool_call:
            FunctionTool.run_async = self._tool_use_monkey_patch()

        if self._original_llm_call:
            litellm.acompletion = self._llm_call_monkey_patch()

        logger.debug("ADKProfilerHandler instrumentation applied successfully.")
        self._instrumented = True

    def uninstrument(self) -> None:
        """ Restore the original Google ADK methods.
        Add an explicit unpatch to avoid side-effects across tests/process lifetime.
        """
        try:
            from google.adk.tools.function_tool import FunctionTool
            if self._original_tool_call:
                FunctionTool.run_async = self._original_tool_call
            if self._original_llm_call:
                litellm.acompletion = self._original_llm_call
            logger.debug("ADKProfilerHandler uninstrumented successfully.")
        except Exception as _e:
            logger.exception("Failed to uninstrument ADKProfilerHandler")

    def _tool_use_monkey_patch(self) -> Callable[..., Any]:
        """
        Returns a function that wraps calls to BaseTool.run_async with usage-logging.
        """
        original_func = self._original_tool_call

        async def wrapped_tool_use(base_tool_instance, *args, **kwargs) -> Any:
            """
            Replicates _tool_use_wrapper logic without wrapt: collects usage stats,
            calls the original, and captures output stats.

            Args:
                base_tool_instance (FunctionTool): The instance of the tool being called.
                *args: Positional arguments to the tool.
                **kwargs: Keyword arguments to the tool.

            Returns:
                Any: The result of the tool execution.
            """
            now = time.time()
            tool_name = ""

            try:
                tool_name = base_tool_instance.name
            except Exception as _e:
                logger.exception("Error getting tool name")
                tool_name = ""

            try:
                # Pre-call usage event - safely extract kwargs args if present
                kwargs_args = (kwargs.get("args", {}) if isinstance(kwargs.get("args"), dict) else {})
                stats = IntermediateStepPayload(
                    event_type=IntermediateStepType.TOOL_START,
                    framework=LLMFrameworkEnum.ADK,
                    name=tool_name,
                    data=StreamEventData(),
                    metadata=TraceMetadata(tool_inputs={
                        "args": args, "kwargs": dict(kwargs_args)
                    }),
                    usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                )

                self.step_manager.push_intermediate_step(stats)

                with self._lock:
                    self.last_call_ts = now

                # Call the original _use(...)
                result = await original_func(base_tool_instance, *args, **kwargs)
                now = time.time()
                # Post-call usage stats - safely extract kwargs args if present
                kwargs_args = (kwargs.get("args", {}) if isinstance(kwargs.get("args"), dict) else {})
                usage_stat = IntermediateStepPayload(
                    event_type=IntermediateStepType.TOOL_END,
                    span_event_timestamp=now,
                    framework=LLMFrameworkEnum.ADK,
                    name=tool_name,
                    data=StreamEventData(
                        input={
                            "args": args, "kwargs": dict(kwargs_args)
                        },
                        output=str(result),
                    ),
                    metadata=TraceMetadata(tool_outputs={"result": str(result)}),
                    usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                )

                self.step_manager.push_intermediate_step(usage_stat)

                return result

            except Exception as _e:
                logger.exception("BaseTool error occured")
                raise

        return wrapped_tool_use

    def _llm_call_monkey_patch(self) -> Callable[..., Any]:
        """
        Returns a function that wraps calls to litellm.acompletion(...) with usage-logging.

        Returns:
            Callable[..., Any]: The wrapped function.
        """
        original_func = self._original_llm_call

        async def wrapped_llm_call(*args, **kwargs) -> Any:
            """
            Replicates _llm_call_wrapper logic without wrapt: collects usage stats,
            calls the original, and captures output stats.

            Args:
                *args: Positional arguments to the LLM call.
                **kwargs: Keyword arguments to the LLM call.

            Returns:
                Any: The result of the LLM call.
            """

            now = time.time()
            with self._lock:
                seconds_between_calls = int(now - self.last_call_ts)
            model_name = kwargs.get("model")
            if not model_name and args:
                first = args[0]
                if isinstance(first, str):
                    model_name = first
            model_name = model_name or ""

            model_input = ""
            try:
                for message in kwargs.get("messages", []):
                    content = message.get("content", "")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict):
                                model_input += str(part.get("text", ""))  # text parts
                            else:
                                model_input += str(part)
                    else:
                        model_input += content or ""
            except Exception as _e:
                logger.exception("Error getting model input")

            # Record the start event
            input_stats = IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                framework=LLMFrameworkEnum.ADK,
                name=model_name,
                data=StreamEventData(input=model_input),
                metadata=TraceMetadata(chat_inputs=copy.deepcopy(kwargs.get("messages", []))),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(),
                    num_llm_calls=1,
                    seconds_between_calls=seconds_between_calls,
                ),
            )

            self.step_manager.push_intermediate_step(input_stats)

            # Call the original litellm.acompletion(...)
            output = await original_func(*args, **kwargs)

            model_output = ""
            try:
                for choice in output.choices:
                    msg = choice.message
                    model_output += msg.content or ""
            except Exception as _e:
                logger.exception("Error getting model output")

            now = time.time()
            # Record the end event
            # Prepare safe metadata and usage
            chat_resp: dict[str, Any] = {}
            try:
                if getattr(output, "choices", []):
                    first_choice = output.choices[0]
                    chat_resp = first_choice.model_dump() if hasattr(
                        first_choice, "model_dump") else getattr(first_choice, "__dict__", {}) or {}
            except Exception as _e:
                logger.exception("Error preparing chat_responses")

            usage_payload: dict[str, Any] = {}
            try:
                usage_obj = getattr(output, "usage", None) or (getattr(output, "model_extra", {}) or {}).get("usage")
                if usage_obj:
                    if hasattr(usage_obj, "model_dump"):
                        usage_payload = usage_obj.model_dump()
                    elif isinstance(usage_obj, dict):
                        usage_payload = usage_obj
            except Exception as _e:
                logger.exception("Error preparing token usage")

            output_stats = IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_END,
                span_event_timestamp=now,
                framework=LLMFrameworkEnum.ADK,
                name=model_name,
                data=StreamEventData(input=model_input, output=model_output),
                metadata=TraceMetadata(chat_responses=chat_resp),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(**usage_payload),
                    num_llm_calls=1,
                    seconds_between_calls=seconds_between_calls,
                ),
            )

            self.step_manager.push_intermediate_step(output_stats)

            with self._lock:
                self.last_call_ts = now

            return output

        return wrapped_llm_call
