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
from crewai.tools import tool_usage

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


class CrewAIProfilerHandler(BaseProfilerCallback):
    """
    A callback manager/handler for CrewAI that intercepts calls to:
      - ToolUsage._use
      - LLM Calls
    to collect usage statistics (tokens, inputs, outputs, time intervals, etc.)
    and store them in AIQ Toolkit's usage_stats queue for subsequent analysis.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.last_call_ts = time.time()
        self.step_manager = Context.get().intermediate_step_manager

        # Original references to CrewAI methods (for uninstrumenting if needed)
        self._original_tool_use = None
        self._original_llm_call = None

    def instrument(self) -> None:
        """
        Monkey-patch the relevant CrewAI methods with usage-stat collection logic.
        Assumes the 'crewai' library is installed.
        """

        # Save the originals
        self._original_tool_use = getattr(tool_usage.ToolUsage, "_use", None)
        self._original_llm_call = getattr(litellm, "completion", None)

        # Patch if available
        if self._original_tool_use:
            tool_usage.ToolUsage._use = self._tool_use_monkey_patch()

        if self._original_llm_call:
            litellm.completion = self._llm_call_monkey_patch()

        logger.debug("CrewAIProfilerHandler instrumentation applied successfully.")

    def _tool_use_monkey_patch(self) -> Callable[..., Any]:
        """
        Returns a function that wraps calls to ToolUsage._use(...) with usage-logging.
        """
        original_func = self._original_tool_use

        def wrapped_tool_use(tool_usage_instance, *args, **kwargs) -> Any:
            """
            Replicates _tool_use_wrapper logic without wrapt: collects usage stats,
            calls the original, and captures output stats.
            """
            now = time.time()
            tool_name = ""

            try:
                tool_info = kwargs.get("tool", "")

                if tool_info:
                    tool_name = tool_info.name
            except Exception as e:
                logger.exception("Error getting tool name: %s", e, exc_info=True)

            try:
                # Pre-call usage event
                stats = IntermediateStepPayload(event_type=IntermediateStepType.TOOL_START,
                                                framework=LLMFrameworkEnum.CREWAI,
                                                name=tool_name,
                                                data=StreamEventData(),
                                                metadata=TraceMetadata(tool_inputs={
                                                    "args": args, "kwargs": dict(kwargs)
                                                }),
                                                usage_info=UsageInfo(token_usage=TokenUsageBaseModel()))

                self.step_manager.push_intermediate_step(stats)

                self.last_call_ts = now

                # Call the original _use(...)
                result = original_func(tool_usage_instance, *args, **kwargs)
                now = time.time()
                # Post-call usage stats
                usage_stat = IntermediateStepPayload(
                    event_type=IntermediateStepType.TOOL_END,
                    span_event_timestamp=now,
                    framework=LLMFrameworkEnum.CREWAI,
                    name=tool_name,
                    data=StreamEventData(input={
                        "args": args, "kwargs": dict(kwargs)
                    }, output=str(result)),
                    metadata=TraceMetadata(tool_outputs={"result": str(result)}),
                    usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                )

                self.step_manager.push_intermediate_step(usage_stat)

                return result

            except Exception as e:
                logger.exception("ToolUsage._use error: %s", e)
                raise

        return wrapped_tool_use

    def _llm_call_monkey_patch(self) -> Callable[..., Any]:
        """
        Returns a function that wraps calls to litellm.completion(...) with usage-logging.
        """
        original_func = self._original_llm_call

        def wrapped_llm_call(*args, **kwargs) -> Any:
            """
            Replicates _llm_call_wrapper logic without wrapt: collects usage stats,
            calls the original, and captures output stats.
            """

            now = time.time()
            seconds_between_calls = int(now - self.last_call_ts)
            model_name = kwargs.get('model', "")

            model_input = ""
            try:
                for message in kwargs.get('messages', []):
                    model_input += message.get('content', "")
            except Exception as e:
                logger.exception("Error getting model input: %s", e, exc_info=True)

            # Record the start event
            input_stats = IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                framework=LLMFrameworkEnum.CREWAI,
                name=model_name,
                data=StreamEventData(input=model_input),
                metadata=TraceMetadata(chat_inputs=copy.deepcopy(kwargs.get('messages', []))),
                usage_info=UsageInfo(token_usage=TokenUsageBaseModel(),
                                     num_llm_calls=1,
                                     seconds_between_calls=seconds_between_calls))

            self.step_manager.push_intermediate_step(input_stats)

            # Call the original litellm.completion(...)
            output = original_func(*args, **kwargs)

            model_output = ""
            try:
                for choice in output.choices:
                    msg = choice.model_extra["message"]
                    model_output += msg.get('content', "")
            except Exception as e:
                logger.exception("Error getting model output: %s", e, exc_info=True)

            now = time.time()
            # Record the end event
            output_stats = IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_END,
                span_event_timestamp=now,
                framework=LLMFrameworkEnum.CREWAI,
                name=model_name,
                data=StreamEventData(input=model_input, output=model_output),
                metadata=TraceMetadata(chat_responses=output.choices[0].model_dump()),
                usage_info=UsageInfo(token_usage=TokenUsageBaseModel(**output.model_extra['usage'].model_dump()),
                                     num_llm_calls=1,
                                     seconds_between_calls=seconds_between_calls))

            self.step_manager.push_intermediate_step(output_stats)

            # (Note: the original code did NOT update self.last_call_ts here)
            return output

        return wrapped_llm_call
