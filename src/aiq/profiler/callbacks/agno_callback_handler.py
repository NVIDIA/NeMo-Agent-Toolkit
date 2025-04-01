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
from uuid import uuid4

import litellm

from aiq.builder.context import AIQContext
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepType
from aiq.data_models.intermediate_step import StreamEventData
from aiq.data_models.intermediate_step import TraceMetadata
from aiq.data_models.intermediate_step import UsageInfo
from aiq.profiler.callbacks.base_callback_class import BaseProfilerCallback
from aiq.profiler.callbacks.token_usage_base_model import TokenUsageBaseModel

logger = logging.getLogger(__name__)


class AgnoProfilerHandler(BaseProfilerCallback):
    """
    A callback manager/handler for Agno that intercepts calls to:
      - Tool execution
      - LLM Calls
    to collect usage statistics (tokens, inputs, outputs, time intervals, etc.)
    and store them in AgentIQ's usage_stats queue for subsequent analysis.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.last_call_ts = time.time()
        self.step_manager = AIQContext.get().intermediate_step_manager

        # Original references to Agno methods (for uninstrumenting if needed)
        self._original_tool_execute = None
        self._original_llm_call = None

    def instrument(self) -> None:
        """
        Monkey-patch the relevant Agno methods with usage-stat collection logic.
        """
        # Save the originals
        self._original_llm_call = getattr(litellm, "completion", None)

        # Patch if available
        if self._original_llm_call:
            litellm.completion = self._llm_call_monkey_patch()

        logger.debug("AgnoProfilerHandler instrumentation applied successfully.")

    def _tool_execute_monkey_patch(self) -> Callable[..., Any]:
        """
        Returns a function that wraps tool execution calls with usage-logging.
        """
        original_func = self._original_tool_execute

        def wrapped_tool_execute(*args, **kwargs) -> Any:
            """
            Collects usage stats for tool execution, calls the original, and captures output stats.
            """
            now = time.time()
            tool_name = kwargs.get("tool_name", "")
            uuid = str(uuid4())

            try:
                # Pre-call usage event
                stats = IntermediateStepPayload(event_type=IntermediateStepType.TOOL_START,
                                                framework=LLMFrameworkEnum.AGNO,
                                                name=tool_name,
                                                UUID=uuid,
                                                data=StreamEventData(),
                                                metadata=TraceMetadata(tool_inputs={
                                                    "args": args, "kwargs": dict(kwargs)
                                                }),
                                                usage_info=UsageInfo(token_usage=TokenUsageBaseModel()))

                self.step_manager.push_intermediate_step(stats)
                self.last_call_ts = now

                # Call the original execute
                result = original_func(*args, **kwargs)
                now = time.time()

                # Post-call usage stats
                usage_stat = IntermediateStepPayload(
                    event_type=IntermediateStepType.TOOL_END,
                    span_event_timestamp=now,
                    framework=LLMFrameworkEnum.AGNO,
                    name=tool_name,
                    UUID=uuid,
                    data=StreamEventData(input={
                        "args": args, "kwargs": dict(kwargs)
                    }, output=str(result)),
                    metadata=TraceMetadata(tool_outputs={"result": str(result)}),
                    usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                )

                self.step_manager.push_intermediate_step(usage_stat)
                return result

            except Exception as e:
                logger.exception("Tool execution error: %s", e)
                raise

        return wrapped_tool_execute

    def _llm_call_monkey_patch(self) -> Callable[..., Any]:
        """
        Returns a function that wraps calls to litellm.completion(...) with usage-logging.
        """
        original_func = self._original_llm_call

        def wrapped_llm_call(*args, **kwargs) -> Any:
            """
            Collects usage stats for LLM calls, calls the original, and captures output stats.
            """
            now = time.time()
            seconds_between_calls = int(now - self.last_call_ts)
            model_name = kwargs.get('model', "")

            model_input = ""
            try:
                for message in kwargs.get('messages', []):
                    model_input += message.get('content', "")
            except Exception as e:
                logger.exception("Error getting model input: %s", e)

            uuid = str(uuid4())

            # Record the start event
            input_stats = IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                framework=LLMFrameworkEnum.AGNO,
                name=model_name,
                UUID=uuid,
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
                logger.exception("Error getting model output: %s", e)

            now = time.time()
            # Record the end event
            output_stats = IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_END,
                span_event_timestamp=now,
                framework=LLMFrameworkEnum.AGNO,
                name=model_name,
                UUID=uuid,
                data=StreamEventData(input=model_input, output=model_output),
                metadata=TraceMetadata(chat_responses=output.choices[0].model_dump()),
                usage_info=UsageInfo(token_usage=TokenUsageBaseModel(**output.model_extra['usage'].model_dump()),
                                     num_llm_calls=1,
                                     seconds_between_calls=seconds_between_calls))

            self.step_manager.push_intermediate_step(output_stats)
            return output

        return wrapped_llm_call
