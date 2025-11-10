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
"""AutoGen callback handler for usage statistics collection. """

import copy
import logging
import threading
import time
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


class AutoGenProfilerHandler(BaseProfilerCallback):
    """
    A callback manager/handler for AutoGen that intercepts calls to:
      - Tools
      - LLMs
    to collect usage statistics (tokens, inputs, outputs, time intervals, etc.)
    and store them in NeMo Agent Toolkit's usage_stats queue for subsequent analysis.
    """

    def __init__(self) -> None:
        """
        Initializes the AutogenProfilerHandler.
        """
        super().__init__()
        self._lock = threading.Lock()
        self.last_call_ts = time.time()
        self.step_manager = Context.get().intermediate_step_manager

        # Original references to AutoGen Tool and LLM methods (for uninstrumenting if needed)
        self._original_tool_call = None
        self._original_llm_call = None
        self._instrumented = False

    def instrument(self) -> None:
        """
        Monkey-patch the relevant AutoGen methods with usage-stat collection logic.
        Assumes the 'autogen-core' library is installed.
        """

        if getattr(self, "_instrumented", False):
            logger.debug("AutoGenProfilerHandler already instrumented; skipping.")
            return
        try:
            from autogen_core.tools import BaseTool

            # from autogen_ext.tools.mcp import McpWorkbench
            from autogen_ext.models.openai import OpenAIChatCompletionClient
        except Exception as _e:
            logger.exception("AutoGen import failed; skipping instrumentation")
            return

        # Save the originals
        self._original_tool_call = getattr(BaseTool, "run_json", None)
        self._original_llm_call = getattr(OpenAIChatCompletionClient, "create", None)

        if self._original_llm_call:
            OpenAIChatCompletionClient.create = self._llm_call_monkey_patch()

        if self._original_tool_call:
            BaseTool.run_json = self._tool_call_monkey_patch()

        logger.debug("AutoGenProfilerHandler instrumentation applied successfully.")
        self._instrumented = True

    def uninstrument(self) -> None:
        """ Restore the original AutoGen methods.
        Add an explicit unpatch to avoid side-effects across tests/process lifetime.
        """
        try:
            if self._original_llm_call:
                from autogen_ext.models.openai import OpenAIChatCompletionClient
                OpenAIChatCompletionClient.create = self._original_llm_call
            if self._original_tool_call:
                from autogen_core.tools import BaseTool
                BaseTool.run_json = self._original_tool_call
            self._instrumented = False
            logger.debug("AutoGenProfilerHandler uninstrumented successfully.")
        except Exception as _e:
            logger.exception("Failed to uninstrument AutoGenProfilerHandler")

    def _llm_call_monkey_patch(self) -> Callable[..., Any]:
        """
        Returns a function that wraps calls to ChatCompletionClient.create(...) with usage-logging.

        Returns:
            Callable[..., Any]: The wrapped function.
        """
        original_func = self._original_llm_call

        async def wrapped_llm_call(*args: Any, **kwargs: Any) -> Any:
            """
            Replicates ChatCompletionClient.create(...) logic without wrapt: collects usage stats,
            calls the original, and captures output stats.

            Args:
                *args (Any): Positional arguments to the LLM call.
                **kwargs (Any): Keyword arguments to the LLM call.

            Returns:
                Any: The result of the LLM call.
            """

            now = time.time()
            with self._lock:
                seconds_between_calls = int(now - self.last_call_ts)

            model_name = None
            try:
                model_name = getattr(args[0], "_raw_config", {}).get("model", None)
            except Exception as _e:
                logger.exception("Error retrieving model name from args[0]._raw_config")
            if not model_name:
                model_name = str(getattr(args[0], "model", "unknown_model"))

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
                framework=LLMFrameworkEnum.AUTOGEN,
                name=model_name,
                data=StreamEventData(input=model_input),
                metadata=TraceMetadata(chat_inputs=copy.deepcopy(kwargs.get("messages", []))),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(),
                    num_llm_calls=1,
                    seconds_between_calls=seconds_between_calls,
                ),
            )

            llm_start_uuid = input_stats.UUID

            self.step_manager.push_intermediate_step(input_stats)

            # Call the original ChatCompletionClient.create(...)

            try:
                output = await original_func(*args, **kwargs)
            except Exception as _e:
                logger.error("Error during LLM call: %s", _e)
                self.step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        event_type=IntermediateStepType.LLM_END,
                        span_event_timestamp=time.time(),
                        framework=LLMFrameworkEnum.AUTOGEN,
                        name=model_name,
                        data=StreamEventData(input=model_input, output=str(_e)),
                        metadata=TraceMetadata(error=str(_e)),
                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                        UUID=llm_start_uuid,
                    ))
                with self._lock:
                    self.last_call_ts = time.time()
                raise

            model_output = ""
            try:
                for content in output.content:
                    msg = str(content)
                    model_output += msg or ""
            except Exception as _e:
                logger.error("Error getting model output")
                self.step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        event_type=IntermediateStepType.LLM_END,
                        span_event_timestamp=time.time(),
                        framework=LLMFrameworkEnum.AUTOGEN,
                        name=model_name,
                        data=StreamEventData(input=model_input, output=str(_e)),
                        metadata=TraceMetadata(error=str(_e)),
                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                        UUID=llm_start_uuid,
                    ))
                with self._lock:
                    self.last_call_ts = time.time()
                raise

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
                logger.error("Error preparing chat_responses")
                self.step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        event_type=IntermediateStepType.LLM_END,
                        span_event_timestamp=time.time(),
                        framework=LLMFrameworkEnum.AUTOGEN,
                        name=model_name,
                        data=StreamEventData(input=model_input, output=str(_e)),
                        metadata=TraceMetadata(error=str(_e)),
                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                        UUID=llm_start_uuid,
                    ))
                with self._lock:
                    self.last_call_ts = time.time()
                raise

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

            output_stats = IntermediateStepPayload(event_type=IntermediateStepType.LLM_END,
                                                   span_event_timestamp=now,
                                                   framework=LLMFrameworkEnum.AUTOGEN,
                                                   name=model_name,
                                                   data=StreamEventData(input=model_input, output=model_output),
                                                   metadata=TraceMetadata(chat_responses=chat_resp),
                                                   usage_info=UsageInfo(
                                                       token_usage=TokenUsageBaseModel(**usage_payload),
                                                       num_llm_calls=1,
                                                       seconds_between_calls=seconds_between_calls,
                                                   ),
                                                   UUID=llm_start_uuid)

            self.step_manager.push_intermediate_step(output_stats)

            with self._lock:
                self.last_call_ts = now

            return output

        return wrapped_llm_call

    def _tool_call_monkey_patch(self) -> Callable[..., Any]:
        """
        Returns a function that wraps calls to BaseTool.run_json(...) with usage-logging.

        Returns:
            Callable[..., Any]: The wrapped function.
        """
        original_func = self._original_tool_call

        async def wrapped_tool_call(*args: Any, **kwargs: Any) -> Any:
            """
            Replicates BaseTool.run_json(...) logic without wrapt: collects usage stats,
            calls the original, and captures output stats.

            Args:
                *args (Any): Positional arguments to the tool call.
                **kwargs (Any): Keyword arguments to the tool call.

            Returns:
                Any: The result of the tool call.
            """

            now = time.time()
            with self._lock:
                seconds_between_calls = int(now - self.last_call_ts)

            tool_name = None
            try:
                tool_name = str(getattr(args[0], "name", "unknown_tool"))
            except Exception as _e:
                logger.exception("Error getting tool name")

            # Record the start event
            input_stats = IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_START,
                framework=LLMFrameworkEnum.AUTOGEN,
                name=tool_name,
                data=StreamEventData(input={}),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(),
                    num_llm_calls=0,
                    seconds_between_calls=seconds_between_calls,
                ),
            )

            tool_start_uuid = input_stats.UUID

            self.step_manager.push_intermediate_step(input_stats)

            tool_input = ""
            try:
                tool_input = str(args[1].kwargs)
            except (IndexError, AttributeError):
                tool_input = str(args[1].get('kwargs', {}))
            except Exception as _e:
                logger.error("Error getting tool input: %s", _e)
                self.step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        event_type=IntermediateStepType.TOOL_END,
                        span_event_timestamp=time.time(),
                        framework=LLMFrameworkEnum.AUTOGEN,
                        name=tool_name,
                        data=StreamEventData(input=tool_input, output=str(_e)),
                        metadata=TraceMetadata(error=str(_e)),
                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                        UUID=tool_start_uuid,
                    ))
                with self._lock:
                    self.last_call_ts = time.time()
                raise

            try:
                # Call the original BaseTool.run_json(...)
                # output = await original_func(*args, **kwargs)
                output = await original_func(*args, **kwargs)
            except Exception as _e:
                logger.error("Tool execution failed with error: %s", _e)
                self.step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        event_type=IntermediateStepType.TOOL_END,
                        span_event_timestamp=time.time(),
                        framework=LLMFrameworkEnum.AUTOGEN,
                        name=tool_name,
                        data=StreamEventData(input=tool_input, output=str(_e)),
                        metadata=TraceMetadata(error=str(_e)),
                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                        UUID=tool_start_uuid,
                    ))
                with self._lock:
                    self.last_call_ts = time.time()
                raise

            tool_output = output

            now = time.time()
            # Record the end event
            kwargs_args = (kwargs.get("args", {}) if isinstance(kwargs.get("args"), dict) else {})
            usage_stat = IntermediateStepPayload(event_type=IntermediateStepType.TOOL_END,
                                                 span_event_timestamp=now,
                                                 framework=LLMFrameworkEnum.AUTOGEN,
                                                 name=tool_name,
                                                 data=StreamEventData(
                                                     input={
                                                         "args": args, "kwargs": dict(kwargs_args)
                                                     },
                                                     output=str(tool_output),
                                                 ),
                                                 metadata=TraceMetadata(tool_outputs={"result": str(tool_output)}),
                                                 usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                                                 UUID=tool_start_uuid)

            self.step_manager.push_intermediate_step(usage_stat)

            return tool_output

        return wrapped_tool_call
