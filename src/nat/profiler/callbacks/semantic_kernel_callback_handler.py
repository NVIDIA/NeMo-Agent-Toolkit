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

import semantic_kernel
from pydantic import BaseModel
from pydantic import Field
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion_base import OpenAIChatCompletionBase

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


class SemanticKernelPatchMethod(BaseModel):
    """
    Stores the module and function to patch in Semantic Kernel.
    """
    module: Any = Field(..., description="The module to patch")
    function: str = Field(..., description="The function to patch")


class SemanticKernelProfilerHandler(BaseProfilerCallback):
    """
    A callback manager/handler for Msft Semantic Kernel that intercepts calls to:

    - Chat Completions Endpoints
    - Tool calls

    to collect usage statistics (tokens, inputs, outputs, time intervals, etc.)
    and store them in NAT's usage_stats queue for subsequent analysis.
    """

    def __init__(self, workflow_llms: dict) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.last_call_ts = time.time()
        self.step_manager = Context.get().intermediate_step_manager
        self._builder_llms = workflow_llms

        # Original references to SK methods
        self._original_tool_call = None

        # Store a mapping of current SK methods we support patching for
        self._patch_methods = {
            "openai_streaming":
                SemanticKernelPatchMethod(module=OpenAIChatCompletionBase,
                                          function="_inner_get_streaming_chat_message_contents"),
            "openai_non_streaming":
                SemanticKernelPatchMethod(module=OpenAIChatCompletionBase, function="_inner_get_chat_message_contents")
        }

    def instrument(self) -> None:
        """
        Monkey-patch the relevant Semantic Kernel methods with usage-stat collection logic.
        """

        functions_to_patch = []

        # Gather the appropriate modules/functions based on your builder config
        for llm in self._builder_llms:
            if self._builder_llms[llm].provider_type == 'openai':  # pylint: disable=consider-using-in
                functions_to_patch.extend(["openai_non_streaming", "openai_streaming"])

        # Grab original reference for the tool call
        self._original_tool_call = getattr(semantic_kernel.Kernel, "invoke_function_call", None)

        # Now do direct monkey-patching: replace each function with a closure
        for method in functions_to_patch:
            patch_method = self._patch_methods[method]
            setattr(patch_method.module,
                    patch_method.function,
                    self._build_llm_call_patch(getattr(patch_method.module, patch_method.function)))

        if self._original_tool_call:
            patched_tool_call = self._build_tool_call_patch(self._original_tool_call)
            setattr(semantic_kernel.Kernel, "invoke_function_call", patched_tool_call)

        logger.debug("SemanticKernelProfilerHandler instrumentation applied successfully.")

    def _build_llm_call_patch(self, original_func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Returns an async monkey-patch that wraps the original chat-completion method.
        Replicates the usage collection from _llm_call_wrapper.
        """

        async def patched_llm_call(sk_self, *args, **kwargs) -> Any:
            now = time.time()
            seconds_between_calls = int(now - self.last_call_ts)
            uuid = str(uuid4())

            # Build the input stats
            if args:
                chat_input = [copy.deepcopy(args[0].model_dump())]
            else:
                # if no args, fallback on kwargs["chat_history"]
                chat_input = [kwargs["chat_history"].model_dump()]

            model_name = sk_self.ai_model_id

            model_input = ""
            try:
                for message in chat_input[0]["messages"]:
                    for item in message["items"]:
                        if "text" in item:
                            model_input += item["text"]
            except Exception as e:
                logger.exception("Error in getting model input: %s", e, exc_info=True)

            input_stats = IntermediateStepPayload(event_type=IntermediateStepType.LLM_START,
                                                  framework=LLMFrameworkEnum.SEMANTIC_KERNEL,
                                                  name=model_name,
                                                  UUID=uuid,
                                                  data=StreamEventData(input=model_input),
                                                  metadata=TraceMetadata(chat_inputs=copy.deepcopy(chat_input)),
                                                  usage_info=UsageInfo(token_usage=TokenUsageBaseModel(),
                                                                       num_llm_calls=1,
                                                                       seconds_between_calls=seconds_between_calls))

            self.step_manager.push_intermediate_step(input_stats)

            # Call the original method
            output = await original_func(sk_self, *args, **kwargs)

            model_output = output[0].content
            now = time.time()
            # Build the output stats
            output_stats = IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_END,
                span_event_timestamp=now,
                framework=LLMFrameworkEnum.SEMANTIC_KERNEL,
                name=model_name,
                UUID=uuid,
                data=StreamEventData(input=model_input, output=model_output),
                metadata=TraceMetadata(chat_responses=output[0].model_dump()),
                usage_info=UsageInfo(token_usage=TokenUsageBaseModel(**output[0].metadata["usage"].model_dump())))

            self.step_manager.push_intermediate_step(output_stats)

            # Update last_call_ts
            self.last_call_ts = time.time()

            return output

        return patched_llm_call

    def _build_tool_call_patch(self, original_func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Returns an async monkey-patch that wraps the original tool call (invoke_function_call).
        Replicates usage collection from _tool_use_wrapper.
        """

        async def patched_tool_call(kernel_self, *args, **kwargs) -> Any:

            uuid = str(uuid4())
            now = time.time()
            # Extract the tool input
            if kwargs:
                tool_input = kwargs["function_call"].model_dump(exclude="content_type")
            else:
                tool_input = args[0].model_dump(exclude="content_type")

            try:
                # Pre-call usage event
                input_stat = IntermediateStepPayload(event_type=IntermediateStepType.TOOL_START,
                                                     framework=LLMFrameworkEnum.SEMANTIC_KERNEL,
                                                     name=tool_input["name"],
                                                     UUID=uuid,
                                                     data=StreamEventData(input=tool_input),
                                                     metadata=TraceMetadata(tool_inputs=copy.deepcopy(tool_input),
                                                                            tool_info=copy.deepcopy(tool_input)),
                                                     usage_info=UsageInfo(token_usage=TokenUsageBaseModel()))

                self.step_manager.push_intermediate_step(input_stat)
                now = time.time()
                # Call the original invoke_function_call
                result = await original_func(kernel_self, *args, **kwargs)

                # Try to get the chat history from kwargs or args
                if kwargs:
                    chat_history = copy.deepcopy(kwargs["chat_history"])
                else:
                    chat_history = copy.deepcopy(args[1])

                # Post-call usage event
                output_stat = IntermediateStepPayload(event_type=IntermediateStepType.TOOL_END,
                                                      span_event_timestamp=now,
                                                      framework=LLMFrameworkEnum.SEMANTIC_KERNEL,
                                                      name=tool_input["name"],
                                                      UUID=uuid,
                                                      data=StreamEventData(input=tool_input,
                                                                           output=[
                                                                               item.model_dump(exclude="content_type")
                                                                               for item in chat_history[-1].items
                                                                           ]),
                                                      metadata=TraceMetadata(tool_outputs=[
                                                          item.model_dump(exclude="content_type")
                                                          for item in chat_history[-1].items
                                                      ],
                                                                             tool_info=copy.deepcopy(tool_input)),
                                                      usage_info=UsageInfo(token_usage=TokenUsageBaseModel()))

                self.step_manager.push_intermediate_step(output_stat)

                return result

            except Exception as e:
                logger.exception("ToolUsage._use error: %s", e)
                raise

        return patched_tool_call
