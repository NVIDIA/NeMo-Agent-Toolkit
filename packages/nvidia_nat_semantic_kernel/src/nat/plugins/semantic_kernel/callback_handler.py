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

import copy
import inspect
import logging
import operator
import threading
import time
from collections.abc import AsyncGenerator
from collections.abc import Callable
from functools import reduce
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field

from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import TraceMetadata
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.profiler_callback import BaseProfilerCallback
from nat.data_models.token_usage import TokenUsageBaseModel

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
        from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion_base import OpenAIChatCompletionBase

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
        import semantic_kernel

        functions_to_patch = set()

        # Gather the appropriate modules/functions based on your builder config
        for llm in self._builder_llms:
            if self._builder_llms[llm].provider_type == 'openai':
                functions_to_patch.update(["openai_non_streaming", "openai_streaming"])

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
        """Build a monkey-patch that records usage statistics around a Semantic Kernel chat-completion method.

        Semantic Kernel's streaming method is an async generator, so it gets a wrapper that is an async
        generator too; the non-streaming method gets a coroutine wrapper.

        Args:
            original_func: The Semantic Kernel chat-completion method to wrap.

        Returns:
            A wrapper with the same calling convention as ``original_func`` that pushes ``LLM_START`` and
            ``LLM_END`` intermediate steps.
        """

        if inspect.isasyncgenfunction(original_func):

            async def patched_streaming_llm_call(sk_self, *args, **kwargs) -> AsyncGenerator[Any, None]:
                uuid, model_name, model_input = self._push_llm_start(sk_self, args, kwargs)

                # Semantic Kernel yields a list of chunks (one per choice) at a time; profile the first choice.
                chunks = []
                async for messages in original_func(sk_self, *args, **kwargs):
                    chunks.extend(message for message in messages if message.choice_index == 0)
                    yield messages

                response = reduce(operator.add, chunks) if chunks else None
                usage = next((chunk.metadata["usage"] for chunk in reversed(chunks) if chunk.metadata.get("usage")),
                             None)
                self._push_llm_end(uuid, model_name, model_input, response, usage)

            return patched_streaming_llm_call

        async def patched_llm_call(sk_self, *args, **kwargs) -> Any:
            uuid, model_name, model_input = self._push_llm_start(sk_self, args, kwargs)

            # Call the original method
            output = await original_func(sk_self, *args, **kwargs)

            response = output[0] if output else None
            usage = response.metadata.get("usage") if response is not None else None
            self._push_llm_end(uuid, model_name, model_input, response, usage)

            return output

        return patched_llm_call

    def _push_llm_start(self, sk_self: Any, args: tuple, kwargs: dict) -> tuple[str, str, str]:
        """Push the ``LLM_START`` intermediate step for a Semantic Kernel chat-completion call.

        Args:
            sk_self: The Semantic Kernel chat-completion service being called.
            args: Positional arguments of the call; the first one is the chat history when present.
            kwargs: Keyword arguments of the call.

        Returns:
            The step UUID, the model name and the text of the model input.
        """
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
            logger.exception("Error in getting model input: %s", e)

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
        return uuid, model_name, model_input

    def _push_llm_end(self, uuid: str, model_name: str, model_input: str, response: Any, usage: Any) -> None:
        """Push the ``LLM_END`` intermediate step for a Semantic Kernel chat-completion call.

        Args:
            uuid: The UUID of the matching ``LLM_START`` step.
            model_name: The name of the model that was called.
            model_input: The text of the model input.
            response: The (combined) chat message content of the first choice, or ``None`` if there is none.
            usage: The Semantic Kernel ``CompletionUsage`` of the call, or ``None`` if the provider sent none.
        """
        if usage is not None:
            usage_data = {key: value for key, value in usage.model_dump().items() if value is not None}
            usage_data.setdefault("total_tokens",
                                  usage_data.get("prompt_tokens", 0) + usage_data.get("completion_tokens", 0))
            token_usage = TokenUsageBaseModel(**usage_data)
        else:
            token_usage = TokenUsageBaseModel()

        # Build the output stats
        output_stats = IntermediateStepPayload(
            event_type=IntermediateStepType.LLM_END,
            span_event_timestamp=time.time(),
            framework=LLMFrameworkEnum.SEMANTIC_KERNEL,
            name=model_name,
            UUID=uuid,
            data=StreamEventData(input=model_input, output=response.content if response is not None else None),
            metadata=TraceMetadata(chat_responses=response.model_dump() if response is not None else None),
            usage_info=UsageInfo(token_usage=token_usage))

        self.step_manager.push_intermediate_step(output_stats)

        # Update last_call_ts
        self.last_call_ts = time.time()

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
                logger.error("ToolUsage._use error: %s", e)
                raise

        return patched_tool_call
