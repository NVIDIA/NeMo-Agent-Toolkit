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

from __future__ import annotations

import copy
import logging
import threading
import time
from typing import Any
from uuid import UUID
from uuid import uuid4

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import LLMResult

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


class LangchainProfilerHandler(AsyncCallbackHandler, BaseProfilerCallback):  # pylint: disable=R0901
    """Callback Handler that tracks NIM info."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    raise_error = True  # Override to raise error and run inline
    run_inline = True

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.last_call_ts = time.time()

        self.step_manager = AIQContext.get().intermediate_step_manager
        self._state = IntermediateStepType.LLM_END

        self._run_id_to_model_name = {}
        self._run_id_to_llm_input = {}
        self._run_id_to_tool_input = {}
        self._run_id_to_start_time = {}

    def __repr__(self) -> str:
        return (f"Tokens Used: {self.total_tokens}\n"
                f"\tPrompt Tokens: {self.prompt_tokens}\n"
                f"\tCompletion Tokens: {self.completion_tokens}\n"
                f"Successful Requests: {self.successful_requests}\n")

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def _extract_token_base_model(self, usage_metadata: dict[str, Any]) -> TokenUsageBaseModel:
        if usage_metadata:
            prompt_tokens = usage_metadata.get("input_tokens", 0)
            completion_tokens = usage_metadata.get("output_tokens", 0)
            total_tokens = usage_metadata.get("total_tokens", 0)

            return TokenUsageBaseModel(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        return TokenUsageBaseModel()

    async def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:

        model_name = ""
        try:
            model_name = kwargs.get("metadata")["ls_model_name"]
        except Exception as e:
            logger.exception("Error getting model name: %s", e, exc_info=True)

        run_id = str(kwargs.get("run_id", str(uuid4())))
        self._run_id_to_model_name[run_id] = model_name

        stats = IntermediateStepPayload(event_type=IntermediateStepType.LLM_START,
                                        framework=LLMFrameworkEnum.LANGCHAIN,
                                        name=model_name,
                                        UUID=run_id,
                                        data=StreamEventData(input=prompts[-1]),
                                        metadata=TraceMetadata(chat_inputs=copy.deepcopy(prompts)),
                                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel(),
                                                             num_llm_calls=1,
                                                             seconds_between_calls=int(time.time() -
                                                                                       self.last_call_ts)))

        self.step_manager.push_intermediate_step(stats)
        self._run_id_to_llm_input[run_id] = prompts[-1]
        self._state = IntermediateStepType.LLM_START
        self.last_call_ts = time.time()
        self._run_id_to_start_time[run_id] = time.time()

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:

        model_name = ""
        try:
            model_name = metadata["ls_model_name"] if metadata else kwargs.get("metadata")["ls_model_name"]
        except Exception as e:
            logger.exception("Error getting model name: %s", e, exc_info=True)

        run_id = str(run_id)
        self._run_id_to_model_name[run_id] = model_name

        stats = IntermediateStepPayload(event_type=IntermediateStepType.LLM_START,
                                        framework=LLMFrameworkEnum.LANGCHAIN,
                                        name=model_name,
                                        UUID=run_id,
                                        data=StreamEventData(input=copy.deepcopy(messages[0])),
                                        metadata=TraceMetadata(chat_inputs=copy.deepcopy(messages[0])),
                                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel(),
                                                             num_llm_calls=1,
                                                             seconds_between_calls=int(time.time() -
                                                                                       self.last_call_ts)))

        self.step_manager.push_intermediate_step(stats)
        self._run_id_to_llm_input[run_id] = messages[0][-1].content
        self._state = IntermediateStepType.LLM_START
        self.last_call_ts = time.time()
        self._run_id_to_start_time[run_id] = time.time()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Collect stats for just the token"""
        model_name = ""
        try:
            model_name = self._run_id_to_model_name.get(str(kwargs.get("run_id", "")), "")
        except Exception as e:
            logger.exception("Error getting model name: %s", e, exc_info=True)

        usage_metadata = {}
        try:
            usage_metadata = kwargs.get("chunk").message.usage_metadata if kwargs.get("chunk") else {}
        except Exception as e:
            logger.exception("Error getting usage metadata: %s", e, exc_info=True)

        stats = IntermediateStepPayload(
            event_type=IntermediateStepType.LLM_NEW_TOKEN,
            framework=LLMFrameworkEnum.LANGCHAIN,
            name=model_name,
            UUID=str(kwargs.get("run_id", str(uuid4()))),
            data=StreamEventData(input=self._run_id_to_llm_input.get(str(kwargs.get("run_id", "")), ""), chunk=token),
            usage_info=UsageInfo(token_usage=self._extract_token_base_model(usage_metadata),
                                 num_llm_calls=1,
                                 seconds_between_calls=int(time.time() - self.last_call_ts)),
            metadata=TraceMetadata(chat_responses=[kwargs.get("chunk")] if kwargs.get("chunk") else []))

        self.step_manager.push_intermediate_step(stats)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""

        usage_metadata = {}

        model_name = ""
        try:
            model_name = response.llm_output["model_name"]
        except Exception as e:
            try:
                model_name = self._run_id_to_model_name.get(str(kwargs.get("run_id", "")), "")
            except Exception as e_inner:
                logger.exception("Error getting model name: %s from outer error %s", e_inner, e, exc_info=True)

        try:
            generation = response.generations[0][0]
        except IndexError:
            generation = None

        if isinstance(generation, ChatGeneration):
            try:
                message = generation.message
                if isinstance(message, AIMessage):
                    usage_metadata = message.usage_metadata
                else:
                    usage_metadata = {}
            except AttributeError:
                usage_metadata = {}

        llm_text_output = generation.message.content if generation else ""

        # update shared state behind lock
        with self._lock:
            usage_stat = IntermediateStepPayload(
                span_event_timestamp=self._run_id_to_start_time.get(str(kwargs.get("run_id", "")), time.time()),
                event_type=IntermediateStepType.LLM_END,
                framework=LLMFrameworkEnum.LANGCHAIN,
                name=model_name,
                UUID=str(kwargs.get("run_id", str(uuid4()))),
                data=StreamEventData(input=self._run_id_to_llm_input.get(str(kwargs.get("run_id", "")), ""),
                                     output=llm_text_output),
                usage_info=UsageInfo(token_usage=self._extract_token_base_model(usage_metadata)),
                metadata=TraceMetadata(chat_responses=[generation] if generation else []))

            self.step_manager.push_intermediate_step(usage_stat)

        self._state = IntermediateStepType.LLM_END

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:

        stats = IntermediateStepPayload(event_type=IntermediateStepType.TOOL_START,
                                        framework=LLMFrameworkEnum.LANGCHAIN,
                                        name=serialized.get("name", ""),
                                        UUID=str(run_id),
                                        data=StreamEventData(input=input_str),
                                        metadata=TraceMetadata(tool_inputs=copy.deepcopy(inputs),
                                                               tool_info=copy.deepcopy(serialized)),
                                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel()))

        self.step_manager.push_intermediate_step(stats)
        self._run_id_to_tool_input[str(run_id)] = input_str
        self._run_id_to_start_time[str(run_id)] = time.time()

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:

        stats = IntermediateStepPayload(event_type=IntermediateStepType.TOOL_END,
                                        span_event_timestamp=self._run_id_to_start_time.get(str(run_id), time.time()),
                                        framework=LLMFrameworkEnum.LANGCHAIN,
                                        name=kwargs.get("name", ""),
                                        UUID=str(run_id),
                                        metadata=TraceMetadata(tool_outputs=output),
                                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                                        data=StreamEventData(input=self._run_id_to_tool_input.get(str(run_id), ""),
                                                             output=output))

        self.step_manager.push_intermediate_step(stats)
