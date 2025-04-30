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

from llama_index.core.callbacks import CBEventType
from llama_index.core.callbacks import EventPayload
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.llms import ChatResponse

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


class LlamaIndexProfilerHandler(BaseCallbackHandler, BaseProfilerCallback):
    """
    A callback handler for LlamaIndex that tracks usage stats similarly to NIMCallbackHandler.
    Collects:

    - Prompts
    - Token usage
    - Response data
    - Time intervals between calls

    and appends them to AIQContextState.usage_stats.
    """

    def __init__(self) -> None:
        BaseCallbackHandler.__init__(self, event_starts_to_ignore=[], event_ends_to_ignore=[])
        BaseProfilerCallback.__init__(self)
        self._lock = threading.Lock()
        self.last_call_ts = time.time()
        self._last_tool_map: dict[str, str] = {}
        self.step_manager = AIQContext.get().intermediate_step_manager

        self._run_id_to_llm_input = {}
        self._run_id_to_tool_input = {}
        self._run_id_to_timestamp = {}

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """
        Called at the *start* of a LlamaIndex "event" (LLM call, Embedding, etc.).
        We capture the prompts or query strings here, if any.
        """

        prompts_or_messages = None
        now = time.time()
        seconds_between_calls = int(now - self.last_call_ts)

        # For LLM or chat calls, look in `payload` for messages/prompts
        if event_type == CBEventType.LLM and payload:
            # For example, "PROMPT" or "MESSAGES" might be in the payload.
            # If found, store them in usage stats (just like your NIMCallbackHandler).
            if EventPayload.PROMPT in payload:
                prompts_or_messages = [payload[EventPayload.PROMPT]]
            elif EventPayload.MESSAGES in payload:
                prompts_or_messages = [str(msg) for msg in payload[EventPayload.MESSAGES]]

            model_name = ""
            try:
                model_name = payload.get(EventPayload.SERIALIZED)['model']
            except Exception as e:
                logger.exception("Error getting model name: %s", e, exc_info=True)

            llm_text_input = " ".join(prompts_or_messages) if prompts_or_messages else ""

            if prompts_or_messages:
                stats = IntermediateStepPayload(event_type=IntermediateStepType.LLM_START,
                                                framework=LLMFrameworkEnum.LLAMA_INDEX,
                                                name=model_name,
                                                UUID=event_id,
                                                data=StreamEventData(input=llm_text_input),
                                                metadata=TraceMetadata(chat_inputs=copy.deepcopy(prompts_or_messages)),
                                                usage_info=UsageInfo(token_usage=TokenUsageBaseModel(),
                                                                     num_llm_calls=1,
                                                                     seconds_between_calls=seconds_between_calls))

                self.step_manager.push_intermediate_step(stats)
                self._run_id_to_llm_input[event_id] = llm_text_input
                self.last_call_ts = now
                self._run_id_to_timestamp[event_id] = time.time()

        elif event_type == CBEventType.FUNCTION_CALL and payload:
            tool_metadata = payload.get(EventPayload.TOOL)
            tool_metadata = {
                "description": tool_metadata.description if hasattr(tool_metadata, "description") else "",
                "fn_schema_str": tool_metadata.fn_schema_str if hasattr(tool_metadata, "fn_schema_str") else "",
                "name": tool_metadata.name if hasattr(tool_metadata, "name") else "",
            }
            stats = IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_START,
                framework=LLMFrameworkEnum.LLAMA_INDEX,
                name=payload.get(EventPayload.TOOL).name,
                UUID=event_id,
                data=StreamEventData(input=copy.deepcopy(payload.get(EventPayload.FUNCTION_CALL))),
                metadata=TraceMetadata(tool_inputs=copy.deepcopy(payload.get(EventPayload.FUNCTION_CALL)),
                                       tool_info=tool_metadata),
                usage_info=UsageInfo(token_usage=TokenUsageBaseModel()))

            self._run_id_to_tool_input[event_id] = copy.deepcopy(payload.get(EventPayload.FUNCTION_CALL))
            self._last_tool_map[event_id] = payload.get(EventPayload.TOOL).name
            self.step_manager.push_intermediate_step(stats)
            self._run_id_to_timestamp[event_id] = time.time()
        return event_id  # must return the event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Called at the *end* of a LlamaIndex "event".
        We collect token usage (if available) and the returned response text.
        """

        if payload and event_type == CBEventType.LLM:
            # Often, token usage is embedded in e.g. payload["RESPONSE"].raw["usage"] for OpenAI-based calls
            response = payload.get(EventPayload.RESPONSE)
            if isinstance(response, ChatResponse):
                llm_text_output = ""

                try:
                    for block in response.message.blocks:
                        llm_text_output += block.text
                except Exception as e:
                    logger.exception("Error getting LLM text output: %s", e, exc_info=True)

                model_name = ""
                try:
                    model_name = response.raw.model
                except Exception as e:
                    logger.exception("Error getting model name: %s", e, exc_info=True)

                # Append usage data to AIQ Toolkit usage stats
                with self._lock:
                    stats = IntermediateStepPayload(
                        event_type=IntermediateStepType.LLM_END,
                        span_event_timestamp=self._run_id_to_timestamp.get(event_id),
                        framework=LLMFrameworkEnum.LLAMA_INDEX,
                        name=model_name,
                        UUID=event_id,
                        data=StreamEventData(input=self._run_id_to_llm_input.get(event_id), output=llm_text_output),
                        metadata=TraceMetadata(chat_responses=response.message if response.message else None),
                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel(**response.additional_kwargs)))
                    self.step_manager.push_intermediate_step(stats)

        elif event_type == CBEventType.FUNCTION_CALL and payload:
            stats = IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_END,
                span_event_timestamp=self._run_id_to_timestamp.get(event_id),
                framework=LLMFrameworkEnum.LLAMA_INDEX,
                name=self._last_tool_map.get(event_id),
                UUID=event_id,
                data=StreamEventData(output=copy.deepcopy(payload.get(EventPayload.FUNCTION_OUTPUT))),
                usage_info=UsageInfo(token_usage=TokenUsageBaseModel()))

            self.step_manager.push_intermediate_step(stats)

    def start_trace(self, trace_id: str | None = None) -> None:
        """Run when an overall trace is launched."""
        pass

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        """Run when an overall trace is exited."""
        pass
