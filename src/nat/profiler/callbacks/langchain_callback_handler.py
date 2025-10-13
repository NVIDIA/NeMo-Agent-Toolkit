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

from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import ToolSchema
from nat.data_models.intermediate_step import TraceMetadata
from nat.data_models.intermediate_step import UsageInfo
from nat.profiler.callbacks.base_callback_class import BaseProfilerCallback
from nat.profiler.callbacks.token_usage_base_model import TokenUsageBaseModel

logger = logging.getLogger(__name__)


def _extract_tools_schema(invocation_params: dict) -> list:

    tools_schema = []
    if invocation_params is not None:
        for tool in invocation_params.get("tools", []):
            tools_schema.append(ToolSchema(**tool))

    return tools_schema


class LangchainProfilerHandler(AsyncCallbackHandler, BaseProfilerCallback):
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

        self.step_manager = Context.get().intermediate_step_manager
        self._state = IntermediateStepType.LLM_END

        self._run_id_to_model_name = {}
        self._run_id_to_llm_input = {}
        self._run_id_to_tool_input = {}
        self._run_id_to_start_time = {}

        # Node tracking state variables
        self._run_id_to_node_name = {}
        self._run_id_to_node_input = {}

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

    def _should_track_node(self, serialized: dict[str, Any], tags: list[str] | None = None) -> bool:
        """
        Determine if a chain execution should be tracked as a LangGraph node.

        We want to track StateGraph nodes (like 'agent', 'tool', 'planner', etc.)
        but filter out internal LangChain runnables to avoid excessive events.

        Args:
            serialized: Serialized information about the chain
            tags: Tags associated with the chain execution

        Returns:
            True if this should be tracked as a node, False otherwise
        """
        # Get the node name - try different possible locations
        node_name = serialized.get("name", "")
        node_id = serialized.get("id", [])

        # If there's no meaningful name or ID, skip tracking
        if not node_name and not node_id:
            return False

        # Common LangGraph node names we want to track
        # These are typical node names used in NAT agents
        tracked_node_names = {
            "agent", "tool", "planner", "executor", "solver", "branch", "action", "__start__", "__end__"
        }

        # Check if this is a known node name
        if node_name.lower() in tracked_node_names:
            return True

        # Check tags for LangGraph indicators
        if tags:
            # LangGraph typically tags nodes with specific markers
            if any("graph" in tag.lower() or "node" in tag.lower() for tag in tags):
                return True

        # Check the ID structure - LangGraph nodes have specific patterns
        if isinstance(node_id, list) and len(node_id) > 0:
            # LangGraph nodes often have "RunnableLambda" or node names in their ID
            id_str = " ".join(str(i) for i in node_id)
            if any(name in id_str for name in tracked_node_names):
                return True

        # Skip common internal runnables to reduce noise
        skip_patterns = [
            "RunnableSequence",
            "RunnableParallel",
            "RunnablePassthrough",
            "RunnableLambda",
            "RunnableBranch",
            "RunnableBinding"
        ]

        # If this looks like an internal runnable and doesn't match our patterns, skip it
        if any(pattern in node_name for pattern in skip_patterns):
            return False

        # For anything else with a meaningful name, track it
        # This allows custom node names to be tracked
        if len(node_name) > 0 and not node_name.startswith("_"):
            return True

        return False

    async def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:

        model_name = ""
        try:
            model_name = kwargs.get("metadata")["ls_model_name"]
        except Exception as e:
            logger.exception("Error getting model name: %s", e)

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
            logger.exception("Error getting model name: %s", e)

        run_id = str(run_id)
        self._run_id_to_model_name[run_id] = model_name

        stats = IntermediateStepPayload(
            event_type=IntermediateStepType.LLM_START,
            framework=LLMFrameworkEnum.LANGCHAIN,
            name=model_name,
            UUID=run_id,
            data=StreamEventData(input=copy.deepcopy(messages[0])),
            metadata=TraceMetadata(chat_inputs=copy.deepcopy(messages[0]),
                                   tools_schema=_extract_tools_schema(kwargs.get("invocation_params", {}))),
            usage_info=UsageInfo(token_usage=TokenUsageBaseModel(),
                                 num_llm_calls=1,
                                 seconds_between_calls=int(time.time() - self.last_call_ts)))

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
            logger.exception("Error getting model name: %s", e)

        usage_metadata = {}
        try:
            usage_metadata = kwargs.get("chunk").message.usage_metadata if kwargs.get("chunk") else {}
        except Exception as e:
            logger.exception("Error getting usage metadata: %s", e)

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
                logger.exception("Error getting model name: %s from outer error %s", e_inner, e)

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

        if generation:
            llm_text_output = generation.message.content
            if "tool_calls" in generation.message.additional_kwargs:
                # add tool calls if included in the output
                tool_calls = generation.message.additional_kwargs['tool_calls']
                llm_text_output = f"{llm_text_output}\n\nTool calls: {tool_calls}"
        else:
            llm_text_output = ""

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

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Track when a LangGraph node starts execution.

        This callback is triggered when any chain/runnable starts, but we filter
        to only track meaningful LangGraph nodes using _should_track_node.
        """
        # Check if we should track this node
        if not self._should_track_node(serialized, tags):
            return

        # Extract node name
        node_name = serialized.get("name", serialized.get("id", ["unknown"])[0] if serialized.get("id") else "unknown")

        run_id_str = str(run_id)
        self._run_id_to_node_name[run_id_str] = node_name
        self._run_id_to_start_time[run_id_str] = time.time()

        # Store inputs (but limit size to avoid memory issues with large states)
        try:
            node_input = copy.deepcopy(inputs)
            self._run_id_to_node_input[run_id_str] = node_input
        except Exception as e:
            logger.debug("Could not copy node inputs for node %s: %s", node_name, e)
            self._run_id_to_node_input[run_id_str] = {"error": "Failed to copy inputs"}

        # Create node info metadata
        node_info = {
            "node_name": node_name,
            "node_type": serialized.get("id", ["unknown"])[0] if serialized.get("id") else "unknown",
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
        }

        stats = IntermediateStepPayload(event_type=IntermediateStepType.NODE_START,
                                        framework=LLMFrameworkEnum.LANGCHAIN,
                                        name=node_name,
                                        UUID=run_id_str,
                                        tags=tags,
                                        data=StreamEventData(input=inputs),
                                        metadata=TraceMetadata(node_inputs=copy.deepcopy(inputs) if inputs else None,
                                                               node_info=node_info,
                                                               provided_metadata=metadata),
                                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel()))

        self.step_manager.push_intermediate_step(stats)
        logger.debug("Tracked NODE_START for node: %s (run_id: %s)", node_name, run_id_str)

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Track when a LangGraph node completes execution.

        This callback is triggered when any chain/runnable ends. We only process
        it if we tracked a corresponding NODE_START event.
        """
        run_id_str = str(run_id)

        # Only track if we recorded a start event for this node
        node_name = self._run_id_to_node_name.get(run_id_str)
        if not node_name:
            return

        # Get start time and calculate duration
        start_time = self._run_id_to_start_time.get(run_id_str, time.time())
        duration = time.time() - start_time

        # Get the original inputs
        node_input = self._run_id_to_node_input.get(run_id_str, {})

        # Create node info with duration
        node_info = {
            "node_name": node_name,
            "duration_seconds": duration,
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
        }

        stats = IntermediateStepPayload(event_type=IntermediateStepType.NODE_END,
                                        span_event_timestamp=start_time,
                                        framework=LLMFrameworkEnum.LANGCHAIN,
                                        name=node_name,
                                        UUID=run_id_str,
                                        data=StreamEventData(input=node_input, output=outputs),
                                        metadata=TraceMetadata(node_inputs=node_input,
                                                               node_outputs=copy.deepcopy(outputs) if outputs else None,
                                                               node_info=node_info),
                                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel()))

        self.step_manager.push_intermediate_step(stats)
        logger.debug("Tracked NODE_END for node: %s (run_id: %s, duration: %.3fs)", node_name, run_id_str, duration)

        # Clean up tracking dictionaries
        self._run_id_to_node_name.pop(run_id_str, None)
        self._run_id_to_node_input.pop(run_id_str, None)
        self._run_id_to_start_time.pop(run_id_str, None)
