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
"""AutoGen callback handler for usage statistics collection.

This module provides profiling instrumentation for AutoGen agents by monkey-patching
LLM client and tool classes to collect telemetry data.

Supported LLM Clients
---------------------
- ``OpenAIChatCompletionClient``: OpenAI and OpenAI-compatible APIs (NIM, LiteLLM)
- ``AzureOpenAIChatCompletionClient``: Azure OpenAI deployments
- ``AnthropicBedrockChatCompletionClient``: AWS Bedrock (Anthropic models)

Supported Methods
-----------------
- ``create``: Non-streaming LLM completions
- ``create_stream``: Streaming LLM completions
- ``BaseTool.run_json``: Tool executions
"""

import copy
import logging
import threading
import time
from collections.abc import AsyncGenerator
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
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


@dataclass
class ClientPatchInfo:
    """Stores original method references for a patched client class."""

    create: Callable[..., Any] | None = None
    create_stream: Callable[..., Any] | None = None


@dataclass
class PatchedClients:
    """Stores all patched client information for restoration."""

    openai: ClientPatchInfo = field(default_factory=ClientPatchInfo)
    azure: ClientPatchInfo = field(default_factory=ClientPatchInfo)
    bedrock: ClientPatchInfo = field(default_factory=ClientPatchInfo)
    tool: Callable[..., Any] | None = None


class AutoGenProfilerHandler(BaseProfilerCallback):
    """Callback handler for AutoGen that intercepts LLM and tool calls for profiling.

    This handler monkey-patches AutoGen client classes to collect usage statistics
    including token usage, inputs, outputs, and timing information.

    Supported clients:
        - OpenAIChatCompletionClient (OpenAI, NIM, LiteLLM)
        - AzureOpenAIChatCompletionClient (Azure OpenAI)
        - AnthropicBedrockChatCompletionClient (AWS Bedrock)

    Supported methods:
        - create (non-streaming)
        - create_stream (streaming)
        - BaseTool.run_json (tool execution)

    Example:
        >>> handler = AutoGenProfilerHandler()
        >>> handler.instrument()
        >>> # ... run AutoGen workflow ...
        >>> handler.uninstrument()
    """

    def __init__(self) -> None:
        """Initialize the AutoGenProfilerHandler."""
        super().__init__()
        self._lock = threading.Lock()
        self.last_call_ts = time.time()
        self.step_manager = Context.get().intermediate_step_manager
        self._patched = PatchedClients()
        self._instrumented = False

    def instrument(self) -> None:
        """Monkey-patch AutoGen methods with usage-stat collection logic.

        Patches the following classes if available:
            - OpenAIChatCompletionClient.create, create_stream
            - AzureOpenAIChatCompletionClient.create, create_stream
            - AnthropicBedrockChatCompletionClient.create
            - BaseTool.run_json

        Does nothing if already instrumented or if imports fail.
        """
        if self._instrumented:
            logger.debug("AutoGenProfilerHandler already instrumented; skipping.")
            return

        # Import and patch tool class
        try:
            from autogen_core.tools import BaseTool
            self._patched.tool = getattr(BaseTool, "run_json", None)
            if self._patched.tool:
                BaseTool.run_json = self._create_tool_wrapper(self._patched.tool)
                logger.debug("Patched BaseTool.run_json")
        except ImportError:
            logger.debug("autogen_core.tools not available; skipping tool instrumentation")

        # Import and patch OpenAI client
        try:
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            self._patched.openai.create = getattr(OpenAIChatCompletionClient, "create", None)
            self._patched.openai.create_stream = getattr(OpenAIChatCompletionClient, "create_stream", None)

            if self._patched.openai.create:
                OpenAIChatCompletionClient.create = self._create_llm_wrapper(self._patched.openai.create)
                logger.debug("Patched OpenAIChatCompletionClient.create")
            if self._patched.openai.create_stream:
                OpenAIChatCompletionClient.create_stream = self._create_stream_wrapper(
                    self._patched.openai.create_stream)
                logger.debug("Patched OpenAIChatCompletionClient.create_stream")
        except ImportError:
            logger.debug("autogen_ext.models.openai not available; skipping OpenAI instrumentation")

        # Import and patch Azure client
        try:
            from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
            self._patched.azure.create = getattr(AzureOpenAIChatCompletionClient, "create", None)
            self._patched.azure.create_stream = getattr(AzureOpenAIChatCompletionClient, "create_stream", None)

            if self._patched.azure.create:
                AzureOpenAIChatCompletionClient.create = self._create_llm_wrapper(self._patched.azure.create)
                logger.debug("Patched AzureOpenAIChatCompletionClient.create")
            if self._patched.azure.create_stream:
                AzureOpenAIChatCompletionClient.create_stream = self._create_stream_wrapper(
                    self._patched.azure.create_stream)
                logger.debug("Patched AzureOpenAIChatCompletionClient.create_stream")
        except ImportError:
            logger.debug("AzureOpenAIChatCompletionClient not available; skipping Azure instrumentation")

        # Import and patch Bedrock client
        try:
            from autogen_ext.models.anthropic import AnthropicBedrockChatCompletionClient
            self._patched.bedrock.create = getattr(AnthropicBedrockChatCompletionClient, "create", None)

            if self._patched.bedrock.create:
                AnthropicBedrockChatCompletionClient.create = self._create_llm_wrapper(self._patched.bedrock.create)
                logger.debug("Patched AnthropicBedrockChatCompletionClient.create")
            # Note: Bedrock client may not have create_stream - check if available
            if hasattr(AnthropicBedrockChatCompletionClient, "create_stream"):
                self._patched.bedrock.create_stream = getattr(AnthropicBedrockChatCompletionClient,
                                                              "create_stream",
                                                              None)
                if self._patched.bedrock.create_stream:
                    AnthropicBedrockChatCompletionClient.create_stream = self._create_stream_wrapper(
                        self._patched.bedrock.create_stream)
                    logger.debug("Patched AnthropicBedrockChatCompletionClient.create_stream")
        except ImportError:
            logger.debug("autogen_ext.models.anthropic not available; skipping Bedrock instrumentation")

        self._instrumented = True
        logger.debug("AutoGenProfilerHandler instrumentation applied successfully.")

    def uninstrument(self) -> None:
        """Restore original AutoGen methods.

        Should be called to clean up monkey patches, especially in test environments.
        """
        try:
            # Restore tool
            if self._patched.tool:
                from autogen_core.tools import BaseTool
                BaseTool.run_json = self._patched.tool
                logger.debug("Restored BaseTool.run_json")

            # Restore OpenAI client
            if self._patched.openai.create or self._patched.openai.create_stream:
                from autogen_ext.models.openai import OpenAIChatCompletionClient
                if self._patched.openai.create:
                    OpenAIChatCompletionClient.create = self._patched.openai.create
                if self._patched.openai.create_stream:
                    OpenAIChatCompletionClient.create_stream = self._patched.openai.create_stream
                logger.debug("Restored OpenAIChatCompletionClient methods")

            # Restore Azure client
            if self._patched.azure.create or self._patched.azure.create_stream:
                from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
                if self._patched.azure.create:
                    AzureOpenAIChatCompletionClient.create = self._patched.azure.create
                if self._patched.azure.create_stream:
                    AzureOpenAIChatCompletionClient.create_stream = self._patched.azure.create_stream
                logger.debug("Restored AzureOpenAIChatCompletionClient methods")

            # Restore Bedrock client
            if self._patched.bedrock.create or self._patched.bedrock.create_stream:
                from autogen_ext.models.anthropic import AnthropicBedrockChatCompletionClient
                if self._patched.bedrock.create:
                    AnthropicBedrockChatCompletionClient.create = self._patched.bedrock.create
                if self._patched.bedrock.create_stream:
                    AnthropicBedrockChatCompletionClient.create_stream = self._patched.bedrock.create_stream
                logger.debug("Restored AnthropicBedrockChatCompletionClient methods")

            # Reset state
            self._patched = PatchedClients()
            self._instrumented = False
            logger.debug("AutoGenProfilerHandler uninstrumented successfully.")

        except Exception:
            logger.exception("Failed to uninstrument AutoGenProfilerHandler")

    def _extract_model_name(self, client: Any) -> str:
        """Extract model name from AutoGen client instance.

        Args:
            client: AutoGen chat completion client instance

        Returns:
            str: Model name or 'unknown_model' if extraction fails
        """
        try:
            raw_config = getattr(client, "_raw_config", {})
            if raw_config and "model" in raw_config:
                return str(raw_config["model"])
        except Exception:
            logger.debug("Failed to extract model from _raw_config")

        try:
            return str(getattr(client, "model", "unknown_model"))
        except Exception:
            return "unknown_model"

    def _extract_input_text(self, messages: list[Any]) -> str:
        """Extract text content from message list.

        Handles both dict-style messages and AutoGen typed message objects
        (UserMessage, AssistantMessage, SystemMessage).

        Args:
            messages: List of message dictionaries or AutoGen message objects

        Returns:
            str: Concatenated text content from messages
        """
        model_input = ""
        try:
            for message in messages:
                # Handle dict-style messages
                if isinstance(message, dict):
                    content = message.get("content", "")
                # Handle AutoGen typed message objects (UserMessage, AssistantMessage, etc.)
                elif hasattr(message, "content"):
                    content = message.content
                else:
                    # Fallback to string conversion
                    content = str(message)

                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            model_input += str(part.get("text", ""))
                        else:
                            model_input += str(part)
                else:
                    model_input += str(content) if content else ""
        except Exception:
            logger.debug("Error extracting input text from messages")
        return model_input

    def _extract_output_text(self, output: Any) -> str:
        """Extract text content from LLM response.

        Args:
            output: LLM response object

        Returns:
            str: Concatenated text content from response
        """
        model_output = ""
        try:
            for content in output.content:
                model_output += str(content) if content else ""
        except Exception:
            logger.debug("Error extracting output text from response")
        return model_output

    def _extract_usage(self, output: Any) -> dict[str, Any]:
        """Extract token usage from LLM response.

        Args:
            output: LLM response object

        Returns:
            dict: Token usage dictionary
        """
        try:
            usage_obj = getattr(output, "usage", None)
            if not usage_obj:
                usage_obj = (getattr(output, "model_extra", {}) or {}).get("usage")
            if usage_obj:
                if hasattr(usage_obj, "model_dump"):
                    return usage_obj.model_dump()
                elif isinstance(usage_obj, dict):
                    return usage_obj
        except Exception:
            logger.debug("Error extracting usage from response")
        return {}

    def _extract_chat_response(self, output: Any) -> dict[str, Any]:
        """Extract chat response metadata from LLM response.

        Args:
            output: LLM response object

        Returns:
            dict: Chat response metadata
        """
        try:
            choices = getattr(output, "choices", [])
            if choices:
                first_choice = choices[0]
                if hasattr(first_choice, "model_dump"):
                    return first_choice.model_dump()
                return getattr(first_choice, "__dict__", {}) or {}
        except Exception:
            logger.debug("Error extracting chat response metadata")
        return {}

    def _create_llm_wrapper(self, original_func: Callable[..., Any]) -> Callable[..., Any]:
        """Create wrapper for non-streaming LLM calls.

        Args:
            original_func: Original create method to wrap

        Returns:
            Callable: Wrapped function with profiling
        """
        handler = self

        async def wrapped_llm_call(*args: Any, **kwargs: Any) -> Any:
            now = time.time()
            with handler._lock:
                seconds_between_calls = int(now - handler.last_call_ts)

            # Extract model info
            client = args[0] if args else None
            model_name = handler._extract_model_name(client) if client else "unknown_model"
            messages = kwargs.get("messages", [])
            model_input = handler._extract_input_text(messages)

            # Push LLM_START event
            start_payload = IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                framework=LLMFrameworkEnum.AUTOGEN,
                name=model_name,
                data=StreamEventData(input=model_input),
                metadata=TraceMetadata(chat_inputs=copy.deepcopy(messages)),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(),
                    num_llm_calls=1,
                    seconds_between_calls=seconds_between_calls,
                ),
            )
            start_uuid = start_payload.UUID
            handler.step_manager.push_intermediate_step(start_payload)

            # Call original function
            try:
                output = await original_func(*args, **kwargs)
            except Exception as e:
                logger.error("Error during LLM call: %s", e)
                handler.step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        event_type=IntermediateStepType.LLM_END,
                        span_event_timestamp=time.time(),
                        framework=LLMFrameworkEnum.AUTOGEN,
                        name=model_name,
                        data=StreamEventData(input=model_input, output=str(e)),
                        metadata=TraceMetadata(error=str(e)),
                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                        UUID=start_uuid,
                    ))
                with handler._lock:
                    handler.last_call_ts = time.time()
                raise

            # Extract response data
            model_output = handler._extract_output_text(output)
            usage_payload = handler._extract_usage(output)
            chat_resp = handler._extract_chat_response(output)

            # Push LLM_END event
            end_time = time.time()
            handler.step_manager.push_intermediate_step(
                IntermediateStepPayload(
                    event_type=IntermediateStepType.LLM_END,
                    span_event_timestamp=end_time,
                    framework=LLMFrameworkEnum.AUTOGEN,
                    name=model_name,
                    data=StreamEventData(input=model_input, output=model_output),
                    metadata=TraceMetadata(chat_responses=chat_resp),
                    usage_info=UsageInfo(
                        token_usage=TokenUsageBaseModel(**usage_payload),
                        num_llm_calls=1,
                        seconds_between_calls=seconds_between_calls,
                    ),
                    UUID=start_uuid,
                ))

            with handler._lock:
                handler.last_call_ts = end_time

            return output

        return wrapped_llm_call

    def _create_stream_wrapper(self, original_func: Callable[..., Any]) -> Callable[..., Any]:
        """Create wrapper for streaming LLM calls.

        Args:
            original_func: Original create_stream method to wrap

        Returns:
            Callable: Wrapped function with profiling
        """
        handler = self

        async def wrapped_stream_call(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            now = time.time()
            with handler._lock:
                seconds_between_calls = int(now - handler.last_call_ts)

            # Extract model info
            client = args[0] if args else None
            model_name = handler._extract_model_name(client) if client else "unknown_model"
            messages = kwargs.get("messages", [])
            model_input = handler._extract_input_text(messages)

            # Push LLM_START event
            start_payload = IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                framework=LLMFrameworkEnum.AUTOGEN,
                name=model_name,
                data=StreamEventData(input=model_input),
                metadata=TraceMetadata(chat_inputs=copy.deepcopy(messages)),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(),
                    num_llm_calls=1,
                    seconds_between_calls=seconds_between_calls,
                ),
            )
            start_uuid = start_payload.UUID
            handler.step_manager.push_intermediate_step(start_payload)

            # Collect streaming output
            output_chunks: list[str] = []
            usage_payload: dict[str, Any] = {}

            try:
                async for chunk in original_func(*args, **kwargs):
                    # Extract text from chunk if available
                    try:
                        if hasattr(chunk, "content") and chunk.content:
                            output_chunks.append(str(chunk.content))
                        # Check for usage in final chunk
                        if hasattr(chunk, "usage") and chunk.usage:
                            if hasattr(chunk.usage, "model_dump"):
                                usage_payload = chunk.usage.model_dump()
                            elif isinstance(chunk.usage, dict):
                                usage_payload = chunk.usage
                    except Exception:
                        pass
                    yield chunk

                # Success path - push LLM_END event after stream completes
                end_time = time.time()
                model_output = "".join(output_chunks)
                handler.step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        event_type=IntermediateStepType.LLM_END,
                        span_event_timestamp=end_time,
                        framework=LLMFrameworkEnum.AUTOGEN,
                        name=model_name,
                        data=StreamEventData(input=model_input, output=model_output),
                        metadata=TraceMetadata(chat_responses={}),
                        usage_info=UsageInfo(
                            token_usage=TokenUsageBaseModel(**usage_payload),
                            num_llm_calls=1,
                            seconds_between_calls=seconds_between_calls,
                        ),
                        UUID=start_uuid,
                    ))
                with handler._lock:
                    handler.last_call_ts = end_time

            except Exception as e:
                # Error path - push error LLM_END event
                logger.error("Error during streaming LLM call: %s", e)
                handler.step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        event_type=IntermediateStepType.LLM_END,
                        span_event_timestamp=time.time(),
                        framework=LLMFrameworkEnum.AUTOGEN,
                        name=model_name,
                        data=StreamEventData(input=model_input, output=str(e)),
                        metadata=TraceMetadata(error=str(e)),
                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                        UUID=start_uuid,
                    ))
                with handler._lock:
                    handler.last_call_ts = time.time()
                raise

        return wrapped_stream_call

    def _create_tool_wrapper(self, original_func: Callable[..., Any]) -> Callable[..., Any]:
        """Create wrapper for tool execution calls.

        Args:
            original_func: Original run_json method to wrap

        Returns:
            Callable: Wrapped function with profiling
        """
        handler = self

        async def wrapped_tool_call(*args: Any, **kwargs: Any) -> Any:
            now = time.time()
            with handler._lock:
                seconds_between_calls = int(now - handler.last_call_ts)

            # Extract tool name
            tool_name = "unknown_tool"
            try:
                tool_name = str(getattr(args[0], "name", "unknown_tool"))
            except Exception:
                logger.debug("Error getting tool name")

            # Extract tool input
            tool_input = ""
            try:
                if len(args) > 1:
                    call_data = args[1]
                    if hasattr(call_data, "kwargs"):
                        tool_input = str(call_data.kwargs)
                    elif isinstance(call_data, dict):
                        tool_input = str(call_data.get("kwargs", {}))
            except Exception:
                logger.debug("Error extracting tool input")

            # Push TOOL_START event
            start_payload = IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_START,
                framework=LLMFrameworkEnum.AUTOGEN,
                name=tool_name,
                data=StreamEventData(input=tool_input),
                metadata=TraceMetadata(tool_inputs={"input": tool_input}),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(),
                    num_llm_calls=0,
                    seconds_between_calls=seconds_between_calls,
                ),
            )
            start_uuid = start_payload.UUID
            handler.step_manager.push_intermediate_step(start_payload)

            # Call original function
            try:
                output = await original_func(*args, **kwargs)
            except Exception as e:
                logger.error("Tool execution failed: %s", e)
                handler.step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        event_type=IntermediateStepType.TOOL_END,
                        span_event_timestamp=time.time(),
                        framework=LLMFrameworkEnum.AUTOGEN,
                        name=tool_name,
                        data=StreamEventData(input=tool_input, output=str(e)),
                        metadata=TraceMetadata(error=str(e)),
                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                        UUID=start_uuid,
                    ))
                with handler._lock:
                    handler.last_call_ts = time.time()
                raise

            # Push TOOL_END event
            end_time = time.time()
            handler.step_manager.push_intermediate_step(
                IntermediateStepPayload(
                    event_type=IntermediateStepType.TOOL_END,
                    span_event_timestamp=end_time,
                    framework=LLMFrameworkEnum.AUTOGEN,
                    name=tool_name,
                    data=StreamEventData(input=tool_input, output=str(output)),
                    metadata=TraceMetadata(tool_outputs={"result": str(output)}),
                    usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                    UUID=start_uuid,
                ))

            with handler._lock:
                handler.last_call_ts = end_time

            return output

        return wrapped_tool_call
