# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Runtime tool-calling agent implementation."""

import json
import logging
from collections.abc import AsyncGenerator
from collections.abc import Iterable
from typing import Any
from typing import cast

from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionMessageParam
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.agent import AgentBaseConfig
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.utils.type_converter import GlobalTypeConverter

from .llm import RuntimeOpenAIClient
from .tool_wrapper import RuntimeToolWrapper

logger = logging.getLogger(__name__)


class RuntimeToolCallingAgentConfig(AgentBaseConfig, name="runtime_tool_calling_agent"):
    """Configuration for the runtime tool-calling agent."""

    description: str = Field(default="Runtime Tool Calling Agent Workflow",
                             description="Description of this functions use.")
    tool_names: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list,
        description="The list of tools to provide to the tool calling agent.",
    )
    handle_tool_errors: bool = Field(
        default=True,
        description="Specify ability to handle tool calling errors.",
    )
    max_iterations: int = Field(
        default=15,
        description="Number of tool calls before stopping the tool calling agent.",
    )
    max_history: int = Field(
        default=15,
        description="Maximum number of messages to keep in the conversation history.",
    )
    system_prompt: str = Field(
        default="You are a helpful assistant that can use tools to help the user. " + \
                "You will be given a task and you will need to use the tools to " + \
                "complete the task. Only attempt to use the tools. Do not make up " + \
                "information. Only use the tools to answer the user's question.",
        description="Provides the system prompt to use with the agent.",
    )
    additional_instructions: str | None = Field(
        default=None,
        description="Additional instructions appended to the system prompt.",
    )
    return_direct: list[FunctionRef] | None = Field(
        default=None,
        description="List of tool names that should return responses directly without LLM processing.",
    )


def _build_tool_calling_prompt(config: RuntimeToolCallingAgentConfig) -> str | None:
    """Create a tool-calling agent system prompt from config."""

    prompt_parts = []
    for prompt in (config.system_prompt, config.additional_instructions):
        if prompt:
            prompt_parts.append(prompt)
    if prompt_parts:
        return " ".join(prompt_parts)
    return None


def _trim_history(messages: list[dict[str, Any]], max_history: int) -> list[dict[str, Any]]:
    """Trim message history to the most recent entries."""

    if max_history <= 0 or len(messages) <= max_history:
        return messages
    return messages[-max_history:]


def _tool_message(tool_call_id: str, content: str) -> dict[str, Any]:
    """Create a tool response message in OpenAI format."""

    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }


def _serialize_tool_output(output: object) -> str:
    """Serialize tool output to a string for the LLM."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, default=str)
    except TypeError:
        return str(output)


def _extract_text_from_chunk(chunk: ChatCompletionChunk) -> str | None:
    """Extract text content from a streaming chat completion chunk."""

    if chunk.choices:
        delta = chunk.choices[0].delta
        if delta is not None and delta.content is not None:
            return delta.content
    return None


def _build_messages(chat_request: ChatRequest, system_prompt: str | None, max_history: int) -> list[dict[str, Any]]:
    """Build OpenAI chat messages from a request."""

    messages = [message.model_dump(exclude_none=True) for message in chat_request.messages]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return _trim_history(messages, max_history)


def _get_tool_calls(message: object) -> list[object]:
    """Extract tool calls from an OpenAI assistant message."""

    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return []
    return list(tool_calls)


def _tool_names(tools: Iterable[RuntimeToolWrapper]) -> set[str]:
    """Return a set of tool names from runtime tools."""

    return {tool.name for tool in tools}


async def _resolve_tool_calls(*,
                              llm: RuntimeOpenAIClient,
                              messages: list[dict[str, Any]],
                              request_params: dict[str, Any],
                              tools_by_name: dict[str, RuntimeToolWrapper],
                              return_direct_names: set[str],
                              max_iterations: int,
                              max_history: int,
                              handle_tool_errors: bool) -> tuple[str, str]:
    """Run tool-calling loop until final response or return-direct tool output."""

    request_params = dict(request_params)
    request_params["stream"] = False

    for iteration in range(max_iterations + 1):
        chat_messages = cast(list[ChatCompletionMessageParam], messages)
        response = cast(ChatCompletion, await llm.chat_completions(messages=chat_messages, **request_params))
        choice = response.choices[0]
        assistant_message = choice.message
        assistant_payload = assistant_message.model_dump(exclude_none=True)

        tool_calls = _get_tool_calls(assistant_message)
        if not tool_calls:
            return "final", assistant_message.content or ""

        if iteration >= max_iterations:
            raise RuntimeError("Runtime tool-calling agent exceeded max_iterations.")

        messages.append(assistant_payload)

        for tool_call in tool_calls:
            tool_call_id = getattr(tool_call, "id", None)
            function_call = getattr(tool_call, "function", None)
            tool_name = getattr(function_call, "name", None) if function_call else None
            arguments = getattr(function_call, "arguments", None) if function_call else None

            if not tool_call_id or not tool_name:
                raise ValueError("Runtime tool-calling agent received malformed tool call.")

            tool = tools_by_name.get(tool_name)
            if tool is None:
                raise ValueError(f"Runtime tool-calling agent received unknown tool '{tool_name}'.")

            parsed_args: dict[str, Any] = {}
            if arguments:
                try:
                    parsed_args = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    if handle_tool_errors:
                        logger.warning("Failed to parse tool arguments for %s: %s", tool_name, exc)
                        messages.append(_tool_message(tool_call_id, f"Error parsing tool arguments: {exc}"))
                        continue
                    raise

            try:
                tool_output = await tool.invoke(**parsed_args)
            except Exception as exc:
                if handle_tool_errors:
                    logger.warning("Tool %s failed: %s", tool_name, exc)
                    messages.append(_tool_message(tool_call_id, f"Tool error: {exc}"))
                    continue
                raise

            serialized_output = _serialize_tool_output(tool_output)
            if tool_name in return_direct_names:
                return "direct", serialized_output

            messages.append(_tool_message(tool_call_id, serialized_output))

        messages[:] = _trim_history(messages, max_history)

    raise RuntimeError("Runtime tool-calling agent did not complete.")


@register_function(config_type=RuntimeToolCallingAgentConfig, framework_wrappers=[LLMFrameworkEnum.RUNTIME])
async def runtime_tool_calling_agent_workflow(config: RuntimeToolCallingAgentConfig, builder: Builder):
    """Build a runtime tool-calling agent workflow."""

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.RUNTIME)
    if not isinstance(llm, RuntimeOpenAIClient):
        raise TypeError("Runtime tool-calling agent requires a RuntimeOpenAIClient LLM wrapper.")

    tools = await builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.RUNTIME)
    if not tools:
        raise ValueError(f"No tools specified for runtime tool-calling agent '{config.llm_name}'")

    return_direct_tools = await builder.get_tools(
        tool_names=config.return_direct,
        wrapper_type=LLMFrameworkEnum.RUNTIME,
    ) if config.return_direct else None
    return_direct_names = _tool_names(return_direct_tools) if return_direct_tools else set()

    tools_by_name = {tool.name: tool for tool in tools}
    tool_schemas = [tool.tool_schema for tool in tools]

    system_prompt = _build_tool_calling_prompt(config)

    async def _response_fn(chat_request_or_message: ChatRequestOrMessage) -> str:
        """Execute the runtime tool-calling agent and return its response."""

        chat_request = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequest)
        messages = _build_messages(chat_request, system_prompt, config.max_history)
        request_params = chat_request.model_dump(exclude={"messages"}, exclude_none=True, exclude_unset=True)
        request_params["tools"] = tool_schemas
        result_type, result_content = await _resolve_tool_calls(
            llm=llm,
            messages=messages,
            request_params=request_params,
            tools_by_name=tools_by_name,
            return_direct_names=return_direct_names,
            max_iterations=config.max_iterations,
            max_history=config.max_history,
            handle_tool_errors=config.handle_tool_errors,
        )
        if result_type == "direct":
            return result_content
        return result_content

    async def _stream_fn(chat_request_or_message: ChatRequestOrMessage) -> AsyncGenerator[str, None]:
        """Execute the runtime tool-calling agent and stream its response."""

        chat_request = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequest)
        messages = _build_messages(chat_request, system_prompt, config.max_history)
        request_params = chat_request.model_dump(exclude={"messages"}, exclude_none=True, exclude_unset=True)
        request_params["tools"] = tool_schemas

        result_type, result_content = await _resolve_tool_calls(
            llm=llm,
            messages=messages,
            request_params=request_params,
            tools_by_name=tools_by_name,
            return_direct_names=return_direct_names,
            max_iterations=config.max_iterations,
            max_history=config.max_history,
            handle_tool_errors=config.handle_tool_errors,
        )

        if result_type == "direct":
            if result_content:
                yield result_content
            return

        stream_params = dict(request_params)
        stream_params["stream"] = True
        chat_messages = cast(list[ChatCompletionMessageParam], messages)
        stream = cast(AsyncGenerator[ChatCompletionChunk, None],
                      await llm.chat_completions(messages=chat_messages, **stream_params))
        async for chunk in stream:
            text = _extract_text_from_chunk(chunk)
            if text:
                yield text

    yield FunctionInfo.create(single_fn=_response_fn, stream_fn=_stream_fn, description=config.description)
