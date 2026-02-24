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
from nat.sdk.agent.agent import Agent
from nat.sdk.agent.state import AgentStatus
from nat.sdk.conversation.runner import ConversationRunner
from nat.sdk.conversation.state import ConversationState
from nat.sdk.event.event import MessageEvent
from nat.sdk.event.event import ObservationEvent
from nat.sdk.llm.builder_client import BuilderLLMClient
from nat.sdk.llm.message import Message
from nat.sdk.llm.message import ToolCall
from nat.sdk.tool.tool import Tool
from nat.utils.type_converter import GlobalTypeConverter

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


def _build_system_prompt(config: RuntimeToolCallingAgentConfig) -> str | None:
    """Create a tool-calling agent system prompt from config."""
    prompt_parts = []
    for prompt in (config.system_prompt, config.additional_instructions):
        if prompt:
            prompt_parts.append(prompt)
    if prompt_parts:
        return " ".join(prompt_parts)
    return None


def _serialize_output(output: object) -> str:
    """Serialize tool output to a string for the LLM."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, default=str)
    except TypeError:
        return str(output)


def _convert_chat_request(chat_request: ChatRequest, system_prompt: str | None) -> list[Message]:
    """Convert a ChatRequest into SDK Message objects."""
    messages: list[Message] = []
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))
    for msg in chat_request.messages:
        d = msg.model_dump(exclude_none=True)
        tool_calls = []
        for tc in d.get("tool_calls", []):
            arguments = tc["function"]["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            tool_calls.append(ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=arguments,
            ))
        messages.append(
            Message(
                role=d["role"],
                content=d.get("content", ""),
                tool_calls=tool_calls if tool_calls else [],
                tool_call_id=d.get("tool_call_id"),
            ))
    return messages


@register_function(config_type=RuntimeToolCallingAgentConfig, framework_wrappers=[LLMFrameworkEnum.RUNTIME])
async def runtime_tool_calling_agent_workflow(config: RuntimeToolCallingAgentConfig, builder: Builder):
    """Build a runtime tool-calling agent workflow."""

    from nat.plugins.runtime.llm import RuntimeOpenAIClient

    runtime_llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.RUNTIME)
    if not isinstance(runtime_llm, RuntimeOpenAIClient):
        raise TypeError("Runtime tool-calling agent requires a RuntimeOpenAIClient LLM wrapper.")

    client = BuilderLLMClient(runtime_llm)

    tools = await builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.RUNTIME)
    if not tools:
        raise ValueError(f"No tools specified for runtime tool-calling agent '{config.llm_name}'")
    tools_by_name: dict[str, Tool] = {tool.name: tool for tool in tools}

    return_direct_names: set[str] = set()
    if config.return_direct:
        rd_tools = await builder.get_tools(tool_names=config.return_direct, wrapper_type=LLMFrameworkEnum.RUNTIME)
        return_direct_names = {t.name for t in rd_tools} if rd_tools else set()

    system_prompt = _build_system_prompt(config)

    async def _response_fn(chat_request_or_message: ChatRequestOrMessage) -> str:
        """Execute the runtime tool-calling agent and return its response."""
        chat_request = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequest)
        initial_messages = _convert_chat_request(chat_request, system_prompt)

        agent = Agent(tools=list(tools_by_name.values()), max_iterations=config.max_iterations)
        state = ConversationState()
        state.agent_state.status = AgentStatus.RUNNING

        runner = ConversationRunner(
            agent=agent,
            client=client,
            state=state,
            tools=tools_by_name,
            return_direct_tools=return_direct_names,
            handle_tool_errors=config.handle_tool_errors,
            max_history=config.max_history,
            initial_messages=initial_messages,
        )

        last_event = await runner.run_until_done()

        if isinstance(last_event, ObservationEvent):
            return _serialize_output(last_event.output)
        if isinstance(last_event, MessageEvent):
            return last_event.content
        return ""

    yield FunctionInfo.from_fn(_response_fn, description=config.description)
