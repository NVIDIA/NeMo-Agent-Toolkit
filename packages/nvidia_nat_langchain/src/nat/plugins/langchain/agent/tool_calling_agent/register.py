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

import datetime
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from collections.abc import Callable

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.agent import AgentBaseConfig
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import ChatResponseChunkChoice
from nat.data_models.api_server import ChoiceDelta
from nat.data_models.api_server import ChoiceDeltaToolCall
from nat.data_models.api_server import ChoiceDeltaToolCallFunction
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.utils.type_converter import GlobalTypeConverter

logger = logging.getLogger(__name__)


class TruncationRetryConfig(BaseModel):
    """Configuration for retrying LLM calls that are truncated (finish_reason='length')."""

    max_retries: int = Field(default=0,
                             description="Number of retries when LLM output is truncated. "
                             "0 disables recovery (raises RuntimeError).")
    token_increment: int | None = Field(default=None,
                                        description="Fixed number of tokens added to max_tokens on each retry. "
                                        "Mutually exclusive with token_scaling. Defaults to 1024 if neither is set.")
    token_scaling: float | None = Field(default=None,
                                        description="Multiplicative factor applied to max_tokens on each retry "
                                        "(e.g. 1.5 = 50%% increase per retry). "
                                        "Mutually exclusive with token_increment.")

    @model_validator(mode="after")
    def _check_scaling_strategy(self) -> "TruncationRetryConfig":
        if self.token_increment is not None and self.token_scaling is not None:
            raise ValueError("Set token_increment or token_scaling, not both.")
        if self.max_retries > 0 and self.token_increment is None and self.token_scaling is None:
            self.token_increment = 1024
        return self

    def build_scaling_fn(self) -> Callable[[int], int]:
        """Build a callable that computes the next max_tokens from the current value."""
        if self.token_scaling is not None:
            factor: float = self.token_scaling
            return lambda current: int(current * factor)
        increment: int = self.token_increment or 1024
        return lambda current: current + increment


class ToolCallAgentWorkflowConfig(AgentBaseConfig, name="tool_calling_agent"):
    """
    A Tool Calling Agent requires an LLM which supports tool calling. A tool Calling Agent utilizes the tool
    input parameters to select the optimal tool.  Supports handling tool errors.
    """
    description: str = Field(default="Tool Calling Agent Workflow", description="Description of this functions use.")
    tool_names: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list, description="The list of tools to provide to the tool calling agent.")
    handle_tool_errors: bool = Field(default=True, description="Specify ability to handle tool calling errors.")
    max_iterations: int = Field(default=15, description="Number of tool calls before stoping the tool calling agent.")
    max_history: int = Field(default=15, description="Maximum number of messages to keep in the conversation history.")

    truncation_retry: TruncationRetryConfig = Field(default_factory=TruncationRetryConfig,
                                                    description="Configuration for retrying truncated LLM responses.")
    max_empty_response_retries: int = Field(
        default=0,
        description="Number of retries when LLM returns an empty response (no content, no tool calls). "
        "0 disables recovery (raises RuntimeError).")
    system_prompt: str | None = Field(default=None, description="Provides the system prompt to use with the agent.")
    additional_instructions: str | None = Field(default=None,
                                                description="Additional instructions appended to the system prompt.")
    return_direct: list[FunctionRef] | None = Field(
        default=None, description="List of tool names that should return responses directly without LLM processing.")


@register_function(config_type=ToolCallAgentWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def tool_calling_agent_workflow(config: ToolCallAgentWorkflowConfig, builder: Builder):
    from langchain_core.messages import AIMessage
    from langchain_core.messages import AIMessageChunk
    from langchain_core.messages import trim_messages
    from langchain_core.messages.base import BaseMessage
    from langgraph.errors import GraphRecursionError
    from langgraph.graph.state import CompiledStateGraph

    from nat.plugins.langchain.agent.base import AGENT_LOG_PREFIX
    from nat.plugins.langchain.agent.tool_calling_agent.agent import ToolCallAgentGraph
    from nat.plugins.langchain.agent.tool_calling_agent.agent import ToolCallAgentGraphState
    from nat.plugins.langchain.agent.tool_calling_agent.agent import create_tool_calling_agent_prompt

    prompt = create_tool_calling_agent_prompt(config)
    # we can choose an LLM for the ReAct agent in the config file
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    # the agent can run any installed tool, simply install the tool and add it to the config file
    # the sample tools provided can easily be copied or changed
    tools = await builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    if not tools:
        raise ValueError(f"No tools specified for Tool Calling Agent '{config.llm_name}'")

    # convert return_direct FunctionRef objects to BaseTool objects
    return_direct_tools = await builder.get_tools(
        tool_names=config.return_direct, wrapper_type=LLMFrameworkEnum.LANGCHAIN) if config.return_direct else None

    # construct the Tool Calling Agent Graph from the configured llm, and tools
    graph: CompiledStateGraph = await ToolCallAgentGraph(
        llm=llm,
        tools=tools,
        prompt=prompt,
        detailed_logs=config.verbose,
        log_response_max_chars=config.log_response_max_chars,
        handle_tool_errors=config.handle_tool_errors,
        return_direct=return_direct_tools,
        max_truncation_retries=config.truncation_retry.max_retries,
        truncation_scaling_fn=config.truncation_retry.build_scaling_fn(),
        max_empty_response_retries=config.max_empty_response_retries,
    ).build_graph()

    async def _response_fn(chat_request_or_message: ChatRequestOrMessage) -> str:
        """
        Main workflow entry function for the Tool Calling Agent.

        This function invokes the Tool Calling Agent Graph and returns the response.

        Args:
            chat_request_or_message (ChatRequestOrMessage): The input message to process

        Returns:
            str: The response from the agent or error message
        """
        try:
            message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequest)

            # initialize the starting state with the user query
            messages: list[BaseMessage] = trim_messages(messages=[m.model_dump() for m in message.messages],
                                                        max_tokens=config.max_history,
                                                        strategy="last",
                                                        token_counter=len,
                                                        start_on="human",
                                                        include_system=True)
            state = ToolCallAgentGraphState(messages=messages)

            # run the Tool Calling Agent Graph
            state = await graph.ainvoke(state, config={'recursion_limit': (config.max_iterations + 1) * 2})
            # setting recursion_limit: 4 allows 1 tool call
            #   - allows the Tool Calling Agent to perform 1 cycle / call 1 single tool,
            #   - but stops the agent when it tries to call a tool a second time

            # get and return the output from the state
            state = ToolCallAgentGraphState(**state)
            output_message = state.messages[-1]
            return str(output_message.content)

        except GraphRecursionError:
            logger.warning(
                "%s Tool Calling Agent reached its maximum iteration limit (%d) without producing a final answer. "
                "This typically means the LLM kept calling tools instead of returning a response.",
                AGENT_LOG_PREFIX,
                config.max_iterations)

            return (f"The tool calling agent could not produce a final answer within {config.max_iterations} "
                    "iterations. The agent repeatedly called tools without converging on a response.")

        except Exception as ex:
            logger.error("%s Tool Calling Agent failed with exception: %s", AGENT_LOG_PREFIX, ex)
            raise

    async def _stream_fn(chat_request_or_message: ChatRequestOrMessage) -> AsyncGenerator[ChatResponseChunk]:
        """
        Streaming workflow entry function for the Tool Calling Agent.

        Uses graph.astream with stream_mode="messages" to yield token-level chunks from the LLM,
        enabling real-time SSE streaming over the OpenAI-compatible /v1/chat/completions endpoint.
        Yields both content tokens and tool call chunks as ChatResponseChunk objects.

        Args:
            chat_request_or_message (ChatRequestOrMessage): The input message to process

        Yields:
            ChatResponseChunk: Streaming chunks containing content deltas or tool call deltas
        """
        chunk_id = str(uuid.uuid4())
        try:
            message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequest)

            messages: list[BaseMessage] = trim_messages(messages=[m.model_dump() for m in message.messages],
                                                        max_tokens=config.max_history,
                                                        strategy="last",
                                                        token_counter=len,
                                                        start_on="human",
                                                        include_system=True)
            state = ToolCallAgentGraphState(messages=messages)

            async for msg, metadata in graph.astream(
                    state,
                    config={'recursion_limit': (config.max_iterations + 1) * 2},
                    stream_mode="messages"):
                if not isinstance(msg, (AIMessage, AIMessageChunk)):
                    continue
                if metadata.get("langgraph_node") != "agent":
                    continue

                if isinstance(msg.content, str) and msg.content:
                    yield ChatResponseChunk.create_streaming_chunk(msg.content, id_=chunk_id)

                tool_calls = getattr(msg, "tool_call_chunks", None) or getattr(msg, "tool_calls", None)
                if tool_calls:
                    delta_tool_calls = []
                    for i, tc in enumerate(tool_calls):
                        idx = tc.get("index") if isinstance(tc.get("index"), int) else i
                        args = tc.get("args", "")
                        if isinstance(args, dict):
                            args = json.dumps(args)
                        delta_tool_calls.append(
                            ChoiceDeltaToolCall(index=idx,
                                                id=tc.get("id"),
                                                type="function" if tc.get("id") else None,
                                                function=ChoiceDeltaToolCallFunction(
                                                    name=tc.get("name"),
                                                    arguments=args,
                                                )))
                    yield ChatResponseChunk(
                        id=chunk_id,
                        choices=[
                            ChatResponseChunkChoice(
                                index=0,
                                delta=ChoiceDelta(tool_calls=delta_tool_calls),
                                finish_reason=None,
                            )
                        ],
                        created=datetime.datetime.now(datetime.UTC),
                        model="unknown-model",
                        object="chat.completion.chunk",
                    )
        except GraphRecursionError:
            logger.warning(
                "%s Tool Calling Agent reached its maximum iteration limit (%d) without producing a final answer. "
                "This typically means the LLM kept calling tools instead of returning a response.",
                AGENT_LOG_PREFIX,
                config.max_iterations)
            yield ChatResponseChunk.create_streaming_chunk(
                f"The tool calling agent could not produce a final answer within {config.max_iterations} "
                "iterations. The agent repeatedly called tools without converging on a response.",
                id_=chunk_id,
            )
        except Exception as ex:
            logger.error("%s Tool Calling Agent streaming failed with exception: %s", AGENT_LOG_PREFIX, ex)
            raise

    yield FunctionInfo.create(single_fn=_response_fn, stream_fn=_stream_fn, description=config.description)
