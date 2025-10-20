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

import logging
import typing

from pydantic import Field

from nat.agent.base import AGENT_LOG_PREFIX
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.openai_mcp import OpenAIMCPSchemaTool

logger = logging.getLogger(__name__)


class ResponsesAPIAgentWorkflowConfig(FunctionBaseConfig, name="responses_api_agent"):
    """
    Defines an NeMo Agent Toolkit function that uses a Responses API
    Agent performs reasoning inbetween tool calls, and utilizes the
    tool names and descriptions to select the optimal tool.
    """

    llm_name: LLMRef = Field(description="The LLM model to use with the agent.")
    verbose: bool = Field(default=False, description="Set the verbosity of the agent's logging.")
    nat_tools: list[FunctionRef] = Field(default_factory=list, description="The list of tools to provide to the agent.")
    mcp_tools: list[OpenAIMCPSchemaTool] = Field(
        default_factory=list,
        description="List of MCP tools to use with the agent. If empty, no MCP tools will be used.")
    builtin_tools: list[dict[str, typing.Any]] = Field(
        default_factory=list,
        description="List of built-in tools to use with the agent. If empty, no built-in tools will be used.")

    max_iterations: int = Field(default=15, description="Number of tool calls before stoping the agent.")
    description: str = Field(default="Agent Workflow", description="The description of this functions use.")
    parallel_tool_calls: bool = Field(default=False,
                                      description="Specify whether to allow parallel tool calls in the agent.")
    handle_tool_errors: bool = Field(
        default=True,
        description="Specify ability to handle tool calling errors. If False, tool errors will raise an exception.")


@register_function(config_type=ResponsesAPIAgentWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def responses_api_agent_workflow(config: ResponsesAPIAgentWorkflowConfig, builder: Builder):
    from langchain_core.messages.human import HumanMessage
    from langchain_core.runnables import Runnable
    from langchain_openai import ChatOpenAI

    from nat.agent.tool_calling_agent.agent import ToolCallAgentGraph
    from nat.agent.tool_calling_agent.agent import ToolCallAgentGraphState

    llm: ChatOpenAI = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    assert llm.use_responses_api, "Responses API Agent requires an LLM that supports the Responses API."

    # Get tools
    tools = []
    nat_tools = await builder.get_tools(tool_names=config.nat_tools, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tools.extend(nat_tools)
    # MCP tools are optional, if provided they will be used by the agent
    tools.extend([m.model_dump() for m in config.mcp_tools])
    # Built-in tools are optional, if provided they will be used by the agent
    tools.extend(config.builtin_tools)

    # Bind tools to LLM
    if tools:
        llm: Runnable = llm.bind_tools(tools=tools, parallel_tool_calls=config.parallel_tool_calls, strict=True)

    if config.verbose:
        logger.info("%s Using LLM: %s with tools: %s", AGENT_LOG_PREFIX, llm.model_name, tools)

    agent = ToolCallAgentGraph(
        llm=llm,
        tools=nat_tools,  # MCP and built-in tools are already bound to the LLM and need not be handled by graph
        detailed_logs=config.verbose,
        handle_tool_errors=config.handle_tool_errors)

    graph = await agent.build_graph()

    async def _response_fn(input_message: str) -> str:
        try:
            # initialize the starting state with the user query
            input_message = HumanMessage(content=input_message)
            state = ToolCallAgentGraphState(messages=[input_message])

            # run the Tool Calling Agent Graph
            state = await graph.ainvoke(state, config={'recursion_limit': (config.max_iterations + 1) * 2})
            # setting recursion_limit: 4 allows 1 tool call
            #   - allows the Tool Calling Agent to perform 1 cycle / call 1 single tool,
            #   - but stops the agent when it tries to call a tool a second time

            # get and return the output from the state
            state = ToolCallAgentGraphState(**state)
            output_message = state.messages[-1]  # pylint: disable=E1136
            content = output_message.content[-1]['text'] if output_message.content and isinstance(
                output_message.content[-1], dict) and 'text' in output_message.content[-1] else str(
                    output_message.content)
            return content
        except Exception as ex:
            logger.exception("%s Tool Calling Agent failed with exception: %s", AGENT_LOG_PREFIX, ex, exc_info=ex)
            if config.verbose:
                return str(ex)
            return "I seem to be having a problem."

    try:
        yield FunctionInfo.from_fn(_response_fn, description=config.description)
    except GeneratorExit:
        logger.exception("%s Workflow exited early!", AGENT_LOG_PREFIX, exc_info=True)
    finally:
        logger.debug("%s Cleaning up react_agent workflow.", AGENT_LOG_PREFIX)
