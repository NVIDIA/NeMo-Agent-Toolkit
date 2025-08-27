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

from pydantic import Field
from pydantic import PositiveInt

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class ToolCallAgentWorkflowConfig(FunctionBaseConfig, name="tool_calling_agent"):
    """
    A Tool Calling Agent requires an LLM which supports tool calling. A tool Calling Agent utilizes the tool
    input parameters to select the optimal tool.  Supports handling tool errors.
    """

    tool_names: list[FunctionRef] = Field(default_factory=list,
                                          description="The list of tools to provide to the tool calling agent.")
    llm_name: LLMRef = Field(description="The LLM model to use with the tool calling agent.")
    verbose: bool = Field(default=False, description="Set the verbosity of the tool calling agent's logging.")
    handle_tool_errors: bool = Field(default=True, description="Specify ability to handle tool calling errors.")
    description: str = Field(default="Tool Calling Agent Workflow", description="Description of this functions use.")
    max_iterations: int = Field(default=15, description="Number of tool calls before stoping the tool calling agent.")
    log_response_max_chars: PositiveInt = Field(
        default=1000, description="Maximum number of characters to display in logs when logging tool responses.")
    system_prompt: str | None = Field(default=None, description="Provides the system prompt to use with the agent.")
    additional_instructions: str | None = Field(default=None,
                                                description="Additional instructions appended to the system prompt.")


@register_function(config_type=ToolCallAgentWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def tool_calling_agent_workflow(config: ToolCallAgentWorkflowConfig, builder: Builder):
    from langchain_core.messages.human import HumanMessage
    from langgraph.graph.graph import CompiledGraph

    from nat.agent.base import AGENT_LOG_PREFIX
    from nat.agent.tool_calling_agent.agent import ToolCallAgentGraph
    from nat.agent.tool_calling_agent.agent import ToolCallAgentGraphState
    from nat.agent.tool_calling_agent.agent import create_tool_calling_agent_prompt

    prompt = create_tool_calling_agent_prompt(config)
    # we can choose an LLM for the ReAct agent in the config file
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    # the agent can run any installed tool, simply install the tool and add it to the config file
    # the sample tools provided can easily be copied or changed
    tools = builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    if not tools:
        raise ValueError(f"No tools specified for Tool Calling Agent '{config.llm_name}'")

    # construct the Tool Calling Agent Graph from the configured llm, and tools
    graph: CompiledGraph = await ToolCallAgentGraph(llm=llm,
                                                    tools=tools,
                                                    prompt=prompt,
                                                    detailed_logs=config.verbose,
                                                    log_response_max_chars=config.log_response_max_chars,
                                                    handle_tool_errors=config.handle_tool_errors).build_graph()

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
            output_message = state.messages[-1]
            return output_message.content
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
