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


class RouterAgentWorkflowConfig(FunctionBaseConfig, name="router_agent"):
    """
    A Router Agent requires an LLM which supports routing. A router agent takes in the incoming message,
    combine it with a prompt and the list of branches, and ask a LLM about which branch to take.
    """
    branches: list[FunctionRef] = Field(default_factory=list,
                                        description="The list of branches to provide to the router agent.")
    llm_name: LLMRef = Field(description="The LLM model to use with the routing agent.")
    system_prompt: str | None = Field(default=None, description="Provides the system prompt to use with the agent.")
    user_prompt: str | None = Field(default=None, description="Provides the prompt to use with the agent.")
    detailed_logs: bool = Field(default=False, description="Set the verbosity of the router agent's logging.")
    log_response_max_chars: PositiveInt = Field(
        default=1000, description="Maximum number of characters to display in logs when logging branch responses.")
    # tool_names: list[FunctionRef] = Field(default_factory=list,
    #                                       description="The list of branches to provide to the routing agent.")
    # llm_name: LLMRef = Field(description="The LLM model to use with the routing agent.")
    # verbose: bool = Field(default=False, description="Set the verbosity of the tool calling agent's logging.")
    # handle_branch_errors: bool = Field(default=True, description="Specify ability to handle branch calling errors.")
    # description: str = Field(default="Routing Agent Workflow", description="Description of this functions use.")
    # max_iterations: int = Field(default=15, description="Number of branch calls before stoping the routing agent.")
    # log_response_max_chars: PositiveInt = Field(
    #     default=1000, description="Maximum number of characters to display in logs when logging branch responses.")
    # system_prompt: str | None = Field(default=None, description="Provides the system prompt to use with the agent.")
    # prompt: str | None = Field(default=None, description="Provides the prompt to use with the agent.")
    # additional_instructions: str | None = Field(default=None,
    #                                             description="Additional instructions appended to the system prompt.")


@register_function(config_type=RouterAgentWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def router_agent_workflow(config: RouterAgentWorkflowConfig, builder: Builder):
    from langchain_core.messages.human import HumanMessage
    from langgraph.graph.graph import CompiledGraph

    from nat.agent.base import AGENT_LOG_PREFIX
    from nat.agent.router_agent.agent import RouterAgentGraph
    from nat.agent.router_agent.agent import RouterGraphState
    from nat.agent.router_agent.agent import create_router_agent_prompt

    # prompt = create_routing_agent_prompt(config)
    # we can choose an LLM for the Routing agent in the config file
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    # the agent can run any installed tool, simply install the tool and add it to the config file
    # the sample tools provided can easily be copied or changed
    branches = builder.get_tools(tool_names=config.branches, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    if not branches:
        raise ValueError(f"No tools specified for Routing Agent '{config.llm_name}'")

    prompt = create_router_agent_prompt(config)
    # construct the Tool Calling Agent Graph from the configured llm, and tools
    graph: CompiledGraph = await RouterAgentGraph(
        llm=llm,
        branches=branches,
        prompt=prompt,
        detailed_logs=config.detailed_logs,
        log_response_max_chars=config.log_response_max_chars,
    ).build_graph()

    async def _response_fn(input_message: str) -> str:
        try:
            # initialize the starting state with the user query
            input_message = HumanMessage(content=input_message)
            state = RouterGraphState(messages=[input_message])

            # run the Tool Calling Agent Graph
            state = await graph.ainvoke(state)

            state = RouterGraphState(**state)
            output_message = state.messages[-1]
            return output_message.content
        except Exception as ex:
            logger.exception("%s Router Agent failed with exception: %s", AGENT_LOG_PREFIX, ex)
            if config.verbose:
                return str(ex)
            return "I seem to be having a problem."

    try:
        yield FunctionInfo.from_fn(_response_fn, description=config.description)
    except GeneratorExit:
        logger.exception("%s Workflow exited early!", AGENT_LOG_PREFIX)
    finally:
        logger.debug("%s Cleaning up router_agent workflow.", AGENT_LOG_PREFIX)
