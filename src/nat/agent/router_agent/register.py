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
    A router agent takes in the incoming message, combines it with a prompt and the list of branches,
    and ask a LLM about which branch to take.
    """
    branches: list[FunctionRef] = Field(default_factory=list,
                                        description="The list of branches to provide to the router agent.")
    llm_name: LLMRef = Field(description="The LLM model to use with the routing agent.")
    description: str = Field(default="Router Agent Workflow", description="Description of this functions use.")
    system_prompt: str | None = Field(default=None, description="Provides the system prompt to use with the agent.")
    user_prompt: str | None = Field(default=None, description="Provides the prompt to use with the agent.")
    max_router_retries: int = Field(
        default=3, description="Maximum number of retries if the router agent fails to choose a branch.")
    detailed_logs: bool = Field(default=False, description="Set the verbosity of the router agent's logging.")
    log_response_max_chars: PositiveInt = Field(
        default=1000, description="Maximum number of characters to display in logs when logging branch responses.")


@register_function(config_type=RouterAgentWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def router_agent_workflow(config: RouterAgentWorkflowConfig, builder: Builder):
    from langchain_core.messages.human import HumanMessage
    from langgraph.graph.state import CompiledStateGraph

    from nat.agent.base import AGENT_LOG_PREFIX
    from nat.agent.router_agent.agent import RouterAgentGraph
    from nat.agent.router_agent.agent import RouterAgentGraphState
    from nat.agent.router_agent.agent import create_router_agent_prompt

    prompt = create_router_agent_prompt(config)
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    branches = builder.get_tools(tool_names=config.branches, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    if not branches:
        raise ValueError(f"No branches specified for Router Agent '{config.llm_name}'")

    graph: CompiledStateGraph = await RouterAgentGraph(
        llm=llm,
        branches=branches,
        prompt=prompt,
        max_router_retries=config.max_router_retries,
        detailed_logs=config.detailed_logs,
        log_response_max_chars=config.log_response_max_chars,
    ).build_graph()

    async def _response_fn(input_message: str) -> str:
        try:
            message = HumanMessage(content=input_message)
            state = RouterAgentGraphState(forward_message=message)

            result_dict = await graph.ainvoke(state)
            result_state = RouterAgentGraphState(**result_dict)

            output_message = result_state.messages[-1]
            return str(output_message.content)

        except Exception as ex:
            logger.exception("%s Router Agent failed with exception: %s", AGENT_LOG_PREFIX, ex)
            if config.detailed_logs:
                return str(ex)
            return "Router agent failed with exception: %s" % ex

    try:
        yield FunctionInfo.from_fn(_response_fn, description=config.description)
    except GeneratorExit:
        logger.exception("%s Workflow exited early!", AGENT_LOG_PREFIX)
    finally:
        logger.debug("%s Cleaning up router_agent workflow.", AGENT_LOG_PREFIX)
