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

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class SecondSearchAgentFunctionConfig(FunctionBaseConfig, name="second_search_agent"):
    """
    NeMo Agent toolkit function template. Please update the description.
    """
    tool_names: list[FunctionRef] = Field(default=[], description="List of tool names to use")
    llm_name: LLMRef = Field(description="LLM name to use")
    max_history: int = Field(default=10, description="Maximum number of historical messages to provide to the agent")
    max_iterations: int = Field(default=15, description="Maximum number of iterations to run the agent")
    handle_parsing_errors: bool = Field(default=True, description="Whether to handle parsing errors")
    verbose: bool = Field(default=True, description="Whether to print verbose output")
    description: str = Field(default="", description="Description of the agent")


@register_function(config_type=SecondSearchAgentFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def second_search_agent_function(config: SecondSearchAgentFunctionConfig, builder: Builder):
    from langchain import hub
    from langchain.agents import AgentExecutor
    from langchain.agents import create_react_agent

    # Create a list of tools for the agent
    tools = builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Use an open source prompt
    prompt = hub.pull("hwchase17/react-chat")

    # Initialize a ReAct agent
    react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt, stop_sequence=["\nObservation"])

    # Initialize an agent executor to iterate through reasoning steps
    agent_executor = AgentExecutor(agent=react_agent,
                                   tools=tools,
                                   max_iterations=config.max_iterations,
                                   handle_parsing_errors=config.handle_parsing_errors,
                                   verbose=config.verbose)

    async def _response_fn(input_message: str) -> str:
        response = await agent_executor.ainvoke({"input": input_message, "chat_history": []})

        return response["output"]

    try:
        yield FunctionInfo.create(single_fn=_response_fn)
    except GeneratorExit:
        print("Function exited early!")
    finally:
        print("Cleaning up second_search_agent workflow.")
