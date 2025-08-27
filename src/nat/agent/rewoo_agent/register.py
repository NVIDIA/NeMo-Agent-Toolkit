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

from pydantic import AliasChoices
from pydantic import Field
from pydantic import PositiveInt

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from nat.utils.type_converter import GlobalTypeConverter

logger = logging.getLogger(__name__)


class ReWOOAgentWorkflowConfig(FunctionBaseConfig, name="rewoo_agent"):
    """
    Defines a NAT function that uses a ReWOO Agent performs reasoning inbetween tool calls, and utilizes the
    tool names and descriptions to select the optimal tool.
    """

    tool_names: list[FunctionRef] = Field(default_factory=list,
                                          description="The list of tools to provide to the rewoo agent.")
    llm_name: LLMRef = Field(description="The LLM model to use with the rewoo agent.")
    verbose: bool = Field(default=False, description="Set the verbosity of the rewoo agent's logging.")
    include_tool_input_schema_in_tool_description: bool = Field(
        default=True, description="Specify inclusion of tool input schemas in the prompt.")
    description: str = Field(default="ReWOO Agent Workflow", description="The description of this functions use.")
    planner_prompt: str | None = Field(
        default=None,
        description="Provides the PLANNER_PROMPT to use with the agent")  # defaults to PLANNER_PROMPT in prompt.py
    solver_prompt: str | None = Field(
        default=None,
        description="Provides the SOLVER_PROMPT to use with the agent")  # defaults to SOLVER_PROMPT in prompt.py
    max_history: int = Field(default=15, description="Maximum number of messages to keep in the conversation history.")
    log_response_max_chars: PositiveInt = Field(
        default=1000, description="Maximum number of characters to display in logs when logging tool responses.")
    use_openai_api: bool = Field(default=False,
                                 description=("Use OpenAI API for the input/output types to the function. "
                                              "If False, strings will be used."))
    additional_planner_instructions: str | None = Field(
        default=None,
        validation_alias=AliasChoices("additional_planner_instructions", "additional_instructions"),
        description="Additional instructions to provide to the agent in addition to the base planner prompt.")
    additional_solver_instructions: str | None = Field(
        default=None,
        description="Additional instructions to provide to the agent in addition to the base solver prompt.")


@register_function(config_type=ReWOOAgentWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def rewoo_agent_workflow(config: ReWOOAgentWorkflowConfig, builder: Builder):
    from langchain.schema import BaseMessage
    from langchain_core.messages import trim_messages
    from langchain_core.messages.human import HumanMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph.graph import CompiledGraph

    from nat.agent.rewoo_agent.prompt import PLANNER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import PLANNER_USER_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_USER_PROMPT

    from .agent import ReWOOAgentGraph
    from .agent import ReWOOGraphState

    # the ReWOO Agent prompts are defined in prompt.py, and can be customized there or by modifying the config option
    # planner_prompt and solver_prompt.
    planner_system_prompt = PLANNER_SYSTEM_PROMPT if config.planner_prompt is None else config.planner_prompt
    if config.additional_planner_instructions:
        planner_system_prompt += f"{config.additional_planner_instructions}"
    if not ReWOOAgentGraph.validate_planner_prompt(planner_system_prompt):
        logger.exception("Invalid planner prompt")
        raise ValueError("Invalid planner prompt")
    planner_prompt = ChatPromptTemplate([("system", planner_system_prompt), ("user", PLANNER_USER_PROMPT)])

    solver_system_prompt = SOLVER_SYSTEM_PROMPT if config.solver_prompt is None else config.solver_prompt
    if config.additional_solver_instructions:
        solver_system_prompt += f"{config.additional_solver_instructions}"
    if not ReWOOAgentGraph.validate_solver_prompt(solver_system_prompt):
        logger.exception("Invalid solver prompt")
        raise ValueError("Invalid solver prompt")
    solver_prompt = ChatPromptTemplate([("system", solver_system_prompt), ("user", SOLVER_USER_PROMPT)])

    # we can choose an LLM for the ReWOO agent in the config file
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # the agent can run any installed tool, simply install the tool and add it to the config file
    # the sample tool provided can easily be copied or changed
    tools = builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    if not tools:
        raise ValueError(f"No tools specified for ReWOO Agent '{config.llm_name}'")

    # construct the ReWOO Agent Graph from the configured llm, prompt, and tools
    graph: CompiledGraph = await ReWOOAgentGraph(llm=llm,
                                                 planner_prompt=planner_prompt,
                                                 solver_prompt=solver_prompt,
                                                 tools=tools,
                                                 use_tool_schema=config.include_tool_input_schema_in_tool_description,
                                                 detailed_logs=config.verbose,
                                                 log_response_max_chars=config.log_response_max_chars).build_graph()

    async def _response_fn(input_message: ChatRequest) -> ChatResponse:
        try:
            # initialize the starting state with the user query
            messages: list[BaseMessage] = trim_messages(messages=[m.model_dump() for m in input_message.messages],
                                                        max_tokens=config.max_history,
                                                        strategy="last",
                                                        token_counter=len,
                                                        start_on="human",
                                                        include_system=True)

            task = HumanMessage(content=messages[-1].content)
            state = ReWOOGraphState(messages=messages, task=task)

            # run the ReWOO Agent Graph
            state = await graph.ainvoke(state)

            # get and return the output from the state
            state = ReWOOGraphState(**state)
            output_message = state.result.content
            return ChatResponse.from_string(output_message)

        except Exception as ex:
            logger.exception("ReWOO Agent failed with exception: %s", ex, exc_info=ex)
            # here, we can implement custom error messages
            if config.verbose:
                return ChatResponse.from_string(str(ex))
            return ChatResponse.from_string("I seem to be having a problem.")

    if (config.use_openai_api):
        yield FunctionInfo.from_fn(_response_fn, description=config.description)

    else:

        async def _str_api_fn(input_message: str) -> str:
            oai_input = GlobalTypeConverter.get().try_convert(input_message, to_type=ChatRequest)
            oai_output = await _response_fn(oai_input)

            return GlobalTypeConverter.get().try_convert(oai_output, to_type=str)

        yield FunctionInfo.from_fn(_str_api_fn, description=config.description)
