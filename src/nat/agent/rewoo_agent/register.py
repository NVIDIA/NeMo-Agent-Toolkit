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
from nat.data_models.agent import AgentBaseConfig
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import Usage
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.utils.type_converter import GlobalTypeConverter

logger = logging.getLogger(__name__)


class ReWOOAgentWorkflowConfig(AgentBaseConfig, name="rewoo_agent"):
    """
    Defines a NAT function that uses a ReWOO Agent performs reasoning inbetween tool calls, and utilizes the
    tool names and descriptions to select the optimal tool.
    """
    description: str = Field(default="ReWOO Agent Workflow", description="The description of this functions use.")
    tool_names: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list, description="The list of tools to provide to the rewoo agent.")
    include_tool_input_schema_in_tool_description: bool = Field(
        default=True, description="Specify inclusion of tool input schemas in the prompt.")
    planner_prompt: str | None = Field(
        default=None,
        description="Provides the PLANNER_PROMPT to use with the agent")  # defaults to PLANNER_PROMPT in prompt.py
    solver_prompt: str | None = Field(
        default=None,
        description="Provides the SOLVER_PROMPT to use with the agent")  # defaults to SOLVER_PROMPT in prompt.py
    tool_call_max_retries: PositiveInt = Field(default=3,
                                               description="The number of retries before raising a tool call error.",
                                               ge=1)
    max_history: int = Field(default=15, description="Maximum number of messages to keep in the conversation history.")
    additional_planner_instructions: str | None = Field(
        default=None,
        validation_alias=AliasChoices("additional_planner_instructions", "additional_instructions"),
        description="Additional instructions to provide to the agent in addition to the base planner prompt.")
    additional_solver_instructions: str | None = Field(
        default=None,
        description="Additional instructions to provide to the agent in addition to the base solver prompt.")
    raise_tool_call_error: bool = Field(default=True,
                                        description="Whether to raise a exception immediately if a tool"
                                        "call fails. If set to False, the tool call error message will be included in"
                                        "the tool response and passed to the next tool.")


@register_function(config_type=ReWOOAgentWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def rewoo_agent_workflow(config: ReWOOAgentWorkflowConfig, builder: Builder):
    from langchain_core.messages import trim_messages
    from langchain_core.messages.base import BaseMessage
    from langchain_core.messages.human import HumanMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph.state import CompiledStateGraph

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
        logger.error("Invalid planner prompt")
        raise ValueError("Invalid planner prompt")
    planner_prompt = ChatPromptTemplate([("system", planner_system_prompt), ("user", PLANNER_USER_PROMPT)])

    solver_system_prompt = SOLVER_SYSTEM_PROMPT if config.solver_prompt is None else config.solver_prompt
    if config.additional_solver_instructions:
        solver_system_prompt += f"{config.additional_solver_instructions}"
    if not ReWOOAgentGraph.validate_solver_prompt(solver_system_prompt):
        logger.error("Invalid solver prompt")
        raise ValueError("Invalid solver prompt")
    solver_prompt = ChatPromptTemplate([("system", solver_system_prompt), ("user", SOLVER_USER_PROMPT)])

    # we can choose an LLM for the ReWOO agent in the config file
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # the agent can run any installed tool, simply install the tool and add it to the config file
    # the sample tool provided can easily be copied or changed
    tools = await builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    if not tools:
        raise ValueError(f"No tools specified for ReWOO Agent '{config.llm_name}'")

    # construct the ReWOO Agent Graph from the configured llm, prompt, and tools
    graph: CompiledStateGraph = await ReWOOAgentGraph(
        llm=llm,
        planner_prompt=planner_prompt,
        solver_prompt=solver_prompt,
        tools=tools,
        use_tool_schema=config.include_tool_input_schema_in_tool_description,
        detailed_logs=config.verbose,
        log_response_max_chars=config.log_response_max_chars,
        tool_call_max_retries=config.tool_call_max_retries,
        raise_tool_call_error=config.raise_tool_call_error).build_graph()

    async def _response_fn(chat_request_or_message: ChatRequestOrMessage) -> ChatResponse | str:
        """
        Main workflow entry function for the ReWOO Agent.

        This function invokes the ReWOO Agent Graph and returns the response.

        Args:
            chat_request_or_message (ChatRequestOrMessage): The input message to process

        Returns:
            ChatResponse | str: The response from the agent or error message
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

            task = HumanMessage(content=messages[-1].content)
            state = ReWOOGraphState(messages=messages, task=task)

            # run the ReWOO Agent Graph
            state = await graph.ainvoke(state)

            # get and return the output from the state
            state = ReWOOGraphState(**state)
            output_message = state.result.content
            # Ensure output_message is a string
            if isinstance(output_message, list | dict):
                output_message = str(output_message)

            # Create usage statistics for the response
            prompt_tokens = sum(len(str(msg.content).split()) for msg in message.messages)
            completion_tokens = len(output_message.split()) if output_message else 0
            total_tokens = prompt_tokens + completion_tokens
            usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
            response = ChatResponse.from_string(output_message, usage=usage)
            if chat_request_or_message.is_string:
                return GlobalTypeConverter.get().convert(response, to_type=str)
            return response
        except Exception as ex:
            logger.error("ReWOO Agent failed with exception: %s", ex)
            raise

    yield FunctionInfo.from_fn(_response_fn, description=config.description)
