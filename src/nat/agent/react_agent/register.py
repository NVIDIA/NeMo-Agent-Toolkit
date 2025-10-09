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
from nat.data_models.optimizable import OptimizableField
from nat.data_models.optimizable import OptimizableMixin
from nat.data_models.optimizable import SearchSpace
from nat.utils.type_converter import GlobalTypeConverter

logger = logging.getLogger(__name__)


class ReActAgentWorkflowConfig(AgentBaseConfig, OptimizableMixin, name="react_agent"):
    """
    Defines a NAT function that uses a ReAct Agent performs reasoning inbetween tool calls, and utilizes the
    tool names and descriptions to select the optimal tool.
    """
    description: str = Field(default="ReAct Agent Workflow", description="The description of this functions use.")
    tool_names: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list, description="The list of tools to provide to the react agent.")
    retry_agent_response_parsing_errors: bool = Field(
        default=True,
        validation_alias=AliasChoices("retry_agent_response_parsing_errors", "retry_parsing_errors"),
        description="Whether to retry when encountering parsing errors in the agent's response.")
    parse_agent_response_max_retries: int = Field(
        default=1,
        validation_alias=AliasChoices("parse_agent_response_max_retries", "max_retries"),
        description="Maximum number of times the Agent may retry parsing errors. "
        "Prevents the Agent from getting into infinite hallucination loops.")
    tool_call_max_retries: int = Field(default=1, description="The number of retries before raising a tool call error.")
    max_tool_calls: int = Field(default=15,
                                validation_alias=AliasChoices("max_tool_calls", "max_iterations"),
                                description="Maximum number of tool calls before stopping the agent.")
    pass_tool_call_errors_to_agent: bool = Field(
        default=True,
        description="Whether to pass tool call errors to agent. If False, failed tool calls will raise an exception.")
    include_tool_input_schema_in_tool_description: bool = Field(
        default=True, description="Specify inclusion of tool input schemas in the prompt.")
    normalize_tool_input_quotes: bool = Field(
        default=True,
        description="Whether to replace single quotes with double quotes in the tool input. "
        "This is useful for tools that expect structured json input.")
    system_prompt: str | None = Field(
        default=None,
        description="Provides the SYSTEM_PROMPT to use with the agent")  # defaults to SYSTEM_PROMPT in prompt.py
    max_history: int = Field(default=15, description="Maximum number of messages to keep in the conversation history.")
    additional_instructions: str | None = OptimizableField(
        default=None,
        description="Additional instructions to provide to the agent in addition to the base prompt.",
        space=SearchSpace(
            is_prompt=True,
            prompt="No additional instructions.",
            prompt_purpose="Additional instructions to provide to the agent in addition to the base prompt.",
        ))


@register_function(config_type=ReActAgentWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def react_agent_workflow(config: ReActAgentWorkflowConfig, builder: Builder):
    from langchain.schema import BaseMessage
    from langchain_core.messages import trim_messages
    from langgraph.graph.state import CompiledStateGraph

    from nat.agent.base import AGENT_LOG_PREFIX
    from nat.agent.react_agent.agent import ReActAgentGraph
    from nat.agent.react_agent.agent import ReActGraphState
    from nat.agent.react_agent.agent import create_react_agent_prompt

    prompt = create_react_agent_prompt(config)

    # we can choose an LLM for the ReAct agent in the config file
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    # the agent can run any installed tool, simply install the tool and add it to the config file
    # the sample tool provided can easily be copied or changed
    tools = await builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    if not tools:
        raise ValueError(f"No tools specified for ReAct Agent '{config.llm_name}'")
    # configure callbacks, for sending intermediate steps
    # construct the ReAct Agent Graph from the configured llm, prompt, and tools
    graph: CompiledStateGraph = await ReActAgentGraph(
        llm=llm,
        prompt=prompt,
        tools=tools,
        use_tool_schema=config.include_tool_input_schema_in_tool_description,
        detailed_logs=config.verbose,
        log_response_max_chars=config.log_response_max_chars,
        retry_agent_response_parsing_errors=config.retry_agent_response_parsing_errors,
        parse_agent_response_max_retries=config.parse_agent_response_max_retries,
        tool_call_max_retries=config.tool_call_max_retries,
        pass_tool_call_errors_to_agent=config.pass_tool_call_errors_to_agent,
        normalize_tool_input_quotes=config.normalize_tool_input_quotes).build_graph()

    async def _response_fn(chat_request_or_message: ChatRequestOrMessage) -> ChatResponse | str:
        """
        Main workflow entry function for the ReAct Agent.

        This function invokes the ReAct Agent Graph and returns the response.

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

            state = ReActGraphState(messages=messages)

            # run the ReAct Agent Graph
            state = await graph.ainvoke(state, config={'recursion_limit': (config.max_tool_calls + 1) * 2})
            # setting recursion_limit: 4 allows 1 tool call
            #   - allows the ReAct Agent to perform 1 cycle / call 1 single tool,
            #   - but stops the agent when it tries to call a tool a second time

            # get and return the output from the state
            state = ReActGraphState(**state)
            output_message = state.messages[-1]
            content = str(output_message.content)

            # Create usage statistics for the response
            prompt_tokens = sum(len(str(msg.content).split()) for msg in message.messages)
            completion_tokens = len(content.split()) if content else 0
            total_tokens = prompt_tokens + completion_tokens
            usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
            response = ChatResponse.from_string(content, usage=usage)
            if chat_request_or_message.is_string:
                return GlobalTypeConverter.get().convert(response, to_type=str)
            return response
        except Exception as ex:
            logger.error("%s ReAct Agent failed with exception: %s", AGENT_LOG_PREFIX, str(ex))
            raise

    yield FunctionInfo.from_fn(_response_fn, description=config.description)
