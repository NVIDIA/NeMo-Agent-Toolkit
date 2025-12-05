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

from nat.agent.react_agent.register import ReActAgentWorkflowConfig
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import ChatResponse

logger = logging.getLogger(__name__)


class PerUserReActAgentWorkflowConfig(ReActAgentWorkflowConfig, name="per_user_react_agent"):
    """
    Per-user version of ReAct Agent for use with per-user function groups like per_user_mcp_client.
    Each user gets their own agent instance with isolated state.
    """
    pass  # Inherit all fields from ReActAgentWorkflowConfig


@register_per_user_function(config_type=PerUserReActAgentWorkflowConfig,
                            input_type=ChatRequest,
                            single_output_type=ChatResponse,
                            framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def per_user_react_agent_workflow(config: PerUserReActAgentWorkflowConfig, builder: Builder):
    """Per-user ReAct Agent - each user gets their own isolated instance."""
    from langchain.schema import BaseMessage
    from langchain_core.messages import trim_messages
    from langgraph.graph.state import CompiledStateGraph

    from nat.agent.base import AGENT_LOG_PREFIX
    from nat.agent.react_agent.agent import ReActAgentGraph
    from nat.agent.react_agent.agent import ReActGraphState
    from nat.agent.react_agent.agent import create_react_agent_prompt
    from nat.data_models.api_server import Usage
    from nat.utils.type_converter import GlobalTypeConverter

    prompt = create_react_agent_prompt(config)
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tools = await builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    if not tools:
        raise ValueError(f"No tools specified for Per-User ReAct Agent '{config.llm_name}'")

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
        try:
            message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequest)
            messages: list[BaseMessage] = trim_messages(messages=[m.model_dump() for m in message.messages],
                                                        max_tokens=config.max_history,
                                                        strategy="last",
                                                        token_counter=len,
                                                        start_on="human",
                                                        include_system=True)
            state = ReActGraphState(messages=messages)
            state = await graph.ainvoke(state, config={'recursion_limit': (config.max_tool_calls + 1) * 2})
            state = ReActGraphState(**state)
            output_message = state.messages[-1]
            content = str(output_message.content)

            prompt_tokens = sum(len(str(msg.content).split()) for msg in message.messages)
            completion_tokens = len(content.split()) if content else 0
            usage = Usage(prompt_tokens=prompt_tokens,
                          completion_tokens=completion_tokens,
                          total_tokens=prompt_tokens + completion_tokens)
            response = ChatResponse.from_string(content, usage=usage)

            if chat_request_or_message.is_string:
                return GlobalTypeConverter.get().convert(response, to_type=str)
            return response
        except Exception as ex:
            logger.error("%s Per-User ReAct Agent failed: %s", AGENT_LOG_PREFIX, str(ex))
            raise

    yield FunctionInfo.from_fn(_response_fn, description=config.description)
