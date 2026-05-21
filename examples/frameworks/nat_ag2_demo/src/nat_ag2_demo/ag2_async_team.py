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
"""AG2 async multi-agent team workflow for NAT.

Demonstrates AG2's native async execution path using ``a_initiate_group_chat``.
Tools are executed asynchronously via ``await`` instead of the thread-pool
workaround previously required by the sync path.

Two agents (TrafficAgent + FinalResponseAgent) collaborate via AutoPattern
GroupChat to answer user questions about Los Angeles traffic conditions.
"""

import logging
from collections.abc import AsyncIterator

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class AG2AsyncTeamConfig(FunctionBaseConfig, name="ag2_async_team"):
    """Configuration for the AG2 async multi-agent team workflow."""

    llm_name: LLMRef = Field(description="The LLM model to use with AG2 agents.")
    tool_names: list[str] = Field(default_factory=list, description="List of tool names to be used by the agents.")
    query_processing_agent_name: str = Field(description="Name of the query processing agent.")
    query_processing_agent_instructions: str = Field(description="Instructions for the query processing agent.")
    final_response_agent_name: str = Field(description="Name of the final response agent.")
    final_response_agent_instructions: str = Field(description="Instructions for the final response agent.")
    max_rounds: int = Field(default=20, description="Maximum number of group chat rounds.")


@register_function(config_type=AG2AsyncTeamConfig, framework_wrappers=[LLMFrameworkEnum.AG2])
async def ag2_async_team(config: AG2AsyncTeamConfig, builder: Builder) -> AsyncIterator[FunctionInfo]:
    """AG2 async multi-agent team workflow using a_initiate_group_chat.

    This variant uses AG2's fully async execution path:
    - ``a_initiate_group_chat`` for async group chat orchestration
    - ``a_generate_tool_calls_reply`` for async tool execution
    - Tools are awaited natively (no ThreadPoolExecutor needed)

    Args:
        config: Configuration for the AG2 async team.
        builder: The NAT workflow builder.

    Yields:
        FunctionInfo wrapping the async workflow callable.
    """
    from autogen import ConversableAgent
    from autogen.agentchat import a_initiate_group_chat
    from autogen.agentchat.group.patterns import AutoPattern

    try:
        llm_config = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.AG2)
        tools = await builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.AG2)

        async def _ag2_async_team_workflow(user_input: str) -> str:
            """Execute the AG2 async AutoPattern group chat workflow.

            Args:
                user_input: The user's question.

            Returns:
                The final formatted response from the team.
            """
            try:
                query_agent = ConversableAgent(
                    name=config.query_processing_agent_name,
                    system_message=config.query_processing_agent_instructions,
                    llm_config=llm_config,
                    human_input_mode="NEVER",
                )

                final_agent = ConversableAgent(
                    name=config.final_response_agent_name,
                    system_message=config.final_response_agent_instructions,
                    llm_config=llm_config,
                    human_input_mode="NEVER",
                )

                user = ConversableAgent(name="user", human_input_mode="NEVER")

                for tool in tools:
                    tool.register_for_llm(query_agent)
                    tool.register_for_execution(user)

                pattern = AutoPattern(
                    initial_agent=query_agent,
                    agents=[query_agent, final_agent],
                    user_agent=user,
                    group_manager_args={"llm_config": llm_config},
                )

                # Use the async variant — tools are awaited natively
                result, _ctx, _last = await a_initiate_group_chat(
                    pattern=pattern,
                    messages=user_input,
                    max_rounds=config.max_rounds,
                )

                for msg in reversed(result.chat_history):
                    content = msg.get("content") or ""
                    if content and "APPROVE" not in content:
                        return content

                return "The workflow finished but no output was generated."

            except Exception:
                logger.exception("Error in AG2 async team workflow")
                return "An internal error occurred during the AG2 async team workflow."

        yield FunctionInfo.from_fn(_ag2_async_team_workflow)

    except GeneratorExit:
        logger.info("AG2 async team workflow exited early")
    except Exception as e:
        logger.error("Failed to initialize AG2 async team workflow: %s", e)
        raise
    finally:
        logger.debug("AG2 async team workflow cleanup completed")
