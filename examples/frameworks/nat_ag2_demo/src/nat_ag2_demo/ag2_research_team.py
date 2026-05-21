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
"""AG2 (formerly AutoGen) research team demo for NAT.

Two agents (researcher + writer) collaborate via AutoPattern GroupChat
to produce a research summary with NAT profiling integration.
"""

from collections.abc import AsyncIterator

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig


class AG2ResearchConfig(FunctionBaseConfig, name="ag2_research_team"):
    """Configuration for the AG2 research team."""

    llm_name: LLMRef = Field(description="NAT LLM config name.")
    max_rounds: int = Field(default=10, description="Max GroupChat rounds.")
    researcher_instructions: str = Field(description="System message for the researcher agent.", )
    writer_instructions: str = Field(description="System message for the writer agent.", )


@register_function(
    config_type=AG2ResearchConfig,
    framework_wrappers=[LLMFrameworkEnum.AG2],
)
async def ag2_research_team(
    config: AG2ResearchConfig,
    builder: Builder,
) -> AsyncIterator[FunctionInfo]:
    """Run a 2-agent research team with AG2.

    A researcher agent investigates the configured topic and a writer agent
    synthesises the findings into a structured summary.

    Args:
        config: Configuration for the research team.
        builder: The NAT workflow builder.

    Yields:
        FunctionInfo wrapping the research team callable.
    """
    from autogen import ConversableAgent
    from autogen.agentchat import a_initiate_group_chat
    from autogen.agentchat.group.patterns import AutoPattern

    llm_config = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.AG2)

    async def _ag2_research_team(task: str) -> str:
        """Run the AG2 research team on the given task.

        Args:
            task: The research topic or question to investigate.

        Returns:
            A structured research summary.
        """
        researcher = ConversableAgent(
            name="researcher",
            system_message=config.researcher_instructions,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

        writer = ConversableAgent(
            name="writer",
            system_message=config.writer_instructions,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

        user = ConversableAgent(name="user", human_input_mode="NEVER")

        pattern = AutoPattern(
            initial_agent=researcher,
            agents=[researcher, writer],
            user_agent=user,
            group_manager_args={"llm_config": llm_config},
        )

        result, _ctx, _last = await a_initiate_group_chat(
            pattern=pattern,
            messages=task,
            max_rounds=config.max_rounds,
        )

        for msg in reversed(result.chat_history):
            content = msg.get("content") or ""
            if content and "TERMINATE" not in content:
                return content

        return "Research complete."

    yield FunctionInfo.from_fn(_ag2_research_team)
