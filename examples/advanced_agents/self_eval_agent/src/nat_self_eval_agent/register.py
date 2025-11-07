# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Main workflow registration for the self-evaluation agent."""

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

# Import tools for automatic registration
# These will be imported once they are created in the next steps
# from . import jira_gatherer
# from . import gitlab_gatherer
# from . import confluence_gatherer
# from . import contribution_aggregator
from .prompts import SELF_EVAL_AGENT_PROMPT

logger = logging.getLogger(__name__)


class SelfEvalAgentWorkflowConfig(FunctionBaseConfig, name="self_eval_agent"):
    """
    Configuration for the Self-Evaluation Agent workflow.

    This agent orchestrates contribution gathering from multiple platforms:
    1. Collects data from JIRA (issues, comments, resolutions)
    2. Collects data from GitLab (commits, merge requests, reviews)
    3. Collects data from Confluence (pages, updates, comments)
    4. Analyzes and ranks contributions by impact and complexity
    5. Generates top-5 lists or comprehensive self-evaluation reports
    """

    tool_names: list[str] = Field(default_factory=list,
                                  description="List of tool names to use for contribution gathering")
    llm_name: LLMRef = Field(description="LLM to use for analysis and report generation")
    default_time_period_days: int = Field(default=30,
                                          description="Default time period in days to look back for contributions")
    agent_prompt: str = Field(default=SELF_EVAL_AGENT_PROMPT,
                              description="The system prompt to use for the self-evaluation agent")


@register_function(config_type=SelfEvalAgentWorkflowConfig)
async def self_eval_agent_workflow(config: SelfEvalAgentWorkflowConfig, builder: Builder):
    """
    Register the self-evaluation agent workflow.

    This workflow uses a ReAct agent to orchestrate contribution gathering and analysis.
    In Phase 1, we focus on basic aggregation. In later phases, we'll add more
    sophisticated analysis and ranking capabilities.

    Args:
        config: Configuration for the self-evaluation agent
        builder: Builder instance for accessing LLMs and tools

    Yields:
        The configured self-evaluation agent function
    """
    logger.info("Initializing Self-Evaluation Agent workflow")

    # Get the LLM for the agent
    llm = await builder.get_llm(config.llm_name)

    # Get all the tools specified in the configuration
    tools = []
    for tool_name in config.tool_names:
        try:
            tool = await builder.get_function(tool_name)
            tools.append(tool)
            logger.debug(f"Loaded tool: {tool_name}")
        except Exception as e:
            logger.error(f"Failed to load tool {tool_name}: {e}")
            raise

    logger.info(f"Loaded {len(tools)} tools for self-evaluation agent")

    async def _self_eval_agent(user_input: str) -> str:
        """
        Process a self-evaluation request.

        Args:
            user_input: User's request (e.g., "Generate my top 5 contributions from last month")

        Returns:
            Generated self-evaluation report or top contributions list
        """
        logger.info(f"Processing self-evaluation request: {user_input[:100]}...")

        # For Phase 1, we'll use a simple ReAct agent pattern
        # In later phases, this will be replaced with a more sophisticated
        # LangGraph-based workflow

        # Build the agent prompt with available tools
        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])

        full_prompt = f"""{config.agent_prompt}

## Available Tools

{tool_descriptions}

## User Request

{user_input}

Please process this request by:
1. Gathering contributions from the relevant platforms
2. Analyzing and organizing the data
3. Generating the requested report format
"""

        # For now, return a placeholder response
        # In Step 2-3, we'll implement the actual tool calling logic
        response = f"""Self-Evaluation Agent initialized successfully.

Configuration:
- Time period: Last {config.default_time_period_days} days
- Available tools: {len(tools)}
- LLM: {config.llm_name}

User request: {user_input}

[Phase 1: Core infrastructure complete. Tool execution will be implemented in Step 2-3]
"""

        logger.info("Self-evaluation request processed")
        return response

    yield _self_eval_agent

    # Cleanup
    logger.info("Self-Evaluation Agent workflow cleanup complete")
