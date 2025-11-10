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
"""Main workflow registration for the self-evaluation agent."""

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

# Import tools for automatic registration
from . import confluence_gatherer  # noqa: F401
from . import gitlab_gatherer  # noqa: F401
from . import jira_gatherer  # noqa: F401

# from . import contribution_aggregator  # Will be added in Step 3
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

    # Import ReAct agent
    from nat.workflows.react_agent import ReActAgentConfig
    from nat.workflows.react_agent import register_react_agent

    # Create ReAct agent configuration
    react_config = ReActAgentConfig(
        tool_names=config.tool_names,
        llm_name=config.llm_name,
        system_prompt=config.agent_prompt,
        verbose=True,
        max_tool_calls=50,  # Allow multiple tool calls for gathering data
        retry_parsing_errors=True,
        max_retries=3,
    )

    # Register and get the ReAct agent
    react_agent_gen = register_react_agent(react_config, builder)
    react_agent = await react_agent_gen.__anext__()

    logger.info("Self-Evaluation Agent initialized with ReAct orchestration")

    yield react_agent

    # Cleanup
    try:
        await react_agent_gen.__anext__()
    except StopAsyncIteration:
        pass
    logger.info("Self-Evaluation Agent workflow cleanup complete")
