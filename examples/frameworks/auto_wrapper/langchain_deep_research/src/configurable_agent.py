# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Research Agent - Standalone script for LangGraph deployment.

This module creates a deep research agent with custom tools and prompts
for conducting web research with strategic thinking and context management.
"""

from datetime import datetime

from deepagents import create_deep_agent
from research_agent.prompts import RESEARCH_WORKFLOW_INSTRUCTIONS
from research_agent.prompts import RESEARCHER_INSTRUCTIONS
from research_agent.prompts import SUBAGENT_DELEGATION_INSTRUCTIONS
from research_agent.tools import tavily_search
from research_agent.tools import think_tool

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.sync_builder import SyncBuilder

# Limits
max_concurrent_research_units = 3
max_researcher_iterations = 3

# Get current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Combine orchestrator instructions (RESEARCHER_INSTRUCTIONS only for sub-agents)
INSTRUCTIONS = (RESEARCH_WORKFLOW_INSTRUCTIONS + "\n\n" + "=" * 80 + "\n\n" + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
    max_concurrent_research_units=max_concurrent_research_units,
    max_researcher_iterations=max_researcher_iterations,
))

# Create research sub-agent
research_sub_agent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": [tavily_search, think_tool],
}

# Model Gemini 3
# model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0.0)

# Model Claude 4.5
# model = init_chat_model(model="anthropic:claude-sonnet-4-5-20250929", temperature=0.0)

# Utilize NAT's builder to get the 'agent' LLM from the config file
model = SyncBuilder.current().get_llm("agent", wrapper_type=LLMFrameworkEnum.LANGCHAIN)

# Create the agent
agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
)
