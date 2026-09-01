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

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from . import hotel_price_tool  # noqa: F401, pylint: disable=unused-import
from . import local_events_tool  # noqa: F401, pylint: disable=unused-import

logger = logging.getLogger(__name__)


class CrewAITravelPlanningWorkflowConfig(FunctionBaseConfig, name="crewai"):
    """
    Structural companion to the semantic_kernel_demo example: the same travel-planning task (an itinerary expert,
    a budget advisor, and a summarizer, each backed by the same two tools), reimplemented against CrewAI instead of
    Semantic Kernel, so the two examples can be run through a framework-parity harness as comparable canonical
    workflows.
    """
    tool_names: list[FunctionRef] = Field(default_factory=list,
                                          description="The list of tools to provide to the crew.")
    llm_name: LLMRef = Field(description="The LLM model to use for every agent in the crew.")
    verbose: bool = Field(default=False, description="Set the verbosity of CrewAI's logging.")
    itinerary_expert_role: str = Field(description="The role of the itinerary expert agent.")
    itinerary_expert_goal: str = Field(description="The goal of the itinerary expert agent.")
    itinerary_expert_backstory: str = Field(description="The backstory of the itinerary expert agent.")
    budget_advisor_role: str = Field(description="The role of the budget advisor agent.")
    budget_advisor_goal: str = Field(description="The goal of the budget advisor agent.")
    budget_advisor_backstory: str = Field(description="The backstory of the budget advisor agent.")
    summarize_agent_role: str = Field(description="The role of the summarizer agent.")
    summarize_agent_goal: str = Field(description="The goal of the summarizer agent.")
    summarize_agent_backstory: str = Field(description="The backstory of the summarizer agent.")


@register_function(config_type=CrewAITravelPlanningWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.CREWAI])
async def crewai_travel_planning_workflow(config: CrewAITravelPlanningWorkflowConfig, builder: Builder):

    import asyncio

    from crewai import Agent
    from crewai import Crew
    from crewai import Process
    from crewai import Task

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.CREWAI)
    tools = await builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.CREWAI)

    itinerary_expert = Agent(
        role=config.itinerary_expert_role,
        goal=config.itinerary_expert_goal,
        backstory=config.itinerary_expert_backstory,
        tools=tools,
        llm=llm,
        verbose=config.verbose,
    )

    budget_advisor = Agent(
        role=config.budget_advisor_role,
        goal=config.budget_advisor_goal,
        backstory=config.budget_advisor_backstory,
        tools=tools,
        llm=llm,
        verbose=config.verbose,
    )

    summarizer = Agent(
        role=config.summarize_agent_role,
        goal=config.summarize_agent_goal,
        backstory=config.summarize_agent_backstory,
        llm=llm,
        verbose=config.verbose,
    )

    itinerary_task = Task(
        description=("Using the local events tool, put together a day-by-day itinerary for this trip: {input}. "
                    "Focus on attractions and activities and their cost; leave hotel pricing to the Budget "
                    "Advisor."),
        expected_output="A day-by-day itinerary listing activities and their individual costs.",
        agent=itinerary_expert,
    )

    budget_task = Task(
        description=("Using the hotel price tool, estimate the total lodging cost for this trip: {input}. "
                    "Combine it with the itinerary's activity costs to produce a total estimated trip cost."),
        expected_output="A cost breakdown covering lodging and activities, plus a total estimated cost.",
        agent=budget_advisor,
        context=[itinerary_task],
    )

    summary_task = Task(
        description=("Compile the itinerary and the budget breakdown into a single, clear, well-structured travel "
                    "plan for: {input}."),
        expected_output="A final travel plan with sections for the itinerary and the cost breakdown.",
        agent=summarizer,
        context=[itinerary_task, budget_task],
    )

    crew = Crew(
        agents=[itinerary_expert, budget_advisor, summarizer],
        tasks=[itinerary_task, budget_task, summary_task],
        process=Process.sequential,
        verbose=config.verbose,
    )

    async def _response_fn(input_message: str) -> str:
        # CrewAI's kickoff() is synchronous; run it off the event loop thread so this
        # coroutine doesn't block NAT's async runtime while the crew executes.
        result = await asyncio.to_thread(crew.kickoff, inputs={"input": input_message})
        return str(result)

    try:
        yield FunctionInfo.from_fn(_response_fn, description="Plans a trip using a crew of specialized agents.")
    except GeneratorExit:
        logger.exception("Exited early!")
    finally:
        logger.debug("Cleaning up")
