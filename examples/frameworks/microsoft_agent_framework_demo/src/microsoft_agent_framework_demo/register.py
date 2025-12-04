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


class SKTravelPlanningWorkflowConfig(FunctionBaseConfig, name="semantic_kernel"):
    tool_names: list[FunctionRef] = Field(default_factory=list,
                                          description="The list of tools to provide to the semantic kernel.")
    llm_name: LLMRef = Field(description="The LLM model to use with the semantic kernel.")
    verbose: bool = Field(default=False, description="Set the verbosity of the semantic kernel's logging.")
    itinerary_expert_name: str = Field(description="The name of the itinerary expert.")
    itinerary_expert_instructions: str = Field(description="The instructions for the itinerary expert.")
    budget_advisor_name: str = Field(description="The name of the budget advisor.")
    budget_advisor_instructions: str = Field(description="The instructions for the budget advisor.")
    summarize_agent_name: str = Field(description="The name of the summarizer agent.")
    summarize_agent_instructions: str = Field(description="The instructions for the summarizer agent.")
    long_term_memory_instructions: str = Field(default="",
                                               description="The instructions for using the long term memory.")


@register_function(config_type=SKTravelPlanningWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.SEMANTIC_KERNEL])
async def semantic_kernel_travel_planning_workflow_orig(config: SKTravelPlanningWorkflowConfig, builder: Builder):

    from agent_framework import ChatAgent


    chat_service = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)

   
    tools = await builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)


    all_tools = [hotel_price_tool.hotel_price, local_events_tool.local_events]


    itinerary_expert_name = config.itinerary_expert_name
    itinerary_expert_instructions = config.itinerary_expert_instructions + config.long_term_memory_instructions
    budget_advisor_name = config.budget_advisor_name
    budget_advisor_instructions = config.budget_advisor_instructions + config.long_term_memory_instructions
    summarize_agent_name = config.summarize_agent_name
    summarize_agent_instructions = config.summarize_agent_instructions + config.long_term_memory_instructions

    agent_itinerary = ChatAgent(
            name=itinerary_expert_name,
            chat_client=chat_service,
            instructions=itinerary_expert_instructions,
            tools=all_tools
        )

    agent_budget = ChatAgent(
            name=budget_advisor_name,
            chat_client=chat_service,
            instructions=budget_advisor_instructions,
            tools=all_tools
        )



    agent_summary = ChatAgent(
            name=summarize_agent_name,
            chat_client=chat_service,
            instructions=summarize_agent_instructions,
            tools=all_tools
        )



    agents=[agent_itinerary, agent_budget, agent_summary]
    current_agent_idx = -1

    def round_robin_speaker(state: GroupChatStateSnapshot) -> str | None:
        if current_agent_idx > len(agents):
            current_agent_idx = -1
        
        round_idx = state["round_index"]
        history = state["history"]

        if not history:
            return None


        response = any(keyword in history[-1].content.lower()
                    for keyword in ["final plan", "total cost", "more information"])

        print("Current response: ", response)

        if response:
            return None
        else:
            current_agent_idx += 1
            return agents[current_agent_idx]


    from agent_framework import GroupChatBuilder, GroupChatStateSnapshot

    # Build the group chat workflow
    chat = (
        GroupChatBuilder()
        .select_speakers(round_robin_speaker, display_name="Orchestrator")
        .participants(agents)
        .build()
    )


    async def _response_fn(input_message: str) -> str:


        from agent_framework import AgentRunUpdateEvent, WorkflowOutputEvent

        task = input_message

        print(f"Task: {task}\n")
        print("=" * 80)
        # Run the workflow
        async for event in chat.run_stream(task):
            if isinstance(event, AgentRunUpdateEvent):
                # Print streaming agent updates
                print(f"[{event.executor_id}]: {event.data}", end="", flush=True)
            elif isinstance(event, WorkflowOutputEvent):
                # Workflow completed
                final_message = event.data
                author = getattr(final_message, "author_name", "System")
                text = getattr(final_message, "text", str(final_message))
                print(f"\n\n[{author}]\n{text}")
                print("-" * 80)

    def convert_dict_to_str(response: dict) -> str:
        return response["output"]

    try:
        yield FunctionInfo.create(single_fn=_response_fn, converters=[convert_dict_to_str])
    except GeneratorExit:
        logger.exception("Exited early!")
    finally:
        logger.debug("Cleaning up")