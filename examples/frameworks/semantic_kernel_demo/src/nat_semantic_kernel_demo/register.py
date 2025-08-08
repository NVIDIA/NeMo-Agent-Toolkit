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
async def semantic_kernel_travel_planning_workflow(config: SKTravelPlanningWorkflowConfig, builder: Builder):

    from semantic_kernel import Kernel
    from semantic_kernel.agents import AgentGroupChat
    from semantic_kernel.agents import ChatCompletionAgent
    from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
    from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
    from semantic_kernel.contents.chat_message_content import ChatMessageContent
    from semantic_kernel.contents.utils.author_role import AuthorRole

    class CostOptimizationStrategy(TerminationStrategy):
        """Termination strategy to decide when agents should stop."""

        async def should_agent_terminate(self, agent, history):
            if not history:
                return False
            return any(keyword in history[-1].content.lower()
                       for keyword in ["final plan", "total cost", "more information"])

    kernel = Kernel()

    chat_service = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)

    kernel.add_service(chat_service)

    tools = builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)

    # Zip config.tool names and tools for kernel add plugin
    for tool_name, tool in zip(config.tool_names, tools):
        kernel.add_plugin(plugin=tool, plugin_name=tool_name)

    itinerary_expert_name = config.itinerary_expert_name
    itinerary_expert_instructions = config.itinerary_expert_instructions + config.long_term_memory_instructions
    budget_advisor_name = config.budget_advisor_name
    budget_advisor_instructions = config.budget_advisor_instructions + config.long_term_memory_instructions
    summarize_agent_name = config.summarize_agent_name
    summarize_agent_instructions = config.summarize_agent_instructions + config.long_term_memory_instructions

    agent_itinerary = ChatCompletionAgent(kernel=kernel,
                                          name=itinerary_expert_name,
                                          instructions=itinerary_expert_instructions,
                                          function_choice_behavior=FunctionChoiceBehavior.Required())

    agent_budget = ChatCompletionAgent(kernel=kernel,
                                       name=budget_advisor_name,
                                       instructions=budget_advisor_instructions,
                                       function_choice_behavior=FunctionChoiceBehavior.Required())

    agent_summary = ChatCompletionAgent(kernel=kernel,
                                        name=summarize_agent_name,
                                        instructions=summarize_agent_instructions,
                                        function_choice_behavior=FunctionChoiceBehavior.Auto())

    chat = AgentGroupChat(
        agents=[agent_itinerary, agent_budget, agent_summary],
        termination_strategy=CostOptimizationStrategy(agents=[agent_summary], maximum_iterations=5),
    )

    async def _response_fn(input_message: str) -> str:
        await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=input_message))
        responses = []
        async for content in chat.invoke():
            # Store only the Summarizer Agent's response
            if content.name == summarize_agent_name:
                responses.append(content.content)

        if not responses:
            logging.error("No response was generated.")
            return {"output": "No response was generated. Please try again."}

        return {"output": "\n".join(responses)}

    def convert_dict_to_str(response: dict) -> str:
        return response["output"]

    try:
        yield FunctionInfo.create(single_fn=_response_fn, converters=[convert_dict_to_str])
    except GeneratorExit:
        logger.exception("Exited early!", exc_info=True)
    finally:
        logger.debug("Cleaning up")
