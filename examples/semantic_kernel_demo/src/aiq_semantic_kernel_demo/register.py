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

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

from . import hotel_price_tool  # noqa: F401, pylint: disable=unused-import
from . import local_events_tool  # noqa: F401, pylint: disable=unused-import

logger = logging.getLogger(__name__)


class SKTravelPlanningWorkflowConfig(FunctionBaseConfig, name="semantic_kernel"):
    tool_names: list[FunctionRef] = []
    llm_name: LLMRef
    verbose: bool = False


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

    # pylint: disable=invalid-name
    ITINERARY_EXPERT_NAME = "ItineraryExpert"
    ITINERARY_EXPERT_INSTRUCTIONS = """
    You are an itinerary expert specializing in creating detailed travel plans.
    Focus on the attractions, best times to visit, and other important logistics.
    Avoid discussing costs or budgets; leave that to the Budget Advisor.
    You have access to long term memory. Always retrieve user preferences from memory before calling any other tools.
    Remember to add, or make up, all arguments when adding to memory (tags, metadata, conversation, user_id, memory).
    ALWAYS include all five parameters.  Example of inputs are.
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": "Can you find me things to do in New York?",
                    },
                    {
                        "role": "assistant",
                        "content": "I've noted you are looking for things to do in New York.",
                    },
                ],
                "user_id": "user_abc",
                "metadata": {
                    "key_value_pairs": {
                        "type": "travel", "relevance": "high"
                    }
                },
                "memory" : "User is looking for things to do in New York."
            }
    """

    BUDGET_ADVISOR_NAME = "BudgetAdvisor"
    BUDGET_ADVISOR_INSTRUCTIONS = """
    You are a budget advisor skilled at estimating costs for travel plans.
    Your job is to provide detailed pricing estimates, optimize for cost-effectiveness,
    and ensure all travel costs fit within a reasonable budget.
    Avoid giving travel advice or suggesting activities.
    You have access to long term memory. Always retrieve user preferences from memory before calling any other tools.
    Remember to add all arguments when searching memory, including the
    conversation, even if you have to fill in some bits yourself.
    Remember to add, or make up, all arguments when adding to memory (tags, metadata, conversation, user_id, memory).
    ALWAYS include all five parameters.  Example of inputs are.
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": "Hi, I'm Alex. Can you find me hotels under $50?",
                    },
                    {
                        "role": "assistant",
                        "content": "Hello Alex! I've noted you are looking for hotels under $50.",
                    },
                ],
                "user_id": "user_abc",
                "metadata": {
                    "key_value_pairs": {
                        "type": "travel", "relevance": "high"
                    }
                },
                "memory" : "User is looking for hotels under $50."
            }
    """

    SUMMARIZE_AGENT_NAME = "Summarizer"
    SUMMARIZE_AGENT_INSTRUCTIONS = """
    You will summarize and create the final plan and format the output.
    If the total cost is not within a provided budget, provide options or ask for more information
    Compile information into a clear, well-structured, user-friendly travel plan. Include sections for the itinerary,
    cost breakdown, and any notes from the budget advisor. Avoid duplicating information.
    You have access to long term memory. Always retrieve user preferences from memory before calling any other tools,
    and write them when it is appropriate to add a user preference.
    Remember to add all arguments when searching memory, including the conversation, even if you have to fill in some
    bits yourself.
    Remember to add, or make up, all arguments when adding to memory (tags, metadata, conversation, user_id, memory).
    ALWAYS include all five parameters.  Example of inputs are.
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": "Hi, I'm Alex. I'm looking for a trip to New York",
                    },
                    {
                        "role": "assistant",
                        "content": "Hello Alex! I've noted you are looking for a trip to New York.",
                    },
                ],
                "user_id": "user_abc",
                "metadata": {
                    "key_value_pairs": {
                        "type": "travel", "relevance": "high"
                    }
                },
                "memory" : "User is looking for a trip to New York."
            }
    """
    # pylint: enable=invalid-name

    kernel = Kernel()

    chat_service = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)

    kernel.add_service(chat_service)

    tools = builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)

    # Zip config.tool names and tools for kernel add plugin
    for tool_name, tool in zip(config.tool_names, tools):
        kernel.add_plugin(plugin=tool, plugin_name=tool_name)

    agent_itinerary = ChatCompletionAgent(kernel=kernel,
                                          name=ITINERARY_EXPERT_NAME,
                                          instructions=ITINERARY_EXPERT_INSTRUCTIONS,
                                          function_choice_behavior=FunctionChoiceBehavior.Required())

    agent_budget = ChatCompletionAgent(kernel=kernel,
                                       name=BUDGET_ADVISOR_NAME,
                                       instructions=BUDGET_ADVISOR_INSTRUCTIONS,
                                       function_choice_behavior=FunctionChoiceBehavior.Required())

    agent_summary = ChatCompletionAgent(kernel=kernel,
                                        name=SUMMARIZE_AGENT_NAME,
                                        instructions=SUMMARIZE_AGENT_INSTRUCTIONS,
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
            if content.name == SUMMARIZE_AGENT_NAME:
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
