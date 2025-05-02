# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

from . import parse_order_agent  # noqa: F401
from . import discount_agent     # noqa: F401
from . import forecast_agent     # noqa: F401
from . import decision_agent     # noqa: F401

logger = logging.getLogger(__name__)

class GraphiteOrderWorkflowConfig(FunctionBaseConfig, name="graphite_order_workflow"):
    tool_names: list[FunctionRef] = []
    llm_name: LLMRef
    verbose: bool = False


@register_function(config_type=GraphiteOrderWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.SEMANTIC_KERNEL])
async def graphite_order_processing_workflow(config: GraphiteOrderWorkflowConfig, builder: Builder):

    from semantic_kernel import Kernel
    from semantic_kernel.agents import AgentGroupChat
    from semantic_kernel.agents import ChatCompletionAgent
    from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
    from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
    from semantic_kernel.contents.chat_message_content import ChatMessageContent
    from semantic_kernel.contents.utils.author_role import AuthorRole

    class OrderCompletionStrategy(TerminationStrategy):
        """Terminate when final decision is made."""
        async def should_agent_terminate(self, agent, history):
            if not history:
                return False
            return any(keyword in history[-1].content.lower()
                       for keyword in ["order accepted", "order rejected"])

    # Agent Names and Instructions
    PARSE_ORDER_AGENT_NAME = "ParseOrderAgent"
    DISCOUNT_AGENT_NAME = "DiscountAgent"
    FORECAST_AGENT_NAME = "ForecastAgent"
    DECISION_AGENT_NAME = "DecisionAgent"

    PARSE_ORDER_AGENT_INSTRUCTIONS = """
    You are responsible for reading and understanding the customer's order from customer_order.json.
    Extract product type and quantity.
    """

    DISCOUNT_AGENT_INSTRUCTIONS = """
    Your job is to calculate the total price after applying any discounts.
    Use price_reference.csv.
    Apply 10% discount if quantity > 10.
    """

    FORECAST_AGENT_INSTRUCTIONS = """
    You forecast the delivery time.
    Use forecast_reference.csv.
    Multiply quantity with days per unit to find estimated delivery days.
    """

    DECISION_AGENT_INSTRUCTIONS = """
    You make the final decision to accept or reject the order.
    Validate product using product_reference.csv.
    Reject if total price > 5000 or delivery time > 30 days.
    """

    kernel = Kernel()

    chat_service = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)

    kernel.add_service(chat_service)

    tools = builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)

    for tool_name, tool in zip(config.tool_names, tools):
        kernel.add_plugin(plugin=tool, plugin_name=tool_name)

    agent_parser = ChatCompletionAgent(kernel=kernel,
                                       name=PARSE_ORDER_AGENT_NAME,
                                       instructions=PARSE_ORDER_AGENT_INSTRUCTIONS,
                                       function_choice_behavior=FunctionChoiceBehavior.Required())

    agent_discount = ChatCompletionAgent(kernel=kernel,
                                         name=DISCOUNT_AGENT_NAME,
                                         instructions=DISCOUNT_AGENT_INSTRUCTIONS,
                                         function_choice_behavior=FunctionChoiceBehavior.Required())

    agent_forecast = ChatCompletionAgent(kernel=kernel,
                                         name=FORECAST_AGENT_NAME,
                                         instructions=FORECAST_AGENT_INSTRUCTIONS,
                                         function_choice_behavior=FunctionChoiceBehavior.Required())

    agent_decision = ChatCompletionAgent(kernel=kernel,
                                         name=DECISION_AGENT_NAME,
                                         instructions=DECISION_AGENT_INSTRUCTIONS,
                                         function_choice_behavior=FunctionChoiceBehavior.Required())

    chat = AgentGroupChat(
        agents=[agent_parser, agent_discount, agent_forecast, agent_decision],
        termination_strategy=OrderCompletionStrategy(agents=[agent_decision], maximum_iterations=5),
    )

    async def _response_fn(input_message: str) -> str:
        await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=input_message))
        responses = []
        async for content in chat.invoke():
            if content.name == DECISION_AGENT_NAME:
                responses.append(content.content)

        if not responses:
            logging.error("No decision made.")
            return {"output": "Order could not be processed. Try again."}

        return {"output": "\n".join(responses)}

    def convert_dict_to_str(response: dict) -> str:
        return response["output"]

    try:
        yield FunctionInfo.create(single_fn=_response_fn, converters=[convert_dict_to_str])
    except GeneratorExit:
        logger.exception("Exited early!", exc_info=True)
    finally:
        logger.debug("Cleaning up workflow resources")
