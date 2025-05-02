# pylint: disable=unused-import
# flake8: noqa

# Import any tools which need to be automatically registered here
from agentic_workflow import parse_order_agent
from agentic_workflow import decision_agent
from agentic_workflow import discount_agent
from agentic_workflow import forecast_agent

import logging

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class OrdersWorkflowConfig(FunctionBaseConfig, name="orders"):
    tool_names: list[FunctionRef] = []
    llm_name: LLMRef
    verbose: bool = False


@register_function(config_type=OrdersWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.SEMANTIC_KERNEL])
async def semantic_kernel_travel_planning_workflow(config: OrdersWorkflowConfig, builder: Builder):

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
    SUMMARIZE_AGENT = "SUMMARIZE_AGENT"
    SUMMARIZE_AGENT_INSTRUCTIONS = """
    You are a summarizing agent that summarizes the results of all the agents and then concludes with a result
    """

    # pylint: disable=invalid-name
    PARSE_ORDER_AGENT = "PARSE_ORDER_AGENT"
    PARSE_ORDER_AGENT_INSTRUCTIONS = """
    You are an parsing expert, that extracts the commodity name, cost and quantity.
    You have access to long term memory. Always retrieve user preferences from memory before calling any other tools.
    Remember to add, or make up, all arguments when adding to memory (commodity, quantity,price).
    ALWAYS include all parameters.If comodity name doesn't match with the existing parsed data, say that the commodity
    was not available. In that case, do not approach decision agent, forecast agent and discount agent. Example of inputs are.
    cylinder 50 dollars and 5 of them
    """

    DECISION_AGENT_NAME = "DECISION_AGENT_NAME"
    DECISION_AGENT_INSTRUCTIONS = """
    you need to take the decision based on the commodity's price. Assert what decision was made.
    """

    DISCOUNT_AGENT_NAME = "DISCOUNT_AGENT_NAME"
    DISCOUNT_AGENT_INSTRUCTIONS = """
    apply the relevant discount, Show how much discount has been added.
    """
    # pylint: enable=invalid-name

    FORECAST_AGENT_NAME = "FORECAST_AGENT_NAME"
    FORECAST_AGENT_INSTRUCTIONS = """
    forecast the date and the price, tell the date it was processed with the price.
    """
    # pylint: enable=invalid-name

    kernel = Kernel()

    chat_service = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)

    kernel.add_service(chat_service)

    tools = builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)

    # Zip config.tool names and tools for kernel add plugin
    for tool_name, tool in zip(config.tool_names, tools):
        kernel.add_plugin(plugin=tool, plugin_name=tool_name)

    parse_order_agent = ChatCompletionAgent(kernel=kernel,
                                          name=PARSE_ORDER_AGENT,
                                          instructions=PARSE_ORDER_AGENT_INSTRUCTIONS,
                                          function_choice_behavior=FunctionChoiceBehavior.Required())
    summarize_order_agent = ChatCompletionAgent(kernel=kernel,
                                          name=SUMMARIZE_AGENT,
                                          instructions=SUMMARIZE_AGENT_INSTRUCTIONS,
                                          function_choice_behavior=FunctionChoiceBehavior.Required())

    decision_agent = ChatCompletionAgent(kernel=kernel,
                                       name=DECISION_AGENT_NAME,
                                       instructions=DISCOUNT_AGENT_INSTRUCTIONS,
                                       function_choice_behavior=FunctionChoiceBehavior.Required())

    discount_agent = ChatCompletionAgent(kernel=kernel,
                                        name=DISCOUNT_AGENT_NAME,
                                        instructions=DISCOUNT_AGENT_INSTRUCTIONS,
                                        function_choice_behavior=FunctionChoiceBehavior.Auto())
    forecast_agent = ChatCompletionAgent(kernel=kernel,
                                        name=FORECAST_AGENT_NAME,
                                        instructions=FORECAST_AGENT_INSTRUCTIONS,
                                        function_choice_behavior=FunctionChoiceBehavior.Auto())

    chat = AgentGroupChat(
        agents=[parse_order_agent, decision_agent, discount_agent,forecast_agent,summarize_order_agent],
       # termination_strategy=CostOptimizationStrategy(agents=[agent_summary], maximum_iterations=5),
    )

    async def _response_fn(input_message: str) -> str:
        await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=input_message))
        responses = []
        async for content in chat.invoke():
            # Store only the Summarizer Agent's response
            if content.name == SUMMARIZE_AGENT:
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
