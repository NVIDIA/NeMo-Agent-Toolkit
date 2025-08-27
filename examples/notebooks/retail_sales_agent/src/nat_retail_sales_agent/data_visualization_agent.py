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
from nat.builder.function import Function
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class DataVisualizationAgentConfig(FunctionBaseConfig, name="data_visualization_agent"):
    """
    NeMo Agent toolkit function config for data visualization.
    """
    llm_name: LLMRef = Field(description="The name of the LLM to use")
    tool_names: list[FunctionRef] = Field(description="The names of the tools to use")
    description: str = Field(description="The description of the agent.")
    prompt: str = Field(description="The prompt to use for the agent.")
    graph_summarizer_fn: FunctionRef = Field(description="The function to use for the graph summarizer.")
    hitl_approval_fn: FunctionRef = Field(description="The function to use for the hitl approval.")
    max_retries: int = Field(default=3, description="The maximum number of retries for the agent.")


@register_function(config_type=DataVisualizationAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def data_visualization_agent_function(config: DataVisualizationAgentConfig, builder: Builder):
    from langchain_core.messages import AIMessage
    from langchain_core.messages import BaseMessage
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import SystemMessage
    from langchain_core.messages import ToolMessage
    from langgraph.graph import StateGraph
    from langgraph.prebuilt import ToolNode
    from pydantic import BaseModel

    class AgentState(BaseModel):
        retry_count: int = 0
        messages: list[BaseMessage]
        approved: bool = True

    tools = builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm_n_tools = llm.bind_tools(tools)

    hitl_approval_fn: Function = builder.get_function(config.hitl_approval_fn)
    graph_summarizer_fn: Function = builder.get_function(config.graph_summarizer_fn)

    async def conditional_edge(state: AgentState):
        try:
            logger.debug("Starting the Tool Calling Conditional Edge")
            messages = state.messages
            last_message = messages[-1]
            logger.info("Last message type: %s", type(last_message))
            logger.info("Has tool_calls: %s", hasattr(last_message, 'tool_calls'))
            if hasattr(last_message, 'tool_calls'):
                logger.info("Tool calls: %s", last_message.tool_calls)

            if (hasattr(last_message, 'tool_calls') and last_message.tool_calls and len(last_message.tool_calls) > 0):
                logger.info("Routing to tools - found non-empty tool calls")
                return "tools"
            logger.info("Routing to check_hitl_approval - no tool calls to execute")
            return "check_hitl_approval"
        except Exception as ex:
            logger.error("Error in conditional_edge: %s", ex)
            if hasattr(state, 'retry_count') and state.retry_count >= config.max_retries:
                logger.warning("Max retries reached, returning without meaningful output")
                return "__end__"
            state.retry_count = getattr(state, 'retry_count', 0) + 1
            logger.warning(
                "Error in the conditional edge: %s, retrying %d times out of %d",
                ex,
                state.retry_count,
                config.max_retries,
            )
            return "data_visualization_agent"

    def approval_conditional_edge(state: AgentState):
        """Route to summarizer if user approved, otherwise end"""
        logger.info("Approval conditional edge: %s", state.approved)
        if hasattr(state, 'approved') and not state.approved:
            return "__end__"
        return "summarize"

    def data_visualization_agent(state: AgentState):
        sys_msg = SystemMessage(content=config.prompt)
        messages = state.messages

        if messages and isinstance(messages[-1], ToolMessage):
            last_tool_msg = messages[-1]
            logger.info("Processing tool result: %s", last_tool_msg.content)
            summary_content = f"I've successfully created the visualization. {last_tool_msg.content}"
            return {"messages": [AIMessage(content=summary_content)]}
        logger.info("Normal agent operation - generating response for: %s", messages[-1] if messages else 'no messages')
        return {"messages": [llm_n_tools.invoke([sys_msg] + state.messages)]}

    async def check_hitl_approval(state: AgentState):
        messages = state.messages
        last_message = messages[-1]
        logger.info("Checking hitl approval: %s", state.approved)
        logger.info("Last message type: %s", type(last_message))
        selected_option = await hitl_approval_fn.acall_invoke()
        if selected_option:
            return {"approved": True}
        return {"approved": False}

    async def summarize_graph(state: AgentState):
        """Summarize the graph using the graph summarizer function"""
        image_path = None
        for msg in state.messages:
            if hasattr(msg, 'content') and msg.content:
                content = str(msg.content)
                import re
                img_ext = r'[a-zA-Z0-9_.-]+\.(?:png|jpg|jpeg|gif|svg)'
                pattern = rf'saved to ({img_ext})|({img_ext})'
                match = re.search(pattern, content)
                if match:
                    image_path = match.group(1) or match.group(2)
                    break

        if not image_path:
            image_path = "sales_trend.png"

        logger.info("Extracted image path for summarization: %s", image_path)
        response = await graph_summarizer_fn.ainvoke(image_path)
        return {"messages": [response]}

    try:
        logger.debug("Building and compiling the Agent Graph")
        builder_graph = StateGraph(AgentState)

        builder_graph.add_node("data_visualization_agent", data_visualization_agent)
        builder_graph.add_node("tools", ToolNode(tools))
        builder_graph.add_node("check_hitl_approval", check_hitl_approval)
        builder_graph.add_node("summarize", summarize_graph)

        builder_graph.add_conditional_edges("data_visualization_agent", conditional_edge)

        builder_graph.set_entry_point("data_visualization_agent")
        builder_graph.add_edge("tools", "data_visualization_agent")

        builder_graph.add_conditional_edges("check_hitl_approval", approval_conditional_edge)

        builder_graph.add_edge("summarize", "__end__")

        agent_executor = builder_graph.compile()

        logger.info("Data Visualization Agent Graph built and compiled successfully")

    except Exception as ex:
        logger.error("Failed to build Data Visualization Agent Graph: %s", ex)
        raise

    async def _arun(user_query: str) -> str:
        """
        Visualize data based on user query.

        Args:
            user_query (str): User query to visualize data

        Returns:
            str: Visualization conclusion from the LLM agent
        """
        input_message = f"User query: {user_query}."
        response = await agent_executor.ainvoke({"messages": [HumanMessage(content=input_message)]})

        return response

    try:
        yield FunctionInfo.from_fn(_arun, description=config.description)

    except GeneratorExit:
        print("Function exited early!")
    finally:
        print("Cleaning up retail_sales_agent workflow.")
