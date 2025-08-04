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

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

from . import utils
from .prompts import TelemetryMetricsAnalysisAgentPrompts


class TelemetryMetricsAnalysisAgentConfig(FunctionBaseConfig, name="telemetry_metrics_analysis_agent"):
    description: str = Field(default=TelemetryMetricsAnalysisAgentPrompts.TOOL_DESCRIPTION,
                             description="Description of the tool for the triage agent.")
    tool_names: list[str] = []
    llm_name: LLMRef
    prompt: str = Field(default=TelemetryMetricsAnalysisAgentPrompts.PROMPT,
                        description="Main prompt for the telemetry metrics analysis agent.")


@register_function(config_type=TelemetryMetricsAnalysisAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def telemetry_metrics_analysis_agent_tool(config: TelemetryMetricsAnalysisAgentConfig, builder: Builder):
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import SystemMessage
    from langgraph.graph import START
    from langgraph.graph import MessagesState
    from langgraph.graph import StateGraph
    from langgraph.prebuilt import ToolNode
    from langgraph.prebuilt import tools_condition

    async def _arun(host_id: str, alert_type: str) -> str:
        """
        Analyze telemetry metrics for a given host and alert type using LLM-powered reasoning.

        Args:
            host_id (str): Identifier of the host to analyze
            alert_type (str): Type of alert that triggered the analysis

        Returns:
            str: Analysis conclusion from the LLM agent
        """
        utils.log_header("Telemetry Metrics Analysis Agent")

        tools = builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        # Bind tools to LLM for parallel execution
        llm_n_tools = llm.bind_tools(tools, parallel_tool_calls=True)

        # Define agent function that processes messages with LLM
        def telemetry_metrics_analysis_agent(state: MessagesState):
            sys_msg = SystemMessage(content=config.prompt)
            return {"messages": [llm_n_tools.invoke([sys_msg] + state["messages"])]}

        # Build the agent execution graph
        builder_graph = StateGraph(MessagesState)

        # Add nodes for agent and tools
        builder_graph.add_node("telemetry_metrics_analysis_agent", telemetry_metrics_analysis_agent)
        builder_graph.add_node("tools", ToolNode(tools))

        # Configure graph edges for execution flow
        builder_graph.add_edge(START, "telemetry_metrics_analysis_agent")
        builder_graph.add_conditional_edges(
            "telemetry_metrics_analysis_agent",
            tools_condition,
        )
        builder_graph.add_edge("tools", "telemetry_metrics_analysis_agent")

        # Compile the execution graph
        agent_executor = builder_graph.compile()

        # Execute analysis and get response
        input_message = f"Host to investigate: {host_id}. Alert type: {alert_type}"
        response = await agent_executor.ainvoke({"messages": [HumanMessage(content=input_message)]})

        conclusion = response["messages"][-1].content

        utils.log_footer()
        return conclusion

    yield FunctionInfo.from_fn(
        _arun,
        description=config.description,
    )
