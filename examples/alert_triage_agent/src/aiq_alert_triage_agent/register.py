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

# pylint: disable=unused-import
# flake8: noqa

import logging
import os

from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

# Import any tools which need to be automatically registered here
from . import categorizer
from . import hardware_check_tool
from . import host_performance_check_tool
from . import maintenance_check
from . import monitoring_process_check_tool
from . import network_connectivity_check_tool
from . import telemetry_metrics_analysis_agent
from . import telemetry_metrics_host_heartbeat_check_tool
from . import telemetry_metrics_host_performance_check_tool
from . import utils
from .prompts import ALERT_TRIAGE_AGENT_PROMPT


class AlertTriageAgentWorkflowConfig(FunctionBaseConfig, name="alert_triage_agent"):
    """
    Configuration for the Alert Triage Agent workflow. This agent orchestrates multiple diagnostic tools
    to analyze and triage alerts by:
    1. Checking for maintenance windows and known issues
    2. Gathering system metrics, hardware status, and connectivity information
    3. Analyzing telemetry data for patterns and anomalies
    4. Categorizing the root cause based on collected evidence
    """
    tool_names: list[str] = []
    llm_name: str


@register_function(config_type=AlertTriageAgentWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def alert_triage_agent_workflow(config: AlertTriageAgentWorkflowConfig, builder: Builder):

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Get tools for alert triage
    tool_names = config.tool_names
    tools = []
    for tool_name in tool_names:
        tool = builder.get_tool(tool_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        tools.append(tool)
    llm_n_tools = llm.bind_tools(tools, parallel_tool_calls=True)

    categorizer_tool = builder.get_tool("categorizer", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    maintenance_check_tool = builder.get_tool("maintenance_check", wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Define assistant function that processes messages with the LLM
    def ata_assistant(state: MessagesState):
        # Create system message with prompt
        sys_msg = SystemMessage(content=ALERT_TRIAGE_AGENT_PROMPT)
        # Invoke LLM with system message and conversation history
        return {"messages": [llm_n_tools.invoke([sys_msg] + state["messages"])]}

    # Initialize state graph for managing conversation flow
    builder_graph = StateGraph(MessagesState)

    # Get tools specified in config
    tools = builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Add nodes to graph
    builder_graph.add_node("ata_assistant", ata_assistant)
    builder_graph.add_node("tools", ToolNode(tools))

    # Define graph edges to control conversation flow
    builder_graph.add_edge(START, "ata_assistant")
    builder_graph.add_conditional_edges(
        "ata_assistant",
        tools_condition,
    )
    builder_graph.add_edge("tools", "ata_assistant")

    # Compile graph into executable agent
    agent_executor = builder_graph.compile()

    async def _process_alert(input_message: str) -> str:
        """Process an alert through maintenance check, agent analysis, and root cause categorization.

        First checks if there is ongoing maintenance. If not, runs the alert through the agent for
        analysis and finally appends root cause categorization to the result.
        """
        # Check if alert is during maintenance window
        maintenance_result = await maintenance_check_tool.arun(input_message)
        if maintenance_result != maintenance_check.NO_ONGOING_MAINTENANCE_STR:
            return maintenance_result

        # Process alert through agent since no maintenance is occurring
        output = await agent_executor.ainvoke({"messages": [HumanMessage(content=input_message)]})
        result = output["messages"][-1].content

        # Determine and append root cause category
        root_cause = await categorizer_tool.arun(result)
        return result + root_cause

    async def _response_fn(input_message: str) -> str:
        """Process alert message and return analysis with recommendations."""
        try:
            result = await _process_alert(input_message)
            return result
        finally:
            utils.logger.info("Finished agent execution")

    async def _response_test_fn(input_message: str) -> str:
        """Test mode response function that processes multiple alerts from a CSV file.

        Args:
            input_message: Not used in test mode, alerts are read from CSV instead

        Returns:
            Confirmation message after processing completes

        Raises:
            ValueError: If TEST_OUTPUT_RELATIVE_FILEPATH environment variable is not set
        """
        output_filepath = os.getenv("TEST_OUTPUT_RELATIVE_FILEPATH")
        if output_filepath is None:
            raise ValueError("TEST_OUTPUT_RELATIVE_FILEPATH environment variable must be set")

        # Load test alerts from CSV file
        df = utils.load_test_data()
        df["output"] = ""  # Initialize output column
        utils.log_header(f"Processing {len(df)} Alerts")

        # Analyze each alert and store results
        for i, (index, row) in enumerate(df.iterrows()):
            alert_msg = row["alert"]
            utils.log_header(f"Alert {i + 1}/{len(df)}", dash_length=50)
            report = await _process_alert(alert_msg)
            df.loc[df.index == index, "output"] = report
            utils.log_footer(dash_length=50)

        utils.log_header("Saving Results")

        # Write results to output CSV
        output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), output_filepath)
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        utils.log_footer()
        return f"Successfully processed {len(df)} alerts. Results saved to {output_filepath}"

    is_test_mode = utils.is_test_mode()
    try:
        if is_test_mode:
            utils.log_header("Running in test mode", dash_length=120, level=logging.INFO)
            yield _response_test_fn
        else:
            yield _response_fn

    except GeneratorExit:
        utils.logger.info("Exited early!")
    finally:
        utils.logger.info("Cleaning up")
