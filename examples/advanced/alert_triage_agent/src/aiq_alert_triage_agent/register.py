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

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from pydantic.fields import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.profiler.decorators.function_tracking import track_function

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
# Import custom evaluator
from .classification_evaluator import register_classification_evaluator
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
    llm_name: LLMRef
    offline_mode: bool = Field(default=True, description="Whether to run in offline model")
    offline_data_path: str | None = Field(
        default="examples/advanced/alert_triage_agent/data/offline_data.csv",
        description="Path to the main offline dataset in CSV format containing alerts and their simulated environments")
    benign_fallback_data_path: str | None = Field(
        default="examples/advanced/alert_triage_agent/data/benign_fallback_offline_data.json",
        description="Path to the JSON file with baseline/normal system behavior data")


@register_function(config_type=AlertTriageAgentWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def alert_triage_agent_workflow(config: AlertTriageAgentWorkflowConfig, builder: Builder):

    llm: BaseChatModel = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

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
    async def ata_assistant(state: MessagesState):
        # Create system message with prompt
        sys_msg = SystemMessage(content=ALERT_TRIAGE_AGENT_PROMPT)
        # Invoke LLM with system message and conversation history
        return {"messages": [await llm_n_tools.ainvoke([sys_msg] + state["messages"])]}

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

    @track_function()
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

    try:
        if config.offline_mode:
            utils.preload_offline_data(offline_data_path=config.offline_data_path,
                                       benign_fallback_data_path=config.benign_fallback_data_path)
            utils.log_header("Running in offline mode", dash_length=120, level=logging.INFO)
            # Note: the output of the offline run will be saved in the output directory set in the config file
            # (the config `output_dir` in the `eval` section)
        yield _response_fn

    except GeneratorExit:
        utils.logger.info("Exited early!")
    finally:
        utils.logger.info("Cleaning up")
