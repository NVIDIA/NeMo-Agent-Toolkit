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

import asyncio
import logging
import typing

from pydantic.fields import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.optimizable import OptimizableMixin
from nat.plugins.profiler.decorators.function_tracking import track_function

# flake8: noqa
# Import tools for automatic registration
from . import event_collector_tool
from . import node_status_tool
from . import pod_health_tool
from . import resource_pressure_tool
from . import severity_classifier
from . import utils
from .prompts import K8S_INFRA_MONITOR_PROMPT


class K8sInfraMonitorWorkflowConfig(FunctionBaseConfig, OptimizableMixin, name="k8s_infra_monitor"):
    """
    Configuration for the Kubernetes Infrastructure Monitor workflow.

    This agent orchestrates multiple diagnostic tools to analyze Kubernetes cluster health by:
    1. Checking node status and conditions
    2. Scanning for unhealthy pods and high restart counts
    3. Collecting and analyzing cluster events
    4. Evaluating resource pressure (CPU, memory, disk)
    5. Classifying incident severity based on collected evidence
    """

    tool_names: list[str] = []
    llm_name: LLMRef
    offline_mode: bool = Field(default=True, description="Whether to run in offline mode")
    offline_data_path: str | None = Field(
        default="examples/advanced_agents/k8s_infra_monitor/data/offline_scenarios.json",
        description="Path to the offline scenario dataset in JSON format",
    )
    agent_prompt: str = Field(
        default=K8S_INFRA_MONITOR_PROMPT,
        description="The system prompt for the infrastructure monitor agent.",
    )


@register_function(config_type=K8sInfraMonitorWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def k8s_infra_monitor_workflow(config: K8sInfraMonitorWorkflowConfig, builder: Builder):

    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    if typing.TYPE_CHECKING:
        from langchain_core.language_models.chat_models import BaseChatModel

    llm: "BaseChatModel" = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Get diagnostic tools
    async def _get_tool(tool_name: str):
        return await builder.get_tool(tool_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    tools = await asyncio.gather(*[_get_tool(name) for name in config.tool_names])
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=True)

    severity_tool = await _get_tool("severity_classifier")

    async def monitor_assistant(state: MessagesState):
        sys_msg = SystemMessage(content=config.agent_prompt)
        return {"messages": [await llm_with_tools.ainvoke([sys_msg] + state["messages"])]}

    # Build the LangGraph state machine
    graph_builder = StateGraph(MessagesState)

    tools = await builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    graph_builder.add_node("monitor_assistant", monitor_assistant)
    graph_builder.add_node("tools", ToolNode(tools))

    graph_builder.add_edge(START, "monitor_assistant")
    graph_builder.add_conditional_edges("monitor_assistant", tools_condition)
    graph_builder.add_edge("tools", "monitor_assistant")

    agent_executor = graph_builder.compile()

    @track_function()
    async def _analyze_cluster(input_message: str) -> str:
        """Analyze a Kubernetes cluster health query through diagnostic tools and severity classification.

        Runs the query through the agent for analysis and appends a severity classification
        to the final report.
        """
        output = await agent_executor.ainvoke({"messages": [HumanMessage(content=input_message)]})
        raw_result = output["messages"][-1].content
        if isinstance(raw_result, str):
            result = raw_result.strip()
        else:
            result = ""

        if not result:
            utils.logger.warning("Agent returned empty report (input_length=%d)", len(input_message))
            result = (
                "The agent was unable to generate a diagnostic report for this query. "
                "This may indicate the LLM model is insufficient for the task complexity. "
                "Consider using a larger model (e.g. meta/llama-3.3-70b-instruct).\n\n"
            )

        # Append severity classification
        severity = await severity_tool.arun(result)
        return result + severity

    async def _response_fn(input_message: str) -> str:
        """Process a cluster health query and return analysis with recommendations."""
        try:
            return await _analyze_cluster(input_message)
        finally:
            utils.logger.info("Finished agent execution")

    try:
        if config.offline_mode:
            utils.preload_offline_data(offline_data_path=config.offline_data_path)
            utils.log_header("Running in offline mode", dash_length=120, level=logging.INFO)
        yield _response_fn
    except GeneratorExit:
        utils.logger.info("Exited early!")
    finally:
        utils.logger.info("Cleaning up")
