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
from datetime import datetime

from aiq_profiler_agent.agent import ProfilerAgent
from aiq_profiler_agent.agent import ProfilerAgentState
from aiq_profiler_agent.data_models import ExecPlan
from aiq_profiler_agent.prompts import RETRY_PROMPT
from aiq_profiler_agent.prompts import SYSTEM_PROMPT
from aiq_profiler_agent.tool import flow_chart  # noqa: F401
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class ProfilerAgentConfig(FunctionBaseConfig, name="profiler_agent"):
    """
    Profiler agent config
    """

    llm_name: LLMRef = Field(..., description="The LLM to use for the profiler agent")
    max_iterations: int = Field(..., description="The maximum number of iterations for the profiler agent")
    tools: list[str] = Field(..., description="The tools to use for the profiler agent")

    sys_prompt: str = Field(
        SYSTEM_PROMPT,
        description="The prompt to use for the PxQuery tool.",
    )

    retry_prompt: str = Field(
        RETRY_PROMPT,
        description="Prompt to use when retrying after parser failure",
    )

    max_retries: int = Field(
        ...,
        description="The maximum number of retries for the profiler agent",
    )


@register_function(config_type=ProfilerAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def profiler_agent(config: ProfilerAgentConfig, builder: Builder):
    """
    Profiler agent that uses Phoenix to analyze LLM telemetry data
    This agent retrieves LLM telemetry data using Phoenix's Client API
    and analyzes the data to provide insights about LLM usage, performance,
    and issues.
    """
    from langchain_core.messages import SystemMessage
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import PromptTemplate
    from langgraph.graph.graph import CompiledGraph

    # Create the agent executor
    tools = builder.get_tools(tool_names=config.tools, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    output_parser = PydanticOutputParser(pydantic_object=ExecPlan)
    tools_dict = {tool.name: tool for tool in tools}
    graph: CompiledGraph = await ProfilerAgent(
        llm=llm,
        tools=tools_dict,
        response_composer_tool=builder.get_tool("response_composer", wrapper_type=LLMFrameworkEnum.LANGCHAIN),
        detailed_logs=True,
        max_retries=config.max_retries,
        retry_prompt=config.retry_prompt,
    ).build_graph()

    async def _profiler_agent(input_message: str) -> str:
        """
        Profiler agent that uses Phoenix to analyze LLM telemetry data
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = PromptTemplate(
            template=config.sys_prompt,
            input_variables=["query"],
            partial_variables={
                "current_time": current_time,
                "output_parser": output_parser.get_format_instructions(),
                "tools": "\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
            },
        )

        state = ProfilerAgentState(messages=[SystemMessage(content=prompt.format(query=input_message))], trace_infos={})
        state = await graph.ainvoke(state, config={"recursion_limit": (config.max_iterations + 1) * 2})
        return state["messages"][-1].content

    try:
        yield FunctionInfo.create(single_fn=_profiler_agent)
    except Exception as e:
        logger.error("Error in profiler agent, exit early", exc_info=True)
        raise e
    finally:
        logger.info("Profiler agent finished")
