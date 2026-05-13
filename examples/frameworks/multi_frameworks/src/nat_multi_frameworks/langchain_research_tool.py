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

import logging
import re

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LangChainResearchConfig(FunctionBaseConfig, name="langchain_researcher_tool"):
    llm_name: LLMRef
    web_tool: FunctionRef


@register_function(config_type=LangChainResearchConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def langchain_research(tool_config: LangChainResearchConfig, builder: Builder):

    import os

    from bs4 import BeautifulSoup
    from langchain_core.messages import AIMessage
    from langchain_core.prompts import PromptTemplate

    api_token: str | None = os.getenv("NVIDIA_API_KEY")

    if not api_token:
        raise ValueError(
            "API token must be provided in the configuration or in the environment variable `NVIDIA_API_KEY`")

    llm = await builder.get_llm(llm_name=tool_config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tavily_tool = await builder.get_tool(fn_name=tool_config.web_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def web_search(topic: str) -> str:
        output = (await tavily_tool.ainvoke(topic))
        output = output.split("\n\n---\n\n")

        return output[0]

    prompt_template: str = """Extract a single keyword or topic from the following user query \
that can be used to search the web. Return ONLY the keyword or topic, nothing else.

User query: {inputs}
"""
    prompt: PromptTemplate = PromptTemplate(
        input_variables=['inputs'],
        template=prompt_template,
    )

    async def execute_tool(out: AIMessage) -> str:
        topic: str = out.content.strip()
        output_summary: str
        try:
            if topic is not None and topic not in ['', '\n']:
                output_summary = (await web_search(topic))
                # Clean HTML tags from the output
                if isinstance(output_summary, str):
                    # Remove HTML tags using BeautifulSoup
                    soup: BeautifulSoup = BeautifulSoup(output_summary, 'html.parser')
                    output_summary = soup.get_text()
                    # Clean up any extra whitespace
                    output_summary = re.sub(r'\s+', ' ', output_summary).strip()
            else:
                output_summary = f"this search on web search with topic:{topic} yield not results"

        except Exception as e:
            output_summary = f"this search on web search with topic:{topic} yield not results with an error:{e}"
            logger.exception("error in executing tool: %s", e)

        return output_summary

    research = (prompt | llm | execute_tool)

    async def _arun(inputs: str) -> str:
        """
        using web search on a given topic extracted from user input
        Args:
            inputs : user input
        """
        output: str = await research.ainvoke(inputs)
        logger.info("output from langchain_research_tool: %s", output)

        return output

    yield FunctionInfo.from_fn(_arun, description="extract relevent information from search the web")
