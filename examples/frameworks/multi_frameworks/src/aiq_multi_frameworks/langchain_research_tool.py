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
import re

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LangChainResearchConfig(FunctionBaseConfig, name="langchain_researcher_tool"):
    llm_name: LLMRef
    web_tool: FunctionRef


@register_function(config_type=LangChainResearchConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def langchain_research(tool_config: LangChainResearchConfig, builder: Builder):

    import os

    from bs4 import BeautifulSoup
    from langchain_core.prompts import PromptTemplate
    from pydantic import BaseModel
    from pydantic import Field

    api_token = os.getenv("NVIDIA_API_KEY")
    os.environ["NVIDIA_API_KEY"] = api_token

    if not api_token:
        raise ValueError(
            "API token must be provided in the configuration or in the environment variable `NVIDIA_API_KEY`")

    llm = await builder.get_llm(llm_name=tool_config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tavily_tool = builder.get_tool(fn_name=tool_config.web_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def web_search(topic: str) -> list[dict]:
        output = (await tavily_tool.ainvoke(topic))
        output = output.split("\n\n---\n\n")

        return output[0]

    prompt_template = """
    You are an expert of extracting topic from user query in order to search on web search engine on a
    topic extracted from user input.
    ------
    {inputs}
    ------
    The output MUST use the following format :
    '''
    topic: a topic or one keyword to input into web search tool, you can extract ONLY ONE KEYWORD or TOPIC
    '''
    Begin!
    [/INST]
    """
    prompt = PromptTemplate(
        input_variables=['inputs'],
        template=prompt_template,
    )

    class TopicExtract(BaseModel):
        topic: str = Field(description="most important keyword that can be used to search on web search engine")

    llm_with_output_structure = llm.with_structured_output(TopicExtract)

    async def execute_tool(out):
        try:
            topic = out.topic
            if topic is not None and topic not in ['', '\n']:
                output_summary = (await web_search(topic))
                # Clean HTML tags from the output
                if isinstance(output_summary, str):
                    # Remove HTML tags using BeautifulSoup
                    soup = BeautifulSoup(output_summary, 'html.parser')
                    output_summary = soup.get_text()
                    # Clean up any extra whitespace
                    output_summary = re.sub(r'\s+', ' ', output_summary).strip()
            else:
                output_summary = f"this search on web search with topic:{topic} yield not results"

        except Exception as e:
            output_summary = f"this search on web search with topic:{topic} yield not results with an error:{e}"
            logger.exception("error in executing tool: %s", e, exc_info=True)
            pass

        return output_summary

    research = (prompt | llm_with_output_structure | execute_tool)

    async def _arun(inputs: str) -> str:
        """
        using web search on a given topic extracted from user input
        Args:
            inputs : user input
        """
        output = (await research.ainvoke(inputs))
        logger.info("output from langchain_research_tool: %s", output)

        return output

    yield FunctionInfo.from_fn(_arun, description="extract relevent information from search the web")
