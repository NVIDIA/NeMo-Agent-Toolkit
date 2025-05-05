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

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

from . import utils
from .prompts import PipelineNodePrompts


class CategorizerToolConfig(FunctionBaseConfig, name="categorizer"):
    description: str = Field(default="This is a categorization tool used at the end of the pipeline.",
                             description="Description of the tool.")
    llm_name: LLMRef


@register_function(config_type=CategorizerToolConfig)
async def categorizer_tool(config: CategorizerToolConfig, builder: Builder):
    # Set up LLM and chain
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    prompt_template = ChatPromptTemplate([("system", PipelineNodePrompts.CATEGORIZER_PROMPT),
                                          MessagesPlaceholder("msgs")])
    categorization_chain = prompt_template | llm

    async def _arun(report: str) -> str:
        tool_name = "Root Cause Categorizer"
        utils.log_header(tool_name)

        result = await categorization_chain.ainvoke({"msgs": [HumanMessage(content=report)]})

        # Extract the markdown heading level from first line of report (e.g. '#' or '##')
        pound_signs = report.split('\n')[0].split(' ')[0]

        # Format the root cause category section:
        # - Add newlines before and after section
        # - Use extracted heading level for consistency
        # - Add extra newline between category and reasoning for readability
        report_section = f"""\n\n{pound_signs} Root Cause Category\n{result.content.replace('\n', '\n\n')}"""

        # Log the result for tracking
        utils.logger.debug(result.content)
        utils.log_footer()

        return report_section

    yield FunctionInfo.from_fn(_arun, description=config.description)
