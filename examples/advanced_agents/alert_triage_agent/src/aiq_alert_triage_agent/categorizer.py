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

import re

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

from . import utils
from .prompts import CategorizerPrompts


class CategorizerToolConfig(FunctionBaseConfig, name="categorizer"):
    description: str = Field(default=CategorizerPrompts.TOOL_DESCRIPTION, description="Description of the tool.")
    llm_name: LLMRef
    prompt: str = Field(default=CategorizerPrompts.PROMPT, description="Main prompt for the categorization task.")


def _extract_markdown_heading_level(report: str) -> str:
    """ Extract the markdown heading level from first line (report title)."""
    m = re.search(r'^(#+)', report, re.MULTILINE)
    pound_signs = m.group(1) if m else "#"
    return pound_signs


@register_function(config_type=CategorizerToolConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def categorizer_tool(config: CategorizerToolConfig, builder: Builder):
    # Set up LLM and chain
    from langchain_core.messages import HumanMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.prompts import MessagesPlaceholder

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    prompt_template = ChatPromptTemplate([("system", config.prompt), MessagesPlaceholder("msgs")])
    categorization_chain = prompt_template | llm

    async def _arun(report: str) -> str:
        tool_name = "Root Cause Categorizer"
        utils.log_header(tool_name)

        result = await categorization_chain.ainvoke({"msgs": [HumanMessage(content=report)]})

        # Extract the title's heading level and add an additional '#' for the section heading
        pound_signs = _extract_markdown_heading_level(report) + "#"

        # Format the root cause category section:
        # - Add newlines before and after section
        # - Use extracted heading level for consistency
        # - Add extra newline between category and reasoning for readability
        report_content = result.content.replace('\n', '\n\n')
        report_section = f"""\n\n{pound_signs} Root Cause Category\n{report_content}"""

        # Log the result for tracking
        utils.logger.debug(report_content)
        utils.log_footer()

        return report_section

    yield FunctionInfo.from_fn(_arun, description=config.description)
