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

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

log = logging.getLogger(__name__)


class CodeGenerationTool(FunctionBaseConfig, name="code_generation"):
    """
    Tool for generating code using the configured LLM.
    """
    llm_name: LLMRef
    verbose: bool = False
    programming_language: str = "Python"
    description: str = ("Useful to generate Python code. For any questions about code generation, you must only use "
                        "this tool!")


@register_function(config_type=CodeGenerationTool)
async def code_generation_tool(config: CodeGenerationTool, builder: Builder):
    from langchain_core.prompts.chat import ChatPromptTemplate

    log.info('Initializing code generation tool\nGetting tool LLM from config')
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    system_prompt = """
You are a helpful code assistant that can teach a junior developer how to code.  Your language of
 choice is {programming_language}. Don't explain the code, just generate the code block itself.
"""
    user_prompt = """
{question}
"""
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
    log.info("Filling tool's prompt variable from config")
    prompt = prompt.partial(programming_language=config.programming_language)
    tool = prompt | llm
    log.info('Initialized code generation tool')

    async def _inner(query: str) -> str:
        log.info('Running code generation tool')
        response = await tool.ainvoke({"question": query})
        if config.verbose:
            log.debug('Tool input was: %s\nTool output is: \n%s', query, response)
        return response.content

    yield FunctionInfo.from_fn(_inner, description=config.description)
