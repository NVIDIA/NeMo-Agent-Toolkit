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
from collections.abc import AsyncGenerator

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.api_server import ChatRequest
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class ReasoningFunctionConfig(FunctionBaseConfig, name="reasoning_agent"):
    """
    Defines an AIQ Toolkit function that performs reasoning on the input data.
    Output is passed to the next function in the workflow.

    Designed to be used with an InterceptingFunction.
    """

    llm_name: LLMRef = Field(description="The name of the LLM to use for reasoning.")
    augmented_fn: FunctionRef = Field(description="The name of the function to reason on.")
    verbose: bool = Field(default=False, description="Whether to log detailed information.")
    reasoning_prompt_template: str = Field(
        default=("You are an expert reasoning model task with creating a detailed execution plan"
                 " for a system that has the following description:\n\n"
                 "**Description:** \n{augmented_function_desc}\n\n"
                 "Given the following input and a list of available tools, please provide a detailed step-by-step plan"
                 " that an instruction following system can use to address the input. Ensure the plan includes:\n\n"
                 "1. Identifying the key components of the input.\n"
                 "2. Determining the most suitable tools for each task.\n"
                 "3. Outlining the sequence of actions to be taken.\n\n"
                 "**Input:** \n{input_text}\n\n"
                 "**Tools and description of the tool:** \n{tools}\n\n"
                 "An example plan could look like this:\n\n"
                 "1. Call tool A with input X\n"
                 "2. Call tool B with input Y\n"
                 "3. Interpret the output of tool A and B\n"
                 "4. Return the final result"
                 "\n\n **PLAN:**\n"),
        description="The reasoning model prompt template.")

    instruction_prompt_template: str = Field(
        default=("Answer the following question based on message history: {input_text}"
                 "\n\nHere is a plan for execution that you could use to guide you if you wanted to:"
                 "\n\n{reasoning_output}"
                 "\n\nNOTE: Remember to follow your guidance on how to format output, etc."
                 "\n\n You must respond with the answer to the original question directly to the user."),
        description="The instruction prompt template.")


@register_function(config_type=ReasoningFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def build_reasoning_function(config: ReasoningFunctionConfig, builder: Builder):
    """
    Build a ReasoningFunction from the provided config.

    Args:
        config (ReasoningFunctionConfig): The config for the ReasoningFunction.
        builder (Builder): The Builder instance to use for building the function.

    Returns:
        ReasoningFunction: The built ReasoningFunction.
    """
    from langchain_core.language_models import BaseChatModel
    from langchain_core.prompts import PromptTemplate

    from aiq.agent.base import AGENT_LOG_PREFIX

    def remove_r1_think_tags(text: str):
        pattern = r'(<think>)?.*?</think>\s*(.*)'

        # Add re.DOTALL flag to make . match newlines
        match = re.match(pattern, text, re.DOTALL)

        if match:
            return match.group(2)

        return text

    # Get the LLM to use for reasoning
    llm: BaseChatModel = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Get the augmented function's description
    augmented_function = builder.get_function(config.augmented_fn)

    # For now, we rely on runtime checking for type conversion

    if augmented_function.description and augmented_function.description != "":
        augmented_function_desc = augmented_function.description
    else:
        raise ValueError(f"Function {config.augmented_fn} does not have a description. Cannot augment "
                         f"function without a description.")

    # Get the function dependencies of the augmented function
    function_used_tools = builder.get_function_dependencies(config.augmented_fn).functions
    tool_names_with_desc: list[tuple[str, str]] = []

    for tool in function_used_tools:
        tool_impl = builder.get_function(tool)
        tool_names_with_desc.append((tool, tool_impl.description if hasattr(tool_impl, "description") else ""))

    # Draft the reasoning prompt for the augmented function
    template = PromptTemplate(template=config.reasoning_prompt_template,
                              input_variables=["augmented_function_desc", "input_text", "tools"],
                              validate_template=True)

    downstream_template = PromptTemplate(template=config.instruction_prompt_template,
                                         input_variables=["input_text", "reasoning_output"],
                                         validate_template=True)

    streaming_inner_fn = None
    single_inner_fn = None

    if augmented_function.has_streaming_output:

        async def streaming_inner(
                input_message: ChatRequest) -> AsyncGenerator[augmented_function.streaming_output_type]:
            """
            Perform reasoning on the input text.

            Args:
                input_message (AIQChatRequest): The input text to reason on.
            """

            input_text = "".join([str(message.model_dump()) + "\n" for message in input_message.messages])

            prompt = await template.ainvoke(
                input={
                    "augmented_function_desc": augmented_function_desc,
                    "input_text": input_text,
                    "tools": "\n".join([f"- {tool[0]}: {tool[1]}" for tool in tool_names_with_desc])
                })

            prompt = prompt.to_string()

            # Get the reasoning output from the LLM
            reasoning_output = ""

            async for chunk in llm.astream(prompt):
                reasoning_output += chunk.content

            reasoning_output = remove_r1_think_tags(reasoning_output)

            output = await downstream_template.ainvoke(input={
                "input_text": input_text, "reasoning_output": reasoning_output
            })

            output = output.to_string()

            if config.verbose:
                logger.info("%s Reasoning plan and input to agent: \n\n%s", AGENT_LOG_PREFIX, output)

            async for chunk in augmented_function.acall_stream(output):
                yield chunk

        streaming_inner_fn = streaming_inner

    if augmented_function.has_single_output:

        async def single_inner(input_message: ChatRequest) -> augmented_function.single_output_type:
            """
            Perform reasoning on the input text.

            Args:
                input_message (AIQChatRequest): The input text to reason on.
            """

            input_text = "".join([str(message.model_dump()) + "\n" for message in input_message.messages])

            prompt = await template.ainvoke(
                input={
                    "augmented_function_desc": augmented_function_desc,
                    "input_text": input_text,
                    "tools": "\n".join([f"- {tool[0]}: {tool[1]}" for tool in tool_names_with_desc])
                })

            prompt = prompt.to_string()

            # Get the reasoning output from the LLM
            reasoning_output = ""

            async for chunk in llm.astream(prompt):
                reasoning_output += chunk.content

            reasoning_output = remove_r1_think_tags(reasoning_output)

            output = await downstream_template.ainvoke(input={
                "input_text": input_text, "reasoning_output": reasoning_output
            })

            output = output.to_string()

            if config.verbose:
                logger.info("%s Reasoning plan and input to agent: \n\n%s", AGENT_LOG_PREFIX, output)

            return await augmented_function.acall_invoke(output)

        single_inner_fn = single_inner

    yield FunctionInfo.create(
        single_fn=single_inner_fn,
        stream_fn=streaming_inner_fn,
        description=("Reasoning function that generates a detailed execution plan for a system based on the input."),
        converters=augmented_function.converter_list)
