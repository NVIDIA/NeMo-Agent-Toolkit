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

from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from nat.utils.string_utils import convert_to_str

logger = logging.getLogger(__name__)


class TTCToolWrapperFunctionConfig(FunctionBaseConfig, name="ttc_tool_wrapper"):
    """
    Configuration for the TTCToolWrapperFunction, which is used to wrap a function that will be executed
    in the inference time scaling pipeline.

    This function is responsible for turning an 'objective' or description for the tool into tool input.

    NOTE: Only supports LLMs with structured output.
    """

    augmented_fn: FunctionRef = Field(description="The name of the function to reason on.")

    input_llm: LLMRef = Field(description="The LLM that will generate input to the function.")
    verbose: bool = Field(default=False, description="Whether to log detailed information.")

    downstream_template: str = Field(
        description="The template for the input LLM to generate structured input to the function.",
        default=("You are highly sophisticated generalist AI assistant. Your objective is to act as a"
                 " conduit between a user's task for a function and the function itself. You will be given a general "
                 "description of the task, or pseudo input for a function. You will also be provided with description "
                 "of the function, its input schema, and the output schema. Your task is to generate structured input "
                 "to the function based on the description of the task and the function's input schema. If you do not "
                 "have enough information to generate structured input, you should respond with 'NOT ENOUGH "
                 "INFORMATION'. \n\n The description of the function is: {function_description}\n\n"
                 "The input schema of the function is: {input_schema}\n\n"
                 "The output schema of the function is: {output_schema}\n\n"
                 "The description of the task is: {task_description}\n\n"
                 "The structured input to the function is: "))

    tool_description: str | None = Field(description="The description of the tool to be used for the function.",
                                         default=None)


@register_function(config_type=TTCToolWrapperFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def register_ttc_tool_wrapper_function(
    config: TTCToolWrapperFunctionConfig,
    builder: Builder,
):
    """
    Register the TTCToolWrapperFunction with the provided builder and configuration.
    """

    try:
        from langchain_core.language_models import BaseChatModel
        from langchain_core.prompts import PromptTemplate
    except ImportError:
        raise ImportError("langchain-core is not installed. Please install it to use SingleShotMultiPlanPlanner.\n"
                          "This error can be resolved by installing nvidia-nat-langchain.")

    augmented_function: Function = builder.get_function(config.augmented_fn)
    input_llm: BaseChatModel = await builder.get_llm(config.input_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    if not augmented_function.has_single_output:
        raise ValueError("TTCToolWrapperFunction only supports functions with a single output.")

    if not augmented_function.has_single_output:
        raise ValueError("TTCToolWrapperFunction only supports functions with a single output.")

    if augmented_function.description and augmented_function.description != "":
        augmented_function_desc = augmented_function.description
    else:
        if not config.tool_description:
            raise ValueError(f"Function {config.augmented_fn} does not have a description. Cannot augment "
                             f"function without a description and without a tool description.")

        augmented_function_desc = config.tool_description

    fn_input_schema: BaseModel = augmented_function.input_schema
    fn_output_schema: BaseModel = augmented_function.single_output_schema

    runnable_llm = input_llm.with_structured_output(schema=fn_input_schema)

    template = PromptTemplate(
        template=config.downstream_template,
        input_variables=["function_description", "input_schema", "output_schema", "task_description"],
        validate_template=True)

    function_description = (f"\nDescription: {augmented_function_desc}\n" +
                            "\n Input should be a thorough description with all relevant information on what "
                            f"the tool should do.  The tool requires information about "
                            f"{fn_input_schema.model_fields}")

    async def single_inner(input_message: str) -> fn_output_schema:
        """
        Inner function to handle the streaming output of the TTCToolWrapperFunction.
        It generates structured input for the augmented function based on the input message.
        """

        prompt = await template.ainvoke(
            input={
                "function_description": augmented_function_desc,
                "input_schema": fn_input_schema,
                "output_schema": fn_output_schema,
                "task_description": input_message
            })

        prompt = prompt.to_string()

        if config.verbose:
            logger.info("TTCToolWrapperFunction: Generated prompt: %s", prompt)

        llm_parsed = await runnable_llm.ainvoke(prompt)

        if not llm_parsed:
            logger.warning("TTCToolWrapperFunction: LLM parsing error")
            return "Not enough information"

        # Call the augmented function with the structured input
        result = await augmented_function.acall_invoke(llm_parsed)

        return result

    yield FunctionInfo.from_fn(fn=single_inner, description=function_description, converters=[convert_to_str])
