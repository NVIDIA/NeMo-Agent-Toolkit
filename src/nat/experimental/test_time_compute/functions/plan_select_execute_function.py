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
from collections.abc import AsyncGenerator

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatRequest
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import TTCStrategyRef
from nat.data_models.function import FunctionBaseConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.ttc_item import TTCItem

logger = logging.getLogger(__name__)


class PlanSelectExecuteFunctionConfig(FunctionBaseConfig, name="plan_select_execute_function"):
    """
    Defines an aiqtoolkit function that performs reasoning on the input data.
    Output is passed to the next function in the workflow.

    Designed to be used with an InterceptingFunction.
    """

    augmented_fn: FunctionRef = Field(description="The name of the function to reason on.")

    planner: TTCStrategyRef = Field(description="The configuration for the planner.")
    editor: TTCStrategyRef | None = Field(description="The configuration for the editor.", default=None)
    scorer: TTCStrategyRef | None = Field(description="The configuration for the scorer.", default=None)
    selector: TTCStrategyRef = Field(description="The configuration for the selector.")

    verbose: bool = Field(default=False, description="Whether to log detailed information.")
    agent_context_prompt_template: str = Field(
        description="The template for the agent context prompt. This prompt is used to provide context about the agent",
        default=("\nThe agent system has the following description:\n"
                 "{description}\n"
                 "And has access to the following tools with functionality:\n"
                 "{tools}\n\n"))

    downstream_template: str = Field(
        description=("The template for the downstream prompt. This prompt is used to provide the reasoning output to"
                     " the executing agent"),
        default=("Answer the following question based on message history: {input_text}"
                 "\n\nHere is a plan for execution that you could use to guide you if you wanted to:"
                 "\n\n{reasoning_output}"
                 "\n\nNOTE: Remember to follow your guidance on how to format output, etc."
                 "\n\n You must respond with the answer to the original question directly to the user."))


@register_function(config_type=PlanSelectExecuteFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def plan_select_execute_function(config: PlanSelectExecuteFunctionConfig, builder: Builder):
    """
    Build a ExecutionPlanningFunction from the provided config.

    Args:
        config (ExecutionPlanningFunctionConfig): The config for the ExecutionPlanningFunction.
        builder (Builder): The Builder instance to use for building the function.

    Returns:
        ExecutionPlanningFunction: The built ExecutionPlanningFunction.
    """

    try:
        from langchain_core.prompts import PromptTemplate
    except ImportError:
        raise ImportError("langchain-core is not installed. Please install it to use SingleShotMultiPlanPlanner.\n"
                          "This error can be resolved by installing aiqtoolkit-langchain.")

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
    tool_list = "Tool: Description\n"

    for tool in function_used_tools:
        tool_impl = builder.get_function(tool)
        tool_list += f"- {tool}: {tool_impl.description if hasattr(tool_impl, 'description') else ''}\n"

    # Draft the reasoning prompt for the augmented function
    template = PromptTemplate(template=config.agent_context_prompt_template,
                              input_variables=["description", "tools"],
                              validate_template=True)

    downstream_template = PromptTemplate(template=config.downstream_template,
                                         input_variables=["input_text", "reasoning_output"],
                                         validate_template=True)

    planner = await builder.get_ttc_strategy(strategy_name=config.planner,
                                             pipeline_type=PipelineTypeEnum.PLANNING,
                                             stage_type=StageTypeEnum.SEARCH)

    selector = await builder.get_ttc_strategy(strategy_name=config.selector,
                                              pipeline_type=PipelineTypeEnum.PLANNING,
                                              stage_type=StageTypeEnum.SELECTION)

    if config.editor:
        editor = await builder.get_ttc_strategy(strategy_name=config.editor,
                                                pipeline_type=PipelineTypeEnum.PLANNING,
                                                stage_type=StageTypeEnum.EDITING)
    else:
        editor = None

    if config.scorer:
        scorer = await builder.get_ttc_strategy(strategy_name=config.scorer,
                                                pipeline_type=PipelineTypeEnum.PLANNING,
                                                stage_type=StageTypeEnum.SCORING)
    else:
        scorer = None

    async def planning_pipeline(prompt, context):

        plans = await planner.ainvoke([TTCItem()], prompt, context)

        if editor:
            plans = await editor.ainvoke(plans, prompt, context)
        if scorer:
            plans = await scorer.ainvoke(plans, prompt, context)

        selected_plan = (await selector.ainvoke(plans, prompt, context))[0]

        return selected_plan

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

            context_prompt = await template.ainvoke(input={"description": augmented_function_desc, "tools": tool_list})

            context_prompt = context_prompt.to_string()

            # Run the TTC pipeline
            planning_item: TTCItem = await planning_pipeline(prompt=input_text, context=context_prompt)

            output = await downstream_template.ainvoke(input={
                "input_text": input_text, "reasoning_output": planning_item.plan
            })

            output = output.to_string()

            if config.verbose:
                logger.info("Reasoning plan and input to agent: \n\n%s", output)

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

            context_prompt = await template.ainvoke(input={"description": augmented_function_desc, "tools": tool_list})

            context_prompt = context_prompt.to_string()

            # Run the TTC pipeline
            planning_item: TTCItem = await planning_pipeline(prompt=input_text, context=context_prompt)

            output = await downstream_template.ainvoke(input={
                "input_text": input_text, "reasoning_output": planning_item.plan
            })

            output = output.to_string()

            if config.verbose:
                logger.info("Reasoning plan and input to agent: \n\n%s", output)

            return await augmented_function.acall_invoke(output)

        single_inner_fn = single_inner

    yield FunctionInfo.create(
        single_fn=single_inner_fn,
        stream_fn=streaming_inner_fn,
        description=("Function that runs an TTC execution planner on input and sends plan downstream"),
        converters=augmented_function.converter_list)
