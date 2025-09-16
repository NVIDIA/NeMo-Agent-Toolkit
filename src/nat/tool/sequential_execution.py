# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from langchain_core.runnables import RunnableConfig
from langchain_core.tools.base import BaseTool
from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.function import FunctionBaseConfig
from nat.utils.type_utils import DecomposedType


class ToolExecutionConfig(BaseModel):
    """Configuration for individual tool execution within sequential execution."""

    use_streaming: bool = Field(default=False, description="Whether to use streaming output for the tool.")
    output_type: type | None = Field(
        default=None,
        description=
        "The customized output type of the tool. If set to different type than the output function, the output type will "
        "be converted using the converters.")


class SequentialExecutionToolConfig(FunctionBaseConfig, name="sequential_execution"):
    sequential_tool_list: list[FunctionRef] = Field(default_factory=list,
                                                    description="A list of functions to execute sequentially.")
    tool_execution_config: dict[str, ToolExecutionConfig] = Field(default_factory=dict,
                                                                  description="Optional"
                                                                  "configuration for each tool in the sequential"
                                                                  "execution. Keys should match the tool names from the"
                                                                  "sequential_tool_list.")


def _validate_function_type_compatibility(src_fn: Function,
                                          target_fn: Function,
                                          tool_execution_config: dict[str, ToolExecutionConfig]) -> bool:
    # Validate if the output type of the source function is compatible with the input type of the target function
    src_fn_config = tool_execution_config.get(src_fn.instance_name, None)
    if src_fn_config:
        if src_fn_config.output_type:
            src_output_type = src_fn_config.output_type
        else:
            src_output_type = src_fn.streaming_output_type if src_fn_config.use_streaming else src_fn.single_output_type
    else:
        src_output_type = src_fn.single_output_type

    target_input_type = target_fn.input_type

    return DecomposedType.is_type_compatible(src_output_type, target_input_type)


# Return the input and output types of the sequential tool list
def _validate_sequential_tool_list(sequential_execution_config: SequentialExecutionToolConfig,
                                   builder: Builder) -> tuple[type, type]:
    sequential_tool_list = sequential_execution_config.sequential_tool_list
    tool_execution_config = sequential_execution_config.tool_execution_config

    function_list: list[Function] = []
    for function_ref in sequential_tool_list:
        function_list.append(builder.get_function(function_ref))
    input_type = function_list[0].input_type

    for src_fn, target_fn in zip(function_list[0:-1], function_list[1:]):
        if not _validate_function_type_compatibility(src_fn, target_fn, tool_execution_config):
            raise ValueError(
                f"The output type of the {src_fn.instance_name} function is not compatible with the input type of the {target_fn.instance_name} function"
            )

    last_fn = function_list[-1]
    last_fn_config = tool_execution_config.get(last_fn.instance_name, None)
    if last_fn_config:
        if last_fn_config.output_type:
            output_type = last_fn_config.output_type
        else:
            output_type = last_fn.streaming_output_type if last_fn_config.use_streaming else last_fn.single_output_type
    else:
        output_type = last_fn.single_output_type
    return (input_type, output_type)


@register_function(config_type=SequentialExecutionToolConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def sequential_execution(_config: SequentialExecutionToolConfig, builder: Builder):

    tools: list[BaseTool] = builder.get_tools(tool_names=_config.sequential_tool_list,
                                              wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tools_dict = {tool.name: tool for tool in tools}
    try:
        input_type, output_type = _validate_sequential_tool_list(_config, builder)
    except ValueError as e:
        raise ValueError(f"The sequential tool list has incompatible types: {e}")

    async def _async_sequential_execution(initial_tool_input):
        sequential_tool_list: list[FunctionRef] = _config.sequential_tool_list
        tool_input = initial_tool_input
        tool_response = None
        for tool_name in sequential_tool_list:
            tool = tools_dict[tool_name]
            tool_execution_config = _config.tool_execution_config.get(tool_name, None)
            if tool_execution_config:
                if tool_execution_config.use_streaming:
                    output = ""
                    async for chunk in tool.astream(tool_input, config=RunnableConfig(callbacks=[])):
                        output += chunk.content
                    tool_response = output
                else:
                    tool_response = await tool.ainvoke(tool_input, config=RunnableConfig(callbacks=[]))
            else:
                tool_response = await tool.ainvoke(tool_input, config=RunnableConfig(callbacks=[]))
            tool_input = tool_response

        return tool_response

    # Dynamically set the annotations for the function
    _async_sequential_execution.__annotations__ = {"initial_tool_input": input_type, "return": output_type}

    yield FunctionInfo.from_fn(_async_sequential_execution, description="Executes a list of functions sequentially.")
