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

import logging
import typing

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

logger = logging.getLogger(__name__)


class ToolExecutionConfig(BaseModel):
    """Configuration for individual tool execution within sequential execution."""

    use_streaming: bool = Field(default=False, description="Whether to use streaming output for the tool.")


class SequentialExecutorConfig(FunctionBaseConfig, name="sequential_executor"):
    """Configuration for sequential execution of a list of functions."""

    tool_list: list[FunctionRef] = Field(default_factory=list,
                                         description="A list of functions to execute sequentially.")
    tool_execution_config: dict[str, ToolExecutionConfig] = Field(default_factory=dict,
                                                                  description="Optional configuration for each"
                                                                  "tool in the sequential execution tool list."
                                                                  "Keys must match the tool names from the"
                                                                  "tool_list.")
    raise_type_incompatibility: bool = Field(
        default=False,
        description="Default to False. Check if the adjacent tools are type compatible,"
        "which means the output type of the previous function is compatible with the input type of the next function."
        "If set to True, any incompatibility will raise an exception. If set to false, the incompatibility will only"
        "generate a warning message and the sequential execution will continue.")


def _get_function_output_type(function: Function, tool_execution_config: dict[str, ToolExecutionConfig]) -> type:
    function_config = tool_execution_config.get(function.instance_name, None)
    if function_config:
        return function.streaming_output_type if function_config.use_streaming else function.single_output_type
    else:
        return function.single_output_type


def _validate_function_type_compatibility(src_fn: Function,
                                          target_fn: Function,
                                          tool_execution_config: dict[str, ToolExecutionConfig]) -> None:
    src_output_type = _get_function_output_type(src_fn, tool_execution_config)
    target_input_type = target_fn.input_type
    logger.debug(
        f"The output type of the {src_fn.instance_name} function is {str(src_output_type)}, is not compatible with"
        f"the input type of the {target_fn.instance_name} function, which is {str(target_input_type)}")

    is_compatible = DecomposedType.is_type_compatible(src_output_type, target_input_type)
    if not is_compatible:
        raise ValueError(
            f"The output type of the {src_fn.instance_name} function is {str(src_output_type)}, is not compatible with"
            f"the input type of the {target_fn.instance_name} function, which is {str(target_input_type)}")


async def _validate_tool_list_type_compatibility(sequential_executor_config: SequentialExecutorConfig,
                                                 builder: Builder) -> tuple[type, type]:
    tool_list = sequential_executor_config.tool_list
    tool_execution_config = sequential_executor_config.tool_execution_config

    function_list = await builder.get_functions(tool_list)
    if not function_list:
        raise RuntimeError("The function list is empty")
    input_type = function_list[0].input_type

    if len(function_list) > 1:
        for src_fn, target_fn in zip(function_list[0:-1], function_list[1:]):
            try:
                _validate_function_type_compatibility(src_fn, target_fn, tool_execution_config)
            except ValueError as e:
                raise ValueError(f"The sequential tool list has incompatible types: {e}")

    output_type = _get_function_output_type(function_list[-1], tool_execution_config)
    logger.debug(f"The input type of the sequential executor tool list is {str(input_type)},"
                 f"the output type is {str(output_type)}")

    return (input_type, output_type)


@register_function(config_type=SequentialExecutorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def sequential_execution(config: SequentialExecutorConfig, builder: Builder):
    logger.debug(f"Initializing sequential executor with tool list: {config.tool_list}")

    tools: list[BaseTool] = await builder.get_tools(tool_names=config.tool_list,
                                                    wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tools_dict: dict[str, BaseTool] = {tool.name: tool for tool in tools}

    try:
        input_type, output_type = await _validate_tool_list_type_compatibility(config, builder)
    except ValueError as e:
        if config.raise_type_incompatibility:
            logger.error(f"The sequential executor tool list has incompatible types: {e}")
            raise
        else:
            logger.warning(f"The sequential executor tool list has incompatible types: {e}")
            input_type = typing.Any
            output_type = typing.Any
    except Exception as e:
        raise ValueError(f"Error with the sequential executor tool list: {e}")

    # The type annotation of _sequential_function_execution is dynamically set according to the tool list
    async def _sequential_function_execution(initial_tool_input):
        logger.debug(f"Executing sequential executor with tool list: {config.tool_list}")

        tool_list: list[FunctionRef] = config.tool_list
        tool_input = initial_tool_input
        tool_response = None

        for tool_name in tool_list:
            tool = tools_dict[tool_name]
            tool_execution_config = config.tool_execution_config.get(tool_name, None)
            logger.debug(f"Executing tool {tool_name} with input: {tool_input}")
            try:
                if tool_execution_config:
                    if tool_execution_config.use_streaming:
                        output = ""
                        async for chunk in tool.astream(tool_input):
                            output += chunk.content
                        tool_response = output
                    else:
                        tool_response = await tool.ainvoke(tool_input)
                else:
                    tool_response = await tool.ainvoke(tool_input)
            except Exception as e:
                logger.error(f"Error with tool {tool_name}: {e}")
                raise

            # The input of the next tool is the response of the previous tool
            tool_input = tool_response

        return tool_response

    # Dynamically set the annotations for the function
    _sequential_function_execution.__annotations__ = {"initial_tool_input": input_type, "return": output_type}
    logger.debug(f"Sequential executor function annotations: {_sequential_function_execution.__annotations__}")

    yield FunctionInfo.from_fn(_sequential_function_execution,
                               description="Executes a list of functions sequentially."
                               "The input of the next tool is the response of the previous tool.")
