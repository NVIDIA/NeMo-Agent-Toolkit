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

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools.base import BaseTool
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class SequentialExecutionToolConfig(FunctionBaseConfig, name="sequential_execution"):
    sequential_tool_list: list = Field(default_factory=list, description="A list of functions to execute sequentially.")


@register_function(config_type=SequentialExecutionToolConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def sequential_execution(_config: SequentialExecutionToolConfig, builder: Builder):

    tools: list[BaseTool] = builder.get_tools(tool_names=_config.sequential_tool_list,
                                              wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tools_dict = {tool.name: tool for tool in tools}

    async def _async_sequential_execution(initial_tool_input: Any) -> Any:
        sequential_tool_list = _config.sequential_tool_list
        tool_input = initial_tool_input
        for tool_name in sequential_tool_list:
            tool = tools_dict[tool_name]
            tool_response = await tool.ainvoke(tool_input, config=RunnableConfig(callbacks=[]))
            tool_input = tool_response
        return tool_input

    yield FunctionInfo.from_fn(_async_sequential_execution, description="Executes a list of functions sequentially.")
