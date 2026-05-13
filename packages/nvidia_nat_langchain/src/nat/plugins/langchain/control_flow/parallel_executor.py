# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import json
import logging
import typing
from collections.abc import AsyncIterator
from time import perf_counter

from langchain_core.tools.base import BaseTool
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class UnknownParallelToolsError(ValueError):
    """Raised when one or more configured tools cannot be resolved."""

    def __init__(self, tool_names: list[str]):
        formatted_tools = ", ".join(f"'{tool_name}'" for tool_name in tool_names)
        super().__init__(f"Parallel executor: unknown tool(s) {formatted_tools}")


class ParallelExecutorConfig(FunctionBaseConfig, name="parallel_executor"):
    """Configuration for parallel execution of independent tools."""

    description: str = Field(default="Parallel Executor Workflow", description="Description of this functions use.")
    tool_list: list[FunctionRef] = Field(default_factory=list,
                                         description="A list of functions to execute in parallel.")
    detailed_logs: bool = Field(default=False, description="Enable detailed fan-out, per-branch, and fan-in logs.")
    return_error_on_exception: bool = Field(
        default=False,
        description="If set to True, branch exceptions are captured and returned as branch error payloads. "
        "If set to False, the first branch exception is raised.")


async def _invoke_branch(tool_name: str,
                         tool: BaseTool,
                         input_message: object,
                         detailed_logs: bool,
                         log_prefix: str,
                         return_error_on_exception: bool) -> typing.Any:
    branch_start = perf_counter()
    if detailed_logs:
        logger.info("%s -> start branch=%s", log_prefix, tool_name)

    try:
        result = await tool.ainvoke(input_message)
    except Exception as exc:
        if detailed_logs:
            logger.exception("%s <- failed branch=%s duration=%.3fs",
                             log_prefix,
                             tool_name,
                             perf_counter() - branch_start)
        if return_error_on_exception:
            return exc
        raise

    if detailed_logs:
        logger.info("%s <- completed branch=%s duration=%.3fs", log_prefix, tool_name, perf_counter() - branch_start)

    return result


def _format_branch_error(error: Exception) -> str:
    return f"ERROR: {type(error).__name__}: {error}"


@register_function(config_type=ParallelExecutorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def parallel_execution(config: ParallelExecutorConfig, builder: Builder) -> AsyncIterator[FunctionInfo]:
    """Create a parallel executor that fans out input to all tools and fans in branch outputs."""
    logger.debug("Initializing parallel executor with tool list: %s", config.tool_list)

    tools: list[BaseTool] = await builder.get_tools(tool_names=config.tool_list,
                                                    wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tools_dict: dict[str, BaseTool] = {str(tool.name): tool for tool in tools}

    missing_tools = [str(tool_name_ref) for tool_name_ref in config.tool_list if str(tool_name_ref) not in tools_dict]
    if missing_tools:
        raise UnknownParallelToolsError(missing_tools)

    async def _parallel_function_execution(input_message: object) -> str:
        workflow_start = perf_counter()
        log_prefix = "[parallel_executor]"
        tool_names = [str(tool_name_ref) for tool_name_ref in config.tool_list]

        if config.detailed_logs:
            logger.info("%s fan-out start for tools=%s", log_prefix, tool_names)

        tasks = [
            _invoke_branch(
                tool_name=tool_name,
                tool=tools_dict[tool_name],
                input_message=input_message,
                detailed_logs=config.detailed_logs,
                log_prefix=log_prefix,
                return_error_on_exception=config.return_error_on_exception,
            ) for tool_name in tool_names
        ]

        results = await asyncio.gather(*tasks)
        output_blocks: list[str] = []
        error_count = 0
        for tool_name, result in zip(tool_names, results):
            if isinstance(result, Exception):
                output_blocks.append(f"{tool_name}:\n{_format_branch_error(result)}")
                error_count += 1
            else:
                result_text = result if isinstance(result, str) else json.dumps(result, default=str)
                output_blocks.append(f"{tool_name}:\n{result_text}")

        if config.detailed_logs:
            logger.info("%s fan-in complete duration=%.3fs success=%d error=%d",
                        log_prefix,
                        perf_counter() - workflow_start,
                        len(output_blocks) - error_count,
                        error_count)

        return "\n\n".join(output_blocks)

    yield FunctionInfo.from_fn(_parallel_function_execution, description=config.description)
