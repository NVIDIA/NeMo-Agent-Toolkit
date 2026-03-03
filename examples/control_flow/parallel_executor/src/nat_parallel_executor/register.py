# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class ParallelExecutorConfig(FunctionBaseConfig, name="parallel_executor"):
    """Configuration for parallel execution of a list of functions."""

    description: str = Field(default="Parallel Executor Workflow")
    tool_list: list[FunctionRef] = Field(default_factory=list)
    detailed_logs: bool = Field(default=False)


@register_function(config_type=ParallelExecutorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def parallel_execution(config: ParallelExecutorConfig, builder: Builder) -> AsyncIterator[FunctionInfo]:
    """
    Create a parallel executor for fan-out/fan-in tool execution.

    Parameters
    ----------
    config : ParallelExecutorConfig
        Configuration for the parallel executor
    builder : Builder
        The NAT builder instance

    Returns
    -------
    A FunctionInfo object that executes tools concurrently and merges results
    """
    tools: list[BaseTool] = await builder.get_tools(
        tool_names=config.tool_list,
        wrapper_type=LLMFrameworkEnum.LANGCHAIN,
    )
    tools_dict: dict[str, BaseTool] = {str(tool.name): tool for tool in tools}

    async def parallel_function_execution(input_message: str) -> str:
        workflow_start = perf_counter()
        tasks: list[Any] = []
        tool_names: list[str] = []
        log_prefix = "[parallel_executor]"

        if config.detailed_logs:
            logger.info("%s fan-out start for tools=%s", log_prefix, list(map(str, config.tool_list)))

        async def _invoke_tool(tool_name: str, tool: BaseTool) -> Any:
            branch_start = perf_counter()
            if config.detailed_logs:
                logger.info("%s -> start branch=%s", log_prefix, tool_name)
            try:
                result = await tool.ainvoke(input_message)
            except Exception as exc:
                if config.detailed_logs:
                    logger.exception(
                        "%s <- failed branch=%s duration=%.3fs",
                        log_prefix,
                        tool_name,
                        perf_counter() - branch_start,
                    )
                return exc

            if config.detailed_logs:
                logger.info(
                    "%s <- completed branch=%s duration=%.3fs",
                    log_prefix,
                    tool_name,
                    perf_counter() - branch_start,
                )
            return result

        for tool_name_ref in config.tool_list:
            tool_name = str(tool_name_ref)
            tool = tools_dict.get(tool_name)
            if tool is None:
                raise ValueError(f"Parallel executor: unknown tool '{tool_name}'")
            tasks.append(_invoke_tool(tool_name, tool))
            tool_names.append(tool_name)

        results = await asyncio.gather(*tasks)

        merged: dict[str, str] = {}
        error_count = 0
        for name, result in zip(tool_names, results):
            if isinstance(result, Exception):
                merged[name] = f"ERROR: {result}"
                error_count += 1
            else:
                merged[name] = str(result)

        if config.detailed_logs:
            logger.info(
                "%s fan-in complete duration=%.3fs success=%d error=%d",
                log_prefix,
                perf_counter() - workflow_start,
                len(merged) - error_count,
                error_count,
            )

        return json.dumps(merged)

    yield FunctionInfo.from_fn(parallel_function_execution, description=config.description)
