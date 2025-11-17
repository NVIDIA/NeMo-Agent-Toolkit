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
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.optimizable import OptimizableMixin

from . import ping_tool  # pylint: disable=unused-import
from . import url_directory  # pylint: disable=unused-import

logger = logging.getLogger(__name__)


class StrandsDemoConfig(FunctionBaseConfig, OptimizableMixin, name="strands_demo"):
    """
    Configuration for Strands demo workflow.

    Note: OptimizableMixin enables parameter optimization when using `nat optimize`.
    For basic usage, this has no effect and can be ignored.
    """
    tool_names: list[FunctionRef] = Field(
        default_factory=list,
        description="NAT tools exposed to the Strands agent",
    )
    llm_name: LLMRef = Field(description="Model to use via Strands wrapper")
    system_prompt: str | None = Field(default=None, description="Optional system prompt")


@register_function(
    config_type=StrandsDemoConfig,
    framework_wrappers=[LLMFrameworkEnum.STRANDS],
)
async def strands_demo(config: StrandsDemoConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """
    Create a Strands agent workflow that queries documentation URLs.

    This workflow demonstrates NeMo Agent toolkit's Strands integration
    by creating an agent that uses a URL directory and HTTP request tool
    to answer questions about Strands documentation.

    Args:
        config: Configuration specifying LLM, tools, and system prompt
        builder: NeMo Agent toolkit builder for resolving components

    Yields:
        FunctionInfo wrapping the agent execution function that processes
        user inputs and returns agent responses as strings
    """

    from strands import Agent  # type: ignore
    from strands_tools import http_request

    llm = await builder.get_llm(
        config.llm_name,
        wrapper_type=LLMFrameworkEnum.STRANDS,
    )
    nat_tools = await builder.get_tools(
        config.tool_names,
        wrapper_type=LLMFrameworkEnum.STRANDS,
    )

    # Combine NAT tools with Strands http_request tool
    all_tools = [*nat_tools, http_request]

    async def _run(inputs: str) -> str:
        try:

            agent = Agent(model=llm, tools=all_tools, system_prompt=config.system_prompt)

            text: str = ""
            async for ev in agent.stream_async(inputs):
                if "data" in ev:
                    text += ev["data"]
            return text or ""
        except Exception as exc:
            logger.exception("Strands demo failed")
            return f"Error: {exc}"

    yield FunctionInfo.from_fn(
        _run,
        description="Run a Strands agent with URL knowledge base",
    )
