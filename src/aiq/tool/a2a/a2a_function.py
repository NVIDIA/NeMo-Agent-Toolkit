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

from typing import Any

from pydantic import Field
from pydantic import HttpUrl

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.tool.function import FunctionBaseConfig

from .a2a_client import A2AToolClient


class A2AFunctionConfig(FunctionBaseConfig, name="a2a_function_wrapper"):
    """Configuration for A2A functions.

    This configuration is used to set up an A2A function wrapper that communicates with an A2A server.
    The wrapper uses the a2a-sdk client under the hood and provides additional functionality like
    retry logic and timeout handling.
    """
    url: HttpUrl = Field(description="URL of the A2A server. Must be a valid HTTP(S) URL.")
    a2a_tool_name: str = Field(
        description="Name of the A2A tool to use. This should match the tool name in the A2A server's agent card.")
    description: str | None = Field(default=None,
                                    description="""
        Description for the tool that will override the description provided by the A2A server.
        Should only be used if the description provided by the server is poor or nonexistent.
        """)
    wait_timeout: int = Field(default=60,
                              gt=0,
                              description="Maximum time in seconds to wait for the A2A server to complete a task.")
    retry_frequency: int = Field(
        default=1, gt=0, description="Time in seconds between retries when polling the A2A server for task completion.")
    post_timeout: int = Field(default=30,
                              gt=0,
                              description="Timeout in seconds for HTTP POST requests to the A2A server.")
    post_sync: bool = Field(default=False,
                            description="""
        If True, use synchronous POST requests for streaming responses.
        This can be useful for certain server configurations but is generally not recommended.
        """)
    parameters: dict[str,
                     Any] = Field(default_factory=dict,
                                  description="Default parameters to include with every request to the A2A server.")


@register_function(config_type=A2AFunctionConfig)
async def a2a_function(config: A2AFunctionConfig, builder: Builder):
    """Generate an AgentIQ Function that wraps a tool provided by the A2A server."""

    # Setup a client for each tool
    tool = A2AToolClient(str(config.url), tool_name=config.a2a_tool_name)

    # Get the AgentCard for the tool. This can be deferred to the first tool call but
    # we need the description if it is not configured in the aiq config file
    await tool.get_card()
    # Set the description if configured otherwise use the AgentCard description
    tool.set_description(config.description)
    tool.set_wait_time(config.wait_timeout, config.retry_frequency)
    tool.set_post_config(config.post_timeout, config.post_sync)

    async def _response_fn(tool_input: str) -> str:
        # Some input adapation may be needed here based on information in the AgentCard
        return await tool.acall(tool_input)

    # TODO: Skip the input schema and converters for now
    yield FunctionInfo.create(
        single_fn=_response_fn,
        description=tool.description,
    )
