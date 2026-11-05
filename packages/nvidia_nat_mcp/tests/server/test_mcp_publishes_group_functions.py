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
"""The MCP server publishes the functions a group makes available as tools, with or without `include`."""

import pytest
from mcp.server.fastmcp import FastMCP

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.builder.function_info import FunctionInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_function_group
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.plugins.mcp.server.front_end_config import MCPFrontEndConfig
from nat.plugins.mcp.server.front_end_plugin_worker import MCPFrontEndPluginWorker


class PublishTextToolsConfig(FunctionGroupBaseConfig, name="mcp_publish_test_text_tools"):
    pass


class PublishEchoWorkflowConfig(FunctionBaseConfig, name="mcp_publish_test_echo_workflow"):
    pass


@pytest.fixture(name="register_components", scope="module", autouse=True)
def register_components_fixture():

    @register_function_group(config_type=PublishTextToolsConfig)
    async def _text_tools(config: PublishTextToolsConfig, builder: Builder):
        group = FunctionGroup(config=config)

        async def upper(text: str) -> str:
            return text.upper()

        async def reverse(text: str) -> str:
            return text[::-1]

        group.add_function("upper", upper, description="upper-case the text")
        group.add_function("reverse", reverse, description="reverse the text")
        yield group

    @register_function(config_type=PublishEchoWorkflowConfig)
    async def _workflow(config, builder: Builder):

        async def _inner(text: str) -> str:
            return text

        yield FunctionInfo.from_fn(_inner, description="echo")


@pytest.mark.parametrize("group_options, tool_names, expected",
                         [
                             ({}, [], {"text__upper", "text__reverse", "mcp_publish_test_echo_workflow"}),
                             ({}, ["text"], {"text__upper", "text__reverse"}),
                             ({}, ["text__upper"], {"text__upper"}),
                             ({
                                 "exclude": ["reverse"]
                             }, ["text"], {"text__upper"}),
                             ({
                                 "include": ["upper"]
                             }, ["text"], {"text__upper"}),
                         ],
                         ids=["no_filter", "group_name", "function_name", "exclude", "include"])
async def test_mcp_publishes_the_group_functions(group_options, tool_names, expected):
    config = Config.model_validate({
        "function_groups": {
            "text": {
                "_type": "mcp_publish_test_text_tools", **group_options
            }
        },
        "workflow": {
            "_type": "mcp_publish_test_echo_workflow"
        },
    })
    config.general.front_end = MCPFrontEndConfig(tool_names=tool_names)
    mcp = FastMCP("test")

    async with WorkflowBuilder.from_config(config=config) as builder:
        await MCPFrontEndPluginWorker(config)._default_add_routes(mcp, builder)
        assert {tool.name for tool in await mcp.list_tools()} == expected
        if "text__upper" in expected:
            content, _ = await mcp.call_tool("text__upper", {"text": "abc"})
            assert "ABC" in content[0].text
