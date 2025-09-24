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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.front_ends.mcp.mcp_front_end_config import MCPFrontEndConfig
from nat.front_ends.mcp.mcp_front_end_plugin import MCPFrontEndPlugin
from nat.test.functions import EchoFunctionConfig


@pytest.fixture
def echo_function_config():
    return EchoFunctionConfig()


@pytest.fixture
def mcp_config(echo_function_config) -> Config:
    mcp_front_end_config = MCPFrontEndConfig(name="Test MCP Server",
                                             host="localhost",
                                             port=9901,
                                             debug=False,
                                             log_level="INFO",
                                             tool_names=["echo"])

    return Config(general=GeneralConfig(front_end=mcp_front_end_config),
                  workflow=echo_function_config,
                  functions={"echo": echo_function_config})


def test_mcp_front_end_plugin_init(mcp_config):
    """Test that the MCP front-end plugin can be initialized correctly."""
    # Create the plugin
    plugin = MCPFrontEndPlugin(full_config=mcp_config)

    # Verify that the plugin has the correct config
    assert plugin.full_config is mcp_config
    assert plugin.front_end_config is mcp_config.general.front_end


@pytest.mark.asyncio
async def test_get_all_functions():
    """Test the _get_all_functions method."""
    # Create a mock workflow
    mock_workflow = MagicMock()
    mock_workflow.functions = {"function1": MagicMock(), "function2": MagicMock()}
    mock_workflow.function_groups = {}
    mock_workflow.config.workflow.type = "test_workflow"
    mock_workflow.config.workflow.workflow_alias = None  # No alias, should use type

    # Create the plugin with a valid config
    config = Config(general=GeneralConfig(front_end=MCPFrontEndConfig()), workflow=EchoFunctionConfig())
    plugin = MCPFrontEndPlugin(full_config=config)
    worker = plugin._get_worker_instance()

    # Test the method
    functions = await worker._get_all_functions(mock_workflow)

    # Verify that the functions were correctly extracted
    assert "function1" in functions
    assert "function2" in functions
    assert "test_workflow" in functions
    assert len(functions) == 3


@patch.object(MCPFrontEndPlugin, 'run')
@pytest.mark.asyncio
async def test_filter_functions(_mock_run, mcp_config):
    """Test function filtering logic directly."""
    # Create a plugin
    plugin = MCPFrontEndPlugin(full_config=mcp_config)

    # Mock workflow with multiple functions
    mock_workflow = MagicMock()
    mock_workflow.functions = {"echo": MagicMock(), "another_function": MagicMock()}
    mock_workflow.function_groups = {}
    mock_workflow.config.workflow.type = "test_workflow"
    worker = plugin._get_worker_instance()

    # Call _get_all_functions first
    all_functions = await worker._get_all_functions(mock_workflow)
    assert len(all_functions) == 3

    # Now simulate filtering with tool_names
    mcp_config.general.front_end.tool_names = ["echo"]
    filtered_functions = {}
    for function_name, function in all_functions.items():
        if function_name in mcp_config.general.front_end.tool_names:
            filtered_functions[function_name] = function

    # Verify filtering worked correctly
    assert len(filtered_functions) == 1
    assert "echo" in filtered_functions


@pytest.mark.asyncio
async def test_workflow_alias_usage_in_mcp_front_end():
    """Test that workflow_alias is properly used in MCP front end plugin worker."""
    from unittest.mock import MagicMock

    from nat.data_models.config import Config
    from nat.front_ends.mcp.mcp_front_end_plugin_worker import MCPFrontEndPluginWorker

    # Create a mock workflow with workflow_alias
    mock_workflow = MagicMock()
    mock_workflow.functions = {"func1": MagicMock()}
    mock_workflow.function_groups = {}

    # Test case 1: workflow_alias is set
    mock_workflow.config.workflow.workflow_alias = "custom_workflow_name"
    mock_workflow.config.workflow.type = "original_type"

    # Create a proper config with the required structure
    config = Config(general=GeneralConfig(front_end=MCPFrontEndConfig()), workflow=EchoFunctionConfig())
    worker = MCPFrontEndPluginWorker(config)

    functions = await worker._get_all_functions(mock_workflow)

    # Should include the workflow under the alias name
    assert "custom_workflow_name" in functions
    assert functions["custom_workflow_name"] == mock_workflow
    assert "func1" in functions

    # Test case 2: workflow_alias is None, should use type
    mock_workflow.config.workflow.workflow_alias = None

    functions = await worker._get_all_functions(mock_workflow)

    # Should include the workflow under the type name
    assert "original_type" in functions
    assert functions["original_type"] == mock_workflow
    assert "func1" in functions


@pytest.mark.asyncio
async def test_workflow_alias_priority_over_type():
    """Test that workflow_alias takes priority over workflow type when both are present."""
    from unittest.mock import MagicMock

    from nat.data_models.config import Config
    from nat.front_ends.mcp.mcp_front_end_plugin_worker import MCPFrontEndPluginWorker

    # Create a mock workflow with both workflow_alias and type
    mock_workflow = MagicMock()
    mock_workflow.functions = {}
    mock_workflow.function_groups = {}
    mock_workflow.config.workflow.workflow_alias = "my_custom_alias"
    mock_workflow.config.workflow.type = "original_workflow_type"

    # Create a proper config with the required structure
    config = Config(general=GeneralConfig(front_end=MCPFrontEndConfig()), workflow=EchoFunctionConfig())
    worker = MCPFrontEndPluginWorker(config)

    functions = await worker._get_all_functions(mock_workflow)

    # Should use alias, not type
    assert "my_custom_alias" in functions
    assert "original_workflow_type" not in functions
    assert functions["my_custom_alias"] == mock_workflow


@pytest.mark.asyncio
async def test_workflow_alias_with_function_groups():
    """Test that workflow_alias works correctly when function groups are present."""
    from unittest.mock import AsyncMock
    from unittest.mock import MagicMock

    from nat.data_models.config import Config
    from nat.front_ends.mcp.mcp_front_end_plugin_worker import MCPFrontEndPluginWorker

    # Create mock functions for function group
    mock_func_group = MagicMock()
    mock_func_group.get_accessible_functions = AsyncMock(return_value={
        "group_func1": MagicMock(), "group_func2": MagicMock()
    })

    # Create a mock workflow
    mock_workflow = MagicMock()
    mock_workflow.functions = {"direct_func": MagicMock()}
    mock_workflow.function_groups = {"group1": mock_func_group}
    mock_workflow.config.workflow.workflow_alias = "aliased_workflow"
    mock_workflow.config.workflow.type = "workflow_type"

    # Create a proper config with the required structure
    config = Config(general=GeneralConfig(front_end=MCPFrontEndConfig()), workflow=EchoFunctionConfig())
    worker = MCPFrontEndPluginWorker(config)

    functions = await worker._get_all_functions(mock_workflow)

    # Should include all functions plus workflow under alias
    assert "aliased_workflow" in functions
    assert functions["aliased_workflow"] == mock_workflow
    assert "direct_func" in functions
    assert "group_func1" in functions
    assert "group_func2" in functions
    assert len(functions) == 4  # workflow + 1 direct + 2 group functions
