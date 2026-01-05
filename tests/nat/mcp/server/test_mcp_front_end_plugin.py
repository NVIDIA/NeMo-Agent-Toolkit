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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.plugins.mcp.server.front_end_config import MCPFrontEndConfig
from nat.plugins.mcp.server.front_end_plugin import MCPFrontEndPlugin
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


async def test_workflow_alias_usage_in_mcp_front_end():
    """Test that workflow_alias is properly used in MCP front end plugin worker."""
    from unittest.mock import MagicMock

    from nat.data_models.config import Config
    from nat.plugins.mcp.server.front_end_plugin_worker import MCPFrontEndPluginWorker

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


async def test_workflow_alias_priority_over_type():
    """Test that workflow_alias takes priority over workflow type when both are present."""
    from unittest.mock import MagicMock

    from nat.data_models.config import Config
    from nat.plugins.mcp.server.front_end_plugin_worker import MCPFrontEndPluginWorker

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


async def test_workflow_alias_with_function_groups():
    """Test that workflow_alias works correctly when function groups are present."""
    from unittest.mock import AsyncMock
    from unittest.mock import MagicMock

    from nat.data_models.config import Config
    from nat.plugins.mcp.server.front_end_plugin_worker import MCPFrontEndPluginWorker

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


async def test_session_manager_creation_for_workflow_vs_function():
    """Test that SessionManager.create is called with correct entry_function for workflows vs regular functions."""
    from unittest.mock import AsyncMock
    from unittest.mock import MagicMock
    from unittest.mock import patch

    from nat.builder.workflow import Workflow
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.data_models.config import Config
    from nat.plugins.mcp.server.front_end_plugin_worker import MCPFrontEndPluginWorker

    # Create a proper config
    config = Config(general=GeneralConfig(front_end=MCPFrontEndConfig()), workflow=EchoFunctionConfig())
    worker = MCPFrontEndPluginWorker(config)

    # Create mock functions - one Workflow and one regular Function
    mock_workflow = MagicMock(spec=Workflow)
    mock_regular_function = MagicMock()  # Regular function, not a Workflow

    # Mock the builder
    mock_builder = MagicMock(spec=WorkflowBuilder)

    # Mock FastMCP
    mock_mcp = MagicMock()

    # Patch _get_all_functions to return our test functions
    with patch.object(worker,
                      '_get_all_functions',
                      return_value={
                          "react_agent": mock_workflow, "echo_function": mock_regular_function
                      }):
        # Patch SessionManager.create to track calls
        with patch('nat.plugins.mcp.server.front_end_plugin_worker.SessionManager.create',
                   new_callable=AsyncMock) as mock_session_create:
            # Configure the mock to return a mock SessionManager
            mock_session_manager = MagicMock()
            mock_session_manager.workflow = mock_workflow
            mock_session_create.return_value = mock_session_manager

            # Patch register_function_with_mcp to avoid actual registration
            with patch('nat.plugins.mcp.server.tool_converter.register_function_with_mcp'):
                # Call the method we're testing
                await worker._default_add_routes(mock_mcp, mock_builder)

        # Verify SessionManager.create was called twice (once for each function)
        assert mock_session_create.call_count == 2

        # Extract the calls
        calls = mock_session_create.call_args_list

        # Find the call for the workflow and the call for the regular function
        workflow_call = None
        function_call = None

        for call in calls:
            # Check the entry_function parameter
            entry_function = call.kwargs.get('entry_function')
            if entry_function is None:
                workflow_call = call
            else:
                function_call = call

        # Verify workflow call used entry_function=None
        assert workflow_call is not None, "Workflow should use entry_function=None"
        assert workflow_call.kwargs['entry_function'] is None

        # Verify regular function call used entry_function=function_name
        assert function_call is not None, "Function should use entry_function=<name>"
        assert function_call.kwargs['entry_function'] == "echo_function"
