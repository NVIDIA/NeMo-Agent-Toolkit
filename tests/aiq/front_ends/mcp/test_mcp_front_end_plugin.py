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

from aiq.data_models.config import Config
from aiq.data_models.config import GeneralConfig
from aiq.front_ends.mcp.mcp_front_end_config import MCPFrontEndConfig
from aiq.front_ends.mcp.mcp_front_end_plugin import MCPFrontEndPlugin
from aiq.test.functions import EchoFunctionConfig


@pytest.fixture
def echo_function_config():
    return EchoFunctionConfig()


@pytest.fixture
def mcp_config(echo_function_config):
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


def test_get_all_functions():
    """Test the _get_all_functions method."""
    # Create a mock workflow
    mock_workflow = MagicMock()
    mock_workflow.functions = {"function1": MagicMock(), "function2": MagicMock()}
    mock_workflow.config.workflow.type = "test_workflow"

    # Create the plugin with a valid config
    config = Config(general=GeneralConfig(front_end=MCPFrontEndConfig()), workflow=EchoFunctionConfig())
    plugin = MCPFrontEndPlugin(full_config=config)

    # Test the method
    functions = plugin._get_all_functions(mock_workflow)

    # Verify that the functions were correctly extracted
    assert "function1" in functions
    assert "function2" in functions
    assert "test_workflow" in functions
    assert len(functions) == 3


@patch.object(MCPFrontEndPlugin, 'run')
def test_filter_functions(mock_run, mcp_config):
    """Test function filtering logic directly."""
    # Create a plugin
    plugin = MCPFrontEndPlugin(full_config=mcp_config)

    # Mock workflow with multiple functions
    mock_workflow = MagicMock()
    mock_workflow.functions = {"echo": MagicMock(), "another_function": MagicMock()}
    mock_workflow.config.workflow.type = "test_workflow"

    # Call _get_all_functions first
    all_functions = plugin._get_all_functions(mock_workflow)
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
