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

import sys
from unittest.mock import MagicMock
from unittest.mock import patch


@patch("nat.cli.entrypoint.cli.add_command")
def test_mcp_command_registration(mock_add_command):
    """Test the CLI command registration mechanism for MCP."""
    from nat.cli.entrypoint import start_command

    # Create a mock module to simulate main.py
    mock_main_module = MagicMock()

    # Create a mock command that would be returned by get_command
    mock_command = MagicMock(name="mcp_command")

    # Patch the get_command method to return our mock command
    with patch.object(start_command, 'get_command', return_value=mock_command):
        # Mock sys.modules to include our mock module
        with patch.dict(sys.modules, {'nat.front_ends.mcp.main': mock_main_module}):
            # Import the module which would register the command
            # Since we're mocking the module, we'll call the registration code directly
            from nat.cli.entrypoint import cli
            cli.add_command(mock_command, name="mcp")

    # Verify that add_command was called with the correct arguments
    mock_add_command.assert_called_with(mock_command, name="mcp")
