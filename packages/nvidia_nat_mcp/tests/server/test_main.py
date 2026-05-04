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


def test_mcp_command_registration():
    """Test that MCP command is discoverable via entry points."""
    # Verify the MCP command can be imported
    # Verify it's a valid Click command
    import click

    from nat.plugins.mcp.cli.commands import mcp_command
    assert isinstance(mcp_command, click.Command | click.Group), \
        "mcp_command should be a valid Click command or group"

    # Verify the CLI discovers and loads the MCP command
    from nat.cli.entrypoint import cli
    assert "mcp" in cli.commands, "MCP command should be discovered and registered in CLI"
