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
"""Tests for CLI plugin discovery system."""

from typing import ClassVar
from unittest.mock import MagicMock
from unittest.mock import patch

import click
import pytest

from nat.cli.plugin_loader import discover_and_load_cli_plugins


class TestPluginLoader:
    """Test CLI plugin discovery and loading."""

    def test_discover_and_load_valid_plugin(self):
        """Test that valid CLI plugins are discovered and loaded."""
        # Create a mock Click command
        mock_command = click.Command(name="test_plugin", callback=lambda: None)

        # Create mock entry point
        mock_ep = MagicMock()
        mock_ep.name = "test"
        mock_ep.value = "test.module:test_command"
        mock_ep.load.return_value = mock_command

        # Create a mock CLI group
        cli_group = click.Group()

        with patch("nat.cli.plugin_loader.entry_points", return_value=[mock_ep]):
            discover_and_load_cli_plugins(cli_group)

        # Verify the command was added
        assert "test" in cli_group.commands
        assert cli_group.commands["test"] == mock_command

    def test_skip_non_click_command(self):
        """Test that non-Click objects are skipped with a warning."""
        # Create a mock entry point that returns a non-Click object
        mock_ep = MagicMock()
        mock_ep.name = "invalid"
        mock_ep.value = "test.module:invalid_object"
        mock_ep.load.return_value = "not a click command"

        cli_group = click.Group()

        with patch("nat.cli.plugin_loader.entry_points", return_value=[mock_ep]):
            with patch("nat.cli.plugin_loader.logger") as mock_logger:
                discover_and_load_cli_plugins(cli_group)

                # Verify warning was logged
                mock_logger.warning.assert_called_once()
                assert "not a Click command/group" in str(mock_logger.warning.call_args)

        # Verify the command was NOT added
        assert "invalid" not in cli_group.commands

    def test_handle_import_error_gracefully(self):
        """Test that ImportError is handled gracefully (plugin not installed)."""
        # Create a mock entry point that raises ImportError
        mock_ep = MagicMock()
        mock_ep.name = "missing_plugin"
        mock_ep.value = "missing.module:command"
        mock_ep.load.side_effect = ImportError("No module named 'missing'")

        cli_group = click.Group()

        with patch("nat.cli.plugin_loader.entry_points", return_value=[mock_ep]):
            with patch("nat.cli.plugin_loader.logger") as mock_logger:
                # Should not raise an exception
                discover_and_load_cli_plugins(cli_group)

                # Verify debug message was logged
                mock_logger.debug.assert_called_once()
                assert "Could not load CLI plugin" in str(mock_logger.debug.call_args)

        # Verify the command was NOT added
        assert "missing_plugin" not in cli_group.commands

    def test_handle_unexpected_error(self):
        """Test that unexpected errors are logged but don't crash."""
        # Create a mock entry point that raises an unexpected error
        mock_ep = MagicMock()
        mock_ep.name = "broken_plugin"
        mock_ep.value = "broken.module:command"
        mock_ep.load.side_effect = RuntimeError("Something went wrong")

        cli_group = click.Group()

        with patch("nat.cli.plugin_loader.entry_points", return_value=[mock_ep]):
            with patch("nat.cli.plugin_loader.logger") as mock_logger:
                # Should not raise an exception
                discover_and_load_cli_plugins(cli_group)

                # Verify warning was logged
                mock_logger.warning.assert_called_once()
                assert "Error loading CLI plugin" in str(mock_logger.warning.call_args)

        # Verify the command was NOT added
        assert "broken_plugin" not in cli_group.commands

    def test_load_multiple_plugins(self):
        """Test that multiple plugins can be loaded."""
        # Create mock commands
        mock_cmd1 = click.Command(name="plugin1", callback=lambda: None)
        mock_cmd2 = click.Command(name="plugin2", callback=lambda: None)

        # Create mock entry points
        mock_ep1 = MagicMock()
        mock_ep1.name = "plugin1"
        mock_ep1.value = "test.module1:cmd1"
        mock_ep1.load.return_value = mock_cmd1

        mock_ep2 = MagicMock()
        mock_ep2.name = "plugin2"
        mock_ep2.value = "test.module2:cmd2"
        mock_ep2.load.return_value = mock_cmd2

        cli_group = click.Group()

        with patch("nat.cli.plugin_loader.entry_points", return_value=[mock_ep1, mock_ep2]):
            discover_and_load_cli_plugins(cli_group)

        # Verify both commands were added
        assert "plugin1" in cli_group.commands
        assert "plugin2" in cli_group.commands
        assert cli_group.commands["plugin1"] == mock_cmd1
        assert cli_group.commands["plugin2"] == mock_cmd2

    def test_load_click_group(self):
        """Test that Click groups (not just commands) can be loaded."""
        # Create a mock Click group
        mock_group = click.Group(name="test_group")

        # Create mock entry point
        mock_ep = MagicMock()
        mock_ep.name = "testgroup"
        mock_ep.value = "test.module:test_group"
        mock_ep.load.return_value = mock_group

        cli_group = click.Group()

        with patch("nat.cli.plugin_loader.entry_points", return_value=[mock_ep]):
            discover_and_load_cli_plugins(cli_group)

        # Verify the group was added
        assert "testgroup" in cli_group.commands
        assert cli_group.commands["testgroup"] == mock_group


@pytest.mark.integration
class TestPluginLoaderIntegration:
    """Integration tests for CLI plugin discovery with real plugins."""

    # Expected core commands that should always be present
    EXPECTED_CORE_COMMANDS: ClassVar[set[str]] = {
        "configure",
        "eval",
        "finetune",
        "info",
        "object-store",
        "optimize",
        "red-team",
        "registry",
        "sizing",
        "start",
        "uninstall",
        "validate",
        "workflow",
    }

    def test_core_commands_discovered(self):
        """Test that all core NAT commands are discovered via entry points."""
        cli_group = click.Group()
        discover_and_load_cli_plugins(cli_group)

        discovered_commands = set(cli_group.commands.keys())
        missing_commands = self.EXPECTED_CORE_COMMANDS - discovered_commands

        assert not missing_commands, f"Missing core commands: {missing_commands}"

    def test_mcp_plugin_discovered(self):
        """Test that MCP plugin is discovered when nvidia-nat-mcp is installed."""
        try:
            # Try importing to check if plugin is available
            import nat.plugins.mcp.cli.commands  # noqa: F401

            cli_group = click.Group()
            discover_and_load_cli_plugins(cli_group)

            # MCP should be discovered and loaded
            assert "mcp" in cli_group.commands
        except ImportError:
            pytest.skip("nvidia-nat-mcp not installed")

    def test_a2a_plugin_discovered(self):
        """Test that A2A plugin is discovered when nvidia-nat-a2a is installed."""
        try:
            # Try importing to check if plugin is available
            import nat.plugins.a2a.cli.commands  # noqa: F401

            cli_group = click.Group()
            discover_and_load_cli_plugins(cli_group)

            # A2A should be discovered and loaded
            assert "a2a" in cli_group.commands
        except ImportError:
            pytest.skip("nvidia-nat-a2a not installed")

    def test_all_commands_together(self):
        """Test that core and plugin commands can coexist."""
        cli_group = click.Group()
        discover_and_load_cli_plugins(cli_group)

        # Should have at minimum all core commands
        min_expected = len(self.EXPECTED_CORE_COMMANDS)
        assert len(cli_group.commands) >= min_expected, \
            f"Should have at least {min_expected} core commands"

        # Verify commands are Click command/group instances
        for name, cmd in cli_group.commands.items():
            assert isinstance(cmd, click.Command | click.Group), f"Command '{name}' is not a valid Click command"

    def test_command_aliases_created(self):
        """Test that 'run' and 'serve' aliases are created from 'start' command."""
        # Import the actual CLI to test the full entrypoint logic
        from nat.cli.entrypoint import cli

        # Verify the start command exists (it's the base for aliases)
        assert "start" in cli.commands, "start command should be discovered"

        # Verify the aliases are created
        assert "run" in cli.commands, "'run' alias should be created from start command"
        assert "serve" in cli.commands, "'serve' alias should be created from start command"

        # Verify they are valid Click commands
        assert isinstance(cli.commands["run"], click.Command | click.Group)
        assert isinstance(cli.commands["serve"], click.Command | click.Group)
