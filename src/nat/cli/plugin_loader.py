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
"""CLI plugin discovery system for loading plugin-specific commands."""

import logging
from importlib.metadata import entry_points

import click

logger = logging.getLogger(__name__)


def discover_and_load_cli_plugins(cli_group: click.Group) -> None:
    """Discover and load CLI command plugins from installed packages.

    This function uses Python entry points to discover CLI commands provided by
    plugin packages. Plugins register their commands under the 'nat.cli' entry
    point group in their pyproject.toml.

    The function handles import errors gracefully - if a plugin cannot be loaded
    (e.g., due to missing dependencies), it logs a debug message but continues
    loading other plugins.

    Args:
        cli_group: The Click group to add discovered commands to

    Example plugin registration in pyproject.toml:
        [project.entry-points.'nat.cli']
        mcp = "nat.plugins.mcp.cli.commands:mcp_command"
    """
    discovered_eps = entry_points(group='nat.cli')

    for ep in discovered_eps:
        try:
            # Load the command from the entry point
            command = ep.load()

            # Verify it's a Click command or group
            if not isinstance(command, click.Command | click.Group):
                logger.warning("CLI plugin '%s' from '%s' is not a Click command/group, skipping", ep.name, ep.value)
                continue

            # Add the command to the CLI group
            cli_group.add_command(command, name=ep.name)
            logger.debug("Loaded CLI plugin: %s from %s", ep.name, ep.value)

        except ImportError as e:
            # Plugin package not installed or missing dependencies - this is expected
            logger.debug(
                "Could not load CLI plugin '%s' from '%s': %s. "
                "This is expected if the plugin package is not installed.",
                ep.name,
                ep.value,
                e)
        except Exception as e:  # noqa: BLE001
            # Unexpected error - log as warning but continue
            logger.warning("Error loading CLI plugin '%s' from '%s': %s", ep.name, ep.value, e, exc_info=True)
