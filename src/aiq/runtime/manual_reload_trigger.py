# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Manual reload trigger for AIQ Toolkit configuration hot-reloading.

This module provides utilities for manually triggering configuration reloads
during development and testing. It's designed to help developers test and
validate the hot-reloading functionality before automatic reloading is implemented.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from aiq.runtime.config_manager import ConfigManager
from aiq.runtime.config_manager import ConfigReloadError
from aiq.runtime.config_manager import ConfigValidationError
from aiq.runtime.loader import reload_config

logger = logging.getLogger(__name__)


@click.group()
@click.option('--config',
              '-c',
              'config_file',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Path to the configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx: click.Context, config_file: Path, verbose: bool) -> None:
    """Manual configuration reload trigger for AIQ Toolkit development."""

    # Set up logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Store config file in context
    ctx.ensure_object(dict)
    ctx.obj['config_file'] = config_file


@cli.command()
@click.option('--validate-only', is_flag=True, help='Only validate the configuration without applying changes')
@click.pass_context
def reload(ctx: click.Context, validate_only: bool) -> None:
    """Reload configuration from file with validation."""

    config_file = ctx.obj['config_file']

    try:
        click.echo(f"{'Validating' if validate_only else 'Reloading'} configuration from {config_file}")

        new_config = reload_config(config_file, validate_only=validate_only)

        if validate_only:
            click.echo("Configuration validation successful!")
        else:
            click.echo("Configuration reload successful!")

        # Show basic config info
        click.echo("Configuration summary:")

        config_dict = new_config.model_dump()
        if 'llms' in config_dict:
            llm_count = len(config_dict['llms'])
            click.echo(f"   - LLMs: {llm_count}")

        if 'tools' in config_dict:
            tool_count = len(config_dict['tools'])
            click.echo(f"   - Tools: {tool_count}")

        if 'workflows' in config_dict:
            workflow_count = len(config_dict['workflows'])
            click.echo(f"   - Workflows: {workflow_count}")

    except ConfigValidationError as e:
        click.echo(f"Configuration validation failed: {e}", err=True)
        sys.exit(1)
    except ConfigReloadError as e:
        click.echo(f"Configuration reload failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--overrides',
              '-o',
              multiple=True,
              help='Configuration overrides in key=value format (e.g., llms.temperature=0.7)')
@click.pass_context
def interactive(ctx: click.Context, overrides: tuple[str, ...]) -> None:
    """Start interactive configuration manager session."""

    config_file = ctx.obj['config_file']

    try:
        # Parse overrides
        override_dict = {}
        for override in overrides:
            if '=' not in override:
                click.echo(f"Invalid override format: {override}. Use key=value format.", err=True)
                return
            key, value = override.split('=', 1)
            override_dict[key.strip()] = value.strip()

        click.echo(f"Starting interactive configuration manager for {config_file}")

        with ConfigManager(config_file) as manager:
            if override_dict:
                manager.set_overrides(override_dict)
                click.echo(f"Applied {len(override_dict)} configuration overrides")

            click.echo("\nInteractive Configuration Manager")
            click.echo("Commands:")
            click.echo("  r | reload     - Reload configuration")
            click.echo("  v | validate   - Validate configuration without reloading")
            click.echo("  s | snapshots  - Show configuration snapshots")
            click.echo("  b [steps]      - Rollback configuration (default: 1 step)")
            click.echo("  c | clear      - Clear configuration snapshots")
            click.echo("  i | info       - Show configuration info")
            click.echo("  q | quit       - Quit interactive mode")
            click.echo("-" * 50)

            while True:
                try:
                    command = click.prompt("\nConfig", default="info").strip().lower()

                    if command in ('q', 'quit'):
                        break
                    elif command in ('r', 'reload'):
                        handle_reload_command(manager)
                    elif command in ('v', 'validate'):
                        handle_validate_command(manager)
                    elif command.startswith('b') or command == 'rollback':
                        handle_rollback_command(manager, command)
                    elif command in ('s', 'snapshots'):
                        handle_snapshots_command(manager)
                    elif command in ('c', 'clear'):
                        handle_clear_command(manager)
                    elif command in ('i', 'info'):
                        handle_info_command(manager)
                    else:
                        click.echo("Unknown command. Type 'q' to quit or see commands above.")

                except KeyboardInterrupt:
                    click.echo("\nInterrupted")
                    break
                except Exception as e:
                    click.echo(f"Error: {e}", err=True)

    except Exception as e:
        click.echo(f"Failed to start interactive session: {e}", err=True)
        sys.exit(1)


def handle_reload_command(manager: ConfigManager) -> None:
    """Handle the reload command."""
    try:
        click.echo("Reloading configuration...")
        manager.reload_config()
        click.echo(f"Configuration reloaded successfully (reload #{manager.reload_count})")
    except ConfigValidationError as e:
        click.echo(f"Configuration validation failed: {e}", err=True)
    except ConfigReloadError as e:
        click.echo(f"Configuration reload failed: {e}", err=True)


def handle_validate_command(manager: ConfigManager) -> None:
    """Handle the validate command."""
    try:
        manager.reload_config(validate_only=True)
        click.echo("Configuration validation successful!")
    except ConfigValidationError as e:
        click.echo(f"Configuration validation failed: {e}", err=True)


def handle_rollback_command(manager: ConfigManager, command: str) -> None:
    """Handle the rollback command."""
    try:
        # Parse steps from command (e.g., "b 2" or "rollback 3")
        parts = command.split()
        steps = 1
        if len(parts) > 1:
            try:
                steps = int(parts[1])
            except ValueError:
                click.echo("Invalid rollback steps. Use: b [number]", err=True)
                return

        if steps <= 0:
            click.echo("Invalid rollback steps. Use: b [number]", err=True)
            return

        for _ in range(steps):
            manager.rollback_config()

        click.echo(f"Rollback completed (reload #{manager.reload_count})")

    except Exception as e:
        if "No snapshots available" in str(e):
            click.echo("No configuration snapshots available for rollback", err=True)
        else:
            click.echo(f"Rollback failed: {e}", err=True)


def handle_clear_command(manager: ConfigManager) -> None:
    """Handle the clear snapshots command."""
    manager.clear_snapshots()
    click.echo("Configuration snapshots cleared")


def handle_snapshots_command(manager: ConfigManager) -> None:
    """Handle the snapshots command."""
    snapshots = manager.get_snapshots()
    if not snapshots:
        click.echo("No configuration snapshots available")
        return

    click.echo(f"Configuration snapshots ({len(snapshots)} total):")
    for i, snapshot in enumerate(snapshots):
        marker = "*" if i == 0 else " "
        click.echo(f"{marker} {i+1}. {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        if snapshot.overrides:
            click.echo(f"     Overrides: {len(snapshot.overrides)} applied")


def handle_info_command(manager: ConfigManager) -> None:
    """Handle the info command."""
    config = manager.current_config
    overrides = manager.config_overrides

    click.echo("Current configuration info:")
    click.echo(f"   Config file: {manager.config_file}")
    click.echo(f"   Reload count: {manager.reload_count}")
    click.echo(f"   Active overrides: {len(overrides)}")

    # Show configuration summary
    config_dict = config.model_dump()

    click.echo("Configuration summary:")
    if 'llms' in config_dict:
        llm_count = len(config_dict['llms'])
        click.echo(f"   LLMs: {llm_count} configured")

    if 'tools' in config_dict:
        tool_count = len(config_dict['tools'])
        click.echo(f"   Tools: {tool_count} configured")

    if 'workflows' in config_dict:
        workflow_count = len(config_dict['workflows'])
        click.echo(f"   Workflows: {workflow_count} configured")

    front_end_type = config_dict.get('general', {}).get('front_end', {}).get('_type', 'unknown')
    click.echo(f"   Front end: {front_end_type}")

    if overrides:
        click.echo("Active overrides:")
        for key, value in overrides.items():
            click.echo(f"   {key}: {value}")


if __name__ == '__main__':
    cli()
