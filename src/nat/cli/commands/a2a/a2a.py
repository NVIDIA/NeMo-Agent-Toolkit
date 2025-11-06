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

import asyncio
import json
import logging
import time

import click

logger = logging.getLogger(__name__)


@click.group(name=__name__, invoke_without_command=False, help="A2A-related commands.")
def a2a_command():
    """
    A2A-related commands.
    """
    return None


# Suppress verbose logs from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)


@a2a_command.group(name="client", invoke_without_command=False, help="A2A client commands.")
def a2a_client_command():
    """
    A2A client commands.
    """
    try:
        from nat.runtime.loader import PluginTypes
        from nat.runtime.loader import discover_and_register_plugins
        discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)
    except ImportError:
        click.echo("[WARNING] A2A client functionality requires nvidia-nat-a2a package.", err=True)
        pass


async def discover_agent(url: str, timeout: int = 30, extended: bool = False, auth_token: str | None = None):
    """Discover A2A agent and fetch AgentCard.

    Args:
        url: A2A agent URL
        timeout: Timeout in seconds
        extended: Fetch authenticated extended card
        auth_token: Auth token for protected agents

    Returns:
        AgentCard object or None if failed
    """
    try:
        from datetime import timedelta

        import httpx

        from nat.plugins.a2a.client_base import A2ABaseClient

        # Create client
        client = A2ABaseClient(base_url=url, task_timeout=timedelta(seconds=timeout))

        async with client:
            agent_card = client.agent_card

            if not agent_card:
                raise RuntimeError(f"Failed to fetch agent card from {url}")

            # TODO: Handle extended card fetch when auth is implemented
            if extended:
                click.echo("[WARNING] Extended card fetch not yet implemented", err=True)

            return agent_card

    except ImportError:
        click.echo(
            "A2A client functionality requires nvidia-nat-a2a package. Install with: uv pip install nvidia-nat-a2a",
            err=True)
        return None
    except Exception as e:
        logger.error(f"Error discovering agent: {e}", exc_info=True)
        raise


def format_agent_card_display(agent_card, verbose: bool = False):
    """Format AgentCard for display.

    Args:
        agent_card: AgentCard object
        verbose: Show full details
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Build content
    content = []

    # Basic info
    content.append(f"[bold]Name:[/bold] {agent_card.name}")
    content.append(f"[bold]Version:[/bold] {agent_card.version}")
    content.append(f"[bold]Protocol Version:[/bold] {agent_card.protocol_version}")
    content.append(f"[bold]URL:[/bold] {agent_card.url}")

    # Transport
    transport = agent_card.preferred_transport or "JSONRPC"
    content.append(f"[bold]Transport:[/bold] {transport} (preferred)")

    # Description
    if agent_card.description:
        content.append(f"[bold]Description:[/bold] {agent_card.description}")

    content.append("")  # Blank line

    # Capabilities
    content.append("[bold]Capabilities:[/bold]")
    caps = agent_card.capabilities
    if caps:
        streaming = "✓" if caps.streaming else "✗"
        content.append(f"  {streaming} Streaming")
        push = "✓" if caps.push_notifications else "✗"
        content.append(f"  {push} Push Notifications")
    else:
        content.append("  None specified")

    content.append("")  # Blank line

    # Skills
    skills = agent_card.skills
    content.append(f"[bold]Skills:[/bold] ({len(skills)})")

    for skill in skills:
        content.append(f"  • [cyan]{skill.id}[/cyan]")
        if skill.name:
            content.append(f"    Name: {skill.name}")
        content.append(f"    Description: {skill.description}")
        if skill.examples:
            if verbose:
                content.append(f"    Examples: {', '.join(repr(e) for e in skill.examples)}")
            else:
                # Show first example in normal mode
                content.append(f"    Example: {repr(skill.examples[0])}")
        if skill.tags:
            content.append(f"    Tags: {', '.join(skill.tags)}")

    content.append("")  # Blank line

    # Input/Output modes
    content.append(f"[bold]Input Modes:[/bold]  {', '.join(agent_card.default_input_modes)}")
    content.append(f"[bold]Output Modes:[/bold] {', '.join(agent_card.default_output_modes)}")

    content.append("")  # Blank line

    # Auth
    if agent_card.security or agent_card.security_schemes:
        content.append("[bold]Auth Required:[/bold] Yes")
        if verbose and agent_card.security_schemes:
            content.append(f"  Schemes: {', '.join(agent_card.security_schemes.keys())}")
    else:
        content.append("[bold]Auth Required:[/bold] None (public agent)")

    # Create panel
    panel = Panel("\n".join(content), title="[bold]Agent Card Discovery[/bold]", border_style="blue", padding=(1, 2))

    console.print(panel)


@a2a_client_command.command(name="discover", help="Discover A2A agent and display AgentCard information.")
@click.option('--url', required=True, help='A2A agent URL (e.g., http://localhost:9999)')
@click.option('--json-output', is_flag=True, help='Output AgentCard as JSON')
@click.option('--verbose', is_flag=True, help='Show full AgentCard details')
@click.option('--save', type=click.Path(), help='Save AgentCard to file')
@click.option('--extended', is_flag=True, help='Fetch authenticated extended card (requires auth)')
@click.option('--auth-token', help='Auth token for protected agents')
@click.option('--timeout', default=30, show_default=True, help='Timeout in seconds')
def a2a_client_discover(url: str,
                        json_output: bool,
                        verbose: bool,
                        save: str | None,
                        extended: bool,
                        auth_token: str | None,
                        timeout: int):
    """Discover A2A agent and display AgentCard information.

    Connects to an A2A agent at the specified URL and fetches its AgentCard,
    which contains information about the agent's capabilities, skills, and
    configuration requirements.

    Args:
        url: A2A agent URL (e.g., http://localhost:9999)
        json_output: Output as JSON instead of formatted display
        verbose: Show full details including all skill information
        save: Save AgentCard JSON to specified file
        extended: Fetch authenticated extended card (not yet implemented)
        auth_token: Auth token for protected agents (not yet implemented)
        timeout: Timeout in seconds for agent connection

    Examples:
        nat a2a client discover --url http://localhost:9999
        nat a2a client discover --url http://localhost:9999 --json-output
        nat a2a client discover --url http://localhost:9999 --verbose
        nat a2a client discover --url http://localhost:9999 --save agent-card.json
    """
    try:
        # Discover agent
        start_time = time.time()
        agent_card = asyncio.run(discover_agent(url, timeout=timeout, extended=extended, auth_token=auth_token))
        elapsed = time.time() - start_time

        if not agent_card:
            click.echo(f"[ERROR] Failed to discover agent at {url}", err=True)
            return

        # JSON output
        if json_output:
            output = agent_card.model_dump_json(indent=2)
            click.echo(output)

            # Save if requested
            if save:
                with open(save, 'w') as f:
                    f.write(output)
                click.echo(f"\n[INFO] Saved to {save}", err=False)

        else:
            # Rich formatted output
            format_agent_card_display(agent_card, verbose=verbose)

            # Save if requested
            if save:
                with open(save, 'w') as f:
                    f.write(agent_card.model_dump_json(indent=2))
                click.echo(f"\n✓ Saved AgentCard to {save}")

            click.echo(f"\n✓ Discovery completed in {elapsed:.2f}s")

    except Exception as e:
        click.echo(f"[ERROR] {e}", err=True)
        logger.error(f"Error in discover command: {e}", exc_info=True)
