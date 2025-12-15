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

from nat.cli.cli_utils.validation import validate_url
from nat.cli.commands.start import start_command

logger = logging.getLogger(__name__)


@click.group(name=__name__, invoke_without_command=False, help="A2A-related commands.")
def a2a_command():
    """
    A2A-related commands.
    """
    return None


# nat a2a serve: reuses the start/a2a frontend command
a2a_command.add_command(start_command.get_command(None, "a2a"), name="serve")  # type: ignore

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


async def discover_agent(url: str, timeout: int = 30):
    """Discover A2A agent and fetch AgentCard.

    Args:
        url: A2A agent URL
        timeout: Timeout in seconds

    Returns:
        AgentCard object or None if failed
    """
    try:
        from datetime import timedelta

        from nat.plugins.a2a.client.client_base import A2ABaseClient

        # Create client
        client = A2ABaseClient(base_url=url, task_timeout=timedelta(seconds=timeout))

        async with client:
            agent_card = client.agent_card

            if not agent_card:
                raise RuntimeError(f"Failed to fetch agent card from {url}")

            return agent_card

    except ImportError:
        click.echo(
            "A2A client functionality requires nvidia-nat-a2a package. Install with: uv pip install nvidia-nat-a2a",
            err=True)
        return None


def format_agent_card_display(agent_card, verbose: bool = False):
    """Format AgentCard for display.

    Args:
        agent_card: AgentCard object
        verbose: Show full details
    """
    from rich.console import Console
    from rich.panel import Panel

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
@click.option('--url', required=True, callback=validate_url, help='A2A agent URL (e.g., http://localhost:9999)')
@click.option('--json-output', is_flag=True, help='Output AgentCard as JSON')
@click.option('--verbose', is_flag=True, help='Show full AgentCard details')
@click.option('--save', type=click.Path(), help='Save AgentCard to file')
@click.option('--timeout', default=30, show_default=True, help='Timeout in seconds')
def a2a_client_discover(url: str, json_output: bool, verbose: bool, save: str | None, timeout: int):
    """Discover A2A agent and display AgentCard information.

    Connects to an A2A agent at the specified URL and fetches its AgentCard,
    which contains information about the agent's capabilities, skills, and
    configuration requirements.

    Args:
        url: A2A agent URL (e.g., http://localhost:9999)
        json_output: Output as JSON instead of formatted display
        verbose: Show full details including all skill information
        save: Save AgentCard JSON to specified file
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
        agent_card = asyncio.run(discover_agent(url, timeout=timeout))
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


async def _create_bearer_token_auth(
    builder,
    bearer_token: str | None,
    bearer_token_env: str | None,
):
    """Create bearer token auth configuration for CLI usage."""
    import os

    from pydantic import SecretStr

    from nat.authentication.api_key.api_key_auth_provider_config import APIKeyAuthProviderConfig
    from nat.data_models.authentication import HeaderAuthScheme

    # Get token from env var or direct input
    if bearer_token_env:
        token_value = os.getenv(bearer_token_env)
        if not token_value:
            raise ValueError(f"Environment variable '{bearer_token_env}' not found or empty")
    elif bearer_token:
        token_value = bearer_token
    else:
        raise ValueError("No bearer token provided")

    # Create API key auth config with Bearer scheme
    auth_config = APIKeyAuthProviderConfig(
        raw_key=SecretStr(token_value),
        auth_scheme=HeaderAuthScheme.BEARER,
    )

    auth_provider_name = "bearer_token_cli"
    await builder.add_auth_provider(auth_provider_name, auth_config)
    return auth_provider_name


async def _load_auth_from_config(
    builder,
    config_path: str,
    auth_provider_name: str,
):
    """Load auth provider from auth-only config file.

    Parses only the authentication section from YAML file.
    No other workflow sections are required.
    """
    import yaml
    from pydantic import TypeAdapter

    from nat.cli.type_registry import GlobalTypeRegistry
    from nat.data_models.authentication import AuthProviderBaseConfig

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # Extract just the authentication section
    if 'authentication' not in config_data:
        raise ValueError("Config file must contain 'authentication' section")

    auth_configs = config_data['authentication']
    if auth_provider_name not in auth_configs:
        raise ValueError(f"Auth provider '{auth_provider_name}' not found in config")

    auth_config_dict = auth_configs[auth_provider_name]

    # Parse the dictionary into the proper AuthProviderBaseConfig subclass
    auth_union_type = GlobalTypeRegistry.get().compute_annotation(AuthProviderBaseConfig)
    auth_config = TypeAdapter(auth_union_type).validate_python(auth_config_dict)

    # Add the auth provider to builder
    await builder.add_auth_provider(auth_provider_name, auth_config)
    return auth_provider_name


async def _create_auth_from_json(
    builder,
    auth_json: str,
):
    """Create auth provider from inline JSON config."""
    from pydantic import TypeAdapter

    from nat.cli.type_registry import GlobalTypeRegistry
    from nat.data_models.authentication import AuthProviderBaseConfig

    auth_config_dict = json.loads(auth_json)

    if '_type' not in auth_config_dict:
        raise ValueError("Auth JSON must contain '_type' field")

    # Parse the dictionary into the proper AuthProviderBaseConfig subclass
    auth_union_type = GlobalTypeRegistry.get().compute_annotation(AuthProviderBaseConfig)
    auth_config = TypeAdapter(auth_union_type).validate_python(auth_config_dict)

    # Add the auth provider to builder
    auth_provider_name = "auth_json_cli"
    await builder.add_auth_provider(auth_provider_name, auth_config)
    return auth_provider_name


async def get_a2a_function_group(
    url: str,
    timeout: int = 30,
    auth_provider_name: str | None = None,
    user_id: str | None = None,
):
    """Load A2A client as a function group with optional authentication.

    Args:
        url: A2A agent URL
        timeout: Timeout in seconds
        auth_provider_name: Optional auth provider name (from builder)
        user_id: Optional user ID for authentication

    Returns:
        Tuple of (builder, group, functions dict) or (None, None, None) if failed
    """
    try:
        from datetime import timedelta

        from nat.builder.workflow_builder import WorkflowBuilder
        from nat.plugins.a2a.client.client_config import A2AClientConfig

        builder = WorkflowBuilder()
        await builder.__aenter__()

        # Create A2A config with optional auth
        config = A2AClientConfig(
            url=url,
            task_timeout=timedelta(seconds=timeout),
            auth_provider=auth_provider_name,
            default_user_id=user_id,
        )

        # Add function group
        group = await builder.add_function_group("a2a_client", config)

        # Get accessible functions
        fns = await group.get_accessible_functions()

        return builder, group, fns

    except ImportError:
        click.echo(
            "A2A client functionality requires nvidia-nat-a2a package. Install with: uv pip install nvidia-nat-a2a",
            err=True)
        return None, None, None
    except Exception as e:
        logger.error(f"Error loading A2A function group: {e}", exc_info=True)
        raise


def format_info_display(info: dict):
    """Format agent info for simple text display."""
    click.secho("Agent Information", fg='cyan', bold=True)
    click.echo(f"  Name:        {info.get('name', 'N/A')}")
    click.echo(f"  Version:     {info.get('version', 'N/A')}")
    click.echo(f"  URL:         {info.get('url', 'N/A')}")

    if info.get('description'):
        click.echo(f"  Description: {info['description']}")

    if info.get('provider'):
        provider = info['provider']
        if provider.get('name'):
            click.echo(f"  Provider:    {provider['name']}")

    caps = info.get('capabilities', {})
    streaming = "✓" if caps.get('streaming') else "✗"
    click.echo(f"  Streaming:   {streaming}")

    click.echo(f"  Skills:      {info.get('num_skills', 0)}")


def format_skills_display(skills_data: dict):
    """Format agent skills for simple text display."""
    agent_name = skills_data.get('agent', 'Unknown')
    skills = skills_data.get('skills', [])

    click.secho(f"Agent Skills ({len(skills)})", fg='cyan', bold=True)
    click.echo(f"  Agent: {agent_name}")
    click.echo()

    for i, skill in enumerate(skills, 1):
        click.secho(f"  [{i}] {skill['id']}", fg='yellow')
        if skill.get('name'):
            click.echo(f"      Name:        {skill['name']}")
        click.echo(f"      Description: {skill['description']}")

        if skill.get('examples'):
            examples = skill['examples']
            if len(examples) == 1:
                click.echo(f"      Example:     {examples[0]}")
            else:
                click.echo(f"      Examples:    {examples[0]}")
                if len(examples) > 1:
                    click.secho(f"                   (+{len(examples)-1} more)", fg='bright_black')

        if skill.get('tags'):
            click.echo(f"      Tags:        {', '.join(skill['tags'])}")

        if i < len(skills):
            click.echo()  # Blank line between skills


def format_call_response_display(message: str, response: str, elapsed: float):
    """Format agent call response for simple text display."""
    # Show query for context
    click.secho(f"Query: {message}", fg='cyan')
    click.echo()

    # Show response (main output)
    click.echo(response)

    # Show timing info in bright green to stderr
    click.echo()
    click.secho(f"({elapsed:.2f}s)", fg='bright_green', err=True)


@a2a_client_command.command(name="get_info", help="Get agent metadata and information.")
@click.option('--url', required=True, callback=validate_url, help='A2A agent URL (e.g., http://localhost:9999)')
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.option('--timeout', default=30, show_default=True, help='Timeout in seconds')
def a2a_client_get_info(url: str, json_output: bool, timeout: int):
    """Get agent metadata including name, version, provider, and capabilities.

    This command connects to an A2A agent and retrieves its metadata.

    Args:
        url: A2A agent URL (e.g., http://localhost:9999)
        json_output: Output as JSON instead of formatted display
        timeout: Timeout in seconds for agent connection

    Examples:
        nat a2a client get_info --url http://localhost:9999
        nat a2a client get_info --url http://localhost:9999 --json-output
    """

    async def run():
        builder = None
        try:
            # Load A2A function group
            builder, group, fns = await get_a2a_function_group(url, timeout=timeout)
            if not builder:
                return

            # Get the get_info function
            fn = fns.get("a2a_client.get_info")
            if not fn:
                click.echo("[ERROR] get_info function not found", err=True)
                return

            # Call the function
            info = await fn.acall_invoke()

            if json_output:
                click.echo(json.dumps(info, indent=2))
            else:
                format_info_display(info)

        except Exception as e:
            click.echo(f"[ERROR] {e}", err=True)
            logger.error(f"Error in get_info command: {e}", exc_info=True)
        finally:
            if builder:
                await builder.__aexit__(None, None, None)

    asyncio.run(run())


@a2a_client_command.command(name="get_skills", help="Get agent skills and capabilities.")
@click.option('--url', required=True, callback=validate_url, help='A2A agent URL (e.g., http://localhost:9999)')
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.option('--timeout', default=30, show_default=True, help='Timeout in seconds')
def a2a_client_get_skills(url: str, json_output: bool, timeout: int):
    """Get detailed list of agent skills and capabilities.

    This command connects to an A2A agent and retrieves all available skills
    with their descriptions, examples, and tags.

    Args:
        url: A2A agent URL (e.g., http://localhost:9999)
        json_output: Output as JSON instead of formatted display
        timeout: Timeout in seconds for agent connection

    Examples:
        nat a2a client get_skills --url http://localhost:9999
        nat a2a client get_skills --url http://localhost:9999 --json-output
    """

    async def run():
        builder = None
        try:
            # Load A2A function group
            builder, group, fns = await get_a2a_function_group(url, timeout=timeout)
            if not builder:
                return

            # Get the get_skills function
            fn = fns.get("a2a_client.get_skills")
            if not fn:
                click.echo("[ERROR] get_skills function not found", err=True)
                return

            # Call the function
            skills_data = await fn.acall_invoke()

            if json_output:
                click.echo(json.dumps(skills_data, indent=2))
            else:
                format_skills_display(skills_data)

        except Exception as e:
            click.echo(f"[ERROR] {e}", err=True)
            logger.error(f"Error in get_skills command: {e}", exc_info=True)
        finally:
            if builder:
                await builder.__aexit__(None, None, None)

    asyncio.run(run())


@a2a_client_command.command(name="call", help="Call the agent with a message.")
@click.option('--url', required=True, callback=validate_url, help='A2A agent URL (e.g., http://localhost:9999)')
@click.option('--message', required=True, help='Message to send to the agent')
@click.option('--task-id', help='Optional task ID for continuing a conversation')
@click.option('--context-id', help='Optional context ID for maintaining context')
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.option('--timeout', default=30, show_default=True, help='Timeout in seconds')
@click.option('--bearer-token', help='Bearer token for authentication')
@click.option('--bearer-token-env', help='Environment variable containing bearer token')
@click.option('--auth-config', type=click.Path(exists=True), help='Auth-only config file (YAML)')
@click.option('--auth-provider', help='Auth provider name from config')
@click.option('--auth-json', help='Inline auth provider config as JSON')
@click.option('--user-id', help='User ID for authentication (optional)')
def a2a_client_call(url: str,
                    message: str,
                    task_id: str | None,
                    context_id: str | None,
                    json_output: bool,
                    timeout: int,
                    bearer_token: str | None,
                    bearer_token_env: str | None,
                    auth_config: str | None,
                    auth_provider: str | None,
                    auth_json: str | None,
                    user_id: str | None):
    """Call an A2A agent with a message and get a response.

    This command connects to an A2A agent, sends a message, and displays the response.
    Use this for one-off queries or testing. For complex workflows with multiple agents
    and tools, create a NAT workflow instead.

    Authentication is optional. If the agent requires authentication, use one of:
    - --bearer-token or --bearer-token-env for simple token auth
    - --auth-config and --auth-provider for config-based auth
    - --auth-json for inline JSON auth configuration

    Args:
        url: A2A agent URL (e.g., http://localhost:9999)
        message: Message to send to the agent
        task_id: Optional task ID for continuing a conversation
        context_id: Optional context ID for maintaining context
        json_output: Output as JSON instead of formatted display
        timeout: Timeout in seconds for agent connection
        bearer_token: Bearer token for authentication
        bearer_token_env: Environment variable containing bearer token
        auth_config: Path to auth-only config file (YAML)
        auth_provider: Auth provider name from config
        auth_json: Inline auth provider config as JSON
        user_id: User ID for authentication

    Examples:
        # Public agent (no auth)
        nat a2a client call --url http://localhost:9999 --message "Hello"

        # Bearer token auth
        nat a2a client call --url http://localhost:9999 --message "Hello" --bearer-token "sk-abc123"

        # Config-based auth
        nat a2a client call --url http://localhost:9999 --message "Hello" \
            --auth-config auth.yml --auth-provider my_oauth --user-id alice

        # Inline JSON auth
        nat a2a client call --url http://localhost:9999 --message "Hello" \
            --auth-json '{"_type": "api_key", "raw_key": "sk-abc123", "auth_scheme": "Bearer"}'
    """

    async def run():
        # Set up authentication callback for CLI workflows
        # This is needed for A2A clients that authenticate during build
        try:
            from nat.builder.context import Context
            from nat.front_ends.console.authentication_flow_handler import ConsoleAuthenticationFlowHandler

            # Create and set the auth handler early so it's available during workflow building
            auth_handler = ConsoleAuthenticationFlowHandler()
            Context.get()._context_state.user_auth_callback.set(auth_handler.authenticate)
            logger.debug("CLI authentication callback registered for A2A client call")
        except ImportError:
            # Console auth handler not available, skip auth handler setup
            logger.debug("Console authentication handler not available, skipping CLI authentication callback setup")

        builder = None
        try:
            # Validate auth options
            auth_methods = sum([bool(bearer_token or bearer_token_env), bool(auth_config), bool(auth_json)])

            if auth_methods > 1:
                click.echo("[ERROR] Use only one authentication method", err=True)
                return

            if auth_provider and not auth_config:
                click.echo("[ERROR] --auth-provider requires --auth-config", err=True)
                return

            # Setup authentication if provided
            auth_provider_name = None
            if bearer_token or bearer_token_env:
                # Bearer token auth
                from nat.builder.workflow_builder import WorkflowBuilder
                builder = WorkflowBuilder()
                await builder.__aenter__()

                try:
                    auth_provider_name = await _create_bearer_token_auth(builder, bearer_token, bearer_token_env)
                except Exception as e:
                    click.echo(f"[ERROR] Failed to configure bearer token authentication: {e}", err=True)
                    return

            elif auth_config:
                # Config-based auth
                from nat.builder.workflow_builder import WorkflowBuilder
                builder = WorkflowBuilder()
                await builder.__aenter__()

                try:
                    if not auth_provider:
                        click.echo("[ERROR] --auth-provider is required with --auth-config", err=True)
                        return
                    auth_provider_name = await _load_auth_from_config(builder, auth_config, auth_provider)
                except Exception as e:
                    click.echo(f"[ERROR] Failed to load auth from config: {e}", err=True)
                    return

            elif auth_json:
                # Inline JSON auth
                from nat.builder.workflow_builder import WorkflowBuilder
                builder = WorkflowBuilder()
                await builder.__aenter__()

                try:
                    auth_provider_name = await _create_auth_from_json(builder, auth_json)
                except Exception as e:
                    click.echo(f"[ERROR] Failed to parse auth JSON: {e}", err=True)
                    return

            # Load A2A function group (with or without auth)
            start_time = time.time()

            if builder:
                # Auth was configured, use existing builder
                from datetime import timedelta

                from nat.plugins.a2a.client.client_config import A2AClientConfig

                config = A2AClientConfig(
                    url=url,
                    task_timeout=timedelta(seconds=timeout),
                    auth_provider=auth_provider_name,
                    default_user_id=user_id,
                )
                group = await builder.add_function_group("a2a_client", config)
                fns = await group.get_accessible_functions()
            else:
                # No auth, use helper function
                builder, group, fns = await get_a2a_function_group(url, timeout=timeout)
                if not builder:
                    return

            # Get the call function
            fn = fns.get("a2a_client.call")
            if not fn:
                click.echo("[ERROR] call function not found", err=True)
                return

            # Call the agent with the message
            response = await fn.acall_invoke(query=message, task_id=task_id, context_id=context_id)
            elapsed = time.time() - start_time

            if json_output:
                result = {"message": message, "response": response, "elapsed": elapsed}
                if task_id:
                    result["task_id"] = task_id
                if context_id:
                    result["context_id"] = context_id
                click.echo(json.dumps(result, indent=2))
            else:
                format_call_response_display(message, response, elapsed)

        except Exception as e:
            click.echo(f"[ERROR] {e}", err=True)
            logger.error(f"Error in call command: {e}", exc_info=True)
        finally:
            if builder:
                await builder.__aexit__(None, None, None)

    asyncio.run(run())
