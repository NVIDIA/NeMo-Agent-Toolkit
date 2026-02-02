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
"""FastMCP CLI commands for NeMo Agent Toolkit."""

from __future__ import annotations

import shutil
import signal
import subprocess
import sys
from pathlib import Path

import click

from nat.cli.commands.start import start_command  # type: ignore[reportMissingImports]
from nat.plugins.fastmcp.cli.utils import iter_file_changes


@click.group(name=__name__, invoke_without_command=False, help="FastMCP-related commands.")
def fastmcp_command():
    """FastMCP-related commands."""
    return None


@fastmcp_command.group(name="server", invoke_without_command=False, help="FastMCP server commands.")
def fastmcp_server_command():
    """FastMCP server commands."""
    return None


def _run_fastmcp_cli(subcommand: list[str], extra_args: list[str]) -> None:
    """Run the upstream `fastmcp` CLI with passthrough arguments.

    Args:
        subcommand: The `fastmcp` subcommand chain to invoke.
        extra_args: Additional CLI arguments to forward.
    """
    fastmcp_exe = shutil.which("fastmcp")
    if fastmcp_exe:
        cmd = [fastmcp_exe, *subcommand, *extra_args]
    else:
        cmd = [sys.executable, "-m", "fastmcp", *subcommand, *extra_args]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise click.ClickException(f"`fastmcp {' '.join(subcommand)}` failed with exit code {result.returncode}")


def _resolve_nat_cli_command() -> list[str]:
    nat_exe = shutil.which("nat")
    if nat_exe:
        return [nat_exe]
    return [sys.executable, "-m", "nat"]


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@fastmcp_server_command.command(
    name="dev",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
    help="Run a FastMCP server in developer mode with auto-reload.",
)
@click.option(
    "--config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="A JSON/YAML file that sets the parameters for the workflow.",
)
@click.option(
    "--override",
    type=(str, str),
    multiple=True,
    help="Override config values using dot notation (e.g., --override llms.nim_llm.temperature 0.7)",
)
@click.option(
    "--reload/--no-reload",
    default=True,
    help="Enable auto-reload on changes (default: enabled).",
)
@click.option(
    "--watch-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    multiple=True,
    help="Additional paths to watch for changes (repeatable).",
)
@click.pass_context
def fastmcp_server_dev(
    ctx: click.Context,
    config_file: Path,
    override: tuple[tuple[str, str], ...],
    reload: bool,
    watch_path: tuple[Path, ...],
) -> None:
    """Developer-focused FastMCP server runner with reload support."""
    base_cmd = _resolve_nat_cli_command() + ["fastmcp", "serve", "--config_file", str(config_file)]
    for key, value in override:
        base_cmd.extend(["--override", key, value])
    if ctx.args:
        base_cmd.extend(ctx.args)

    def start_server() -> subprocess.Popen:
        return subprocess.Popen(base_cmd)

    if not reload:
        proc = start_server()
        proc.wait()
        if proc.returncode != 0:
            raise click.ClickException(f"FastMCP server exited with code {proc.returncode}")
        return

    watch_paths = {config_file}
    watch_paths.update(watch_path)

    proc = start_server()
    try:
        for _changes in iter_file_changes(watch_paths, debounce_ms=750):
            click.echo("Change detected. Restarting FastMCP server...")
            _stop_process(proc)
            proc = start_server()
    except KeyboardInterrupt:
        _stop_process(proc)


@fastmcp_server_command.command(
    name="install",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
    help="Install MCP servers in various clients and formats.",
)
@click.pass_context
def fastmcp_server_install(ctx: click.Context) -> None:
    """Install FastMCP assets via the upstream CLI."""
    _run_fastmcp_cli(["install"], list(ctx.args))


# nat fastmcp server run: reuse the start/fastmcp frontend command
fastmcp_server_command.add_command(start_command.get_command(None, "fastmcp"), name="run")  # type: ignore

# Optional alias for convenience: nat fastmcp serve
fastmcp_command.add_command(start_command.get_command(None, "fastmcp"), name="serve")  # type: ignore
