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

import shutil
import subprocess
import sys

import click

from nat.cli.commands.start import start_command  # pylint: disable=import-error,no-name-in-module


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
