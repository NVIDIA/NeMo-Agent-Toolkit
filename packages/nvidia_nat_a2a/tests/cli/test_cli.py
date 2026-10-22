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

import click
import pytest
from click.testing import CliRunner


def test_a2a_plugin_discovered():
    """Test that A2A plugin is discovered when nvidia-nat-a2a is installed."""
    import nat.plugins.a2a.cli.commands  # noqa: F401
    from nat.cli.plugin_loader import discover_and_load_cli_plugins

    cli_group = click.Group()
    discover_and_load_cli_plugins(cli_group)

    # A2A should be discovered and loaded
    assert "a2a" in cli_group.commands


@pytest.mark.parametrize("command", ["discover", "get_info", "get_skills", "call"])
def test_client_commands_exit_nonzero_when_agent_unreachable(command, monkeypatch):
    """A failed client command must exit non-zero so shell callers can detect it."""
    from nat.plugins.a2a.cli import commands

    async def _unreachable(url, *args, **kwargs):
        raise RuntimeError(f"Failed to fetch agent card from {url}")

    monkeypatch.setattr(commands, "discover_agent", _unreachable)
    monkeypatch.setattr(commands, "get_a2a_function_group", _unreachable)

    args = ["client", command, "--url", "http://localhost:1"]
    if command == "call":
        args += ["--message", "hi"]

    result = CliRunner().invoke(commands.a2a_command, args)

    assert result.exit_code == 1
    assert "Error: Failed to fetch agent card from http://localhost:1" in result.output


@pytest.mark.parametrize("extra_args, expected",
                         [(["--bearer-token", "x", "--auth-json", "{}"], "Error: Use only one authentication method"),
                          (["--auth-provider", "p"], "Error: --auth-provider requires --auth-config")])
def test_call_exits_nonzero_on_invalid_auth_options(extra_args, expected):
    from nat.plugins.a2a.cli import commands

    args = ["client", "call", "--url", "http://localhost:1", "--message", "hi", *extra_args]
    result = CliRunner().invoke(commands.a2a_command, args)

    assert result.exit_code == 1
    assert expected in result.output


def test_call_exits_nonzero_on_auth_config_without_auth_provider(tmp_path):
    from nat.plugins.a2a.cli import commands

    auth_config = tmp_path / "auth.yml"
    auth_config.write_text("authentication: {}\n")

    args = ["client", "call", "--url", "http://localhost:1", "--message", "hi", "--auth-config", str(auth_config)]
    result = CliRunner().invoke(commands.a2a_command, args)

    assert result.exit_code == 1
    assert "Error: --auth-provider is required with --auth-config" in result.output
