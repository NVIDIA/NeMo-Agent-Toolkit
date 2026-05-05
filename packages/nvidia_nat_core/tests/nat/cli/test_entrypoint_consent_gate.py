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
"""Tests for the consent-prompt gate in ``nat.cli.entrypoint``.

The first-run telemetry consent prompt must NOT fire when the user is
running ``nat configure telemetry [--enable|--disable|--status]``. That
subcommand exists *to manage* the very decision the prompt asks about;
prompting first would (a) break the read-only contract of ``--status``
and (b) interleave a yes-default prompt with an explicit ``--disable``
request.
"""
from __future__ import annotations

import pytest

from nat.cli import entrypoint


@pytest.mark.parametrize(
    "argv,invoked,expected",
    [
        # Skip cases (prompt should be suppressed)
        (["nat", "configure", "telemetry", "--status"], "configure", True),
        (["nat", "configure", "telemetry", "--enable"], "configure", True),
        (["nat", "configure", "telemetry", "--disable"], "configure", True),
        (["nat", "configure", "telemetry"], "configure", True),
        # Root-level flags before "configure" don't fool the detection.
        (["nat", "--log-level", "DEBUG", "configure", "telemetry", "--status"], "configure", True),
        # Other subcommands of `nat configure` (hypothetical) must still prompt.
        (["nat", "configure", "something-else"], "configure", False),
        (["nat", "configure"], "configure", False),
        # Non-configure commands always prompt.
        (["nat", "run"], "run", False),
        (["nat", "info", "list-components"], "info", False),
        (["nat", "serve"], "serve", False),
        # Defensive: missing invoked_subcommand (e.g. --help short-circuit).
        (["nat", "--help"], None, False),
        # Defensive: argv missing "configure" but invoked_subcommand says
        # otherwise (extremely rare; click aliasing). Returns False, which
        # means we fall back to prompting — safe default.
        (["nat", "telemetry"], "configure", False),
    ])
def test_is_invoking_configure_telemetry(monkeypatch, argv, invoked, expected):
    monkeypatch.setattr("sys.argv", argv)
    assert entrypoint._is_invoking_configure_telemetry(invoked) is expected


def test_root_callback_skips_prompt_for_configure_telemetry(monkeypatch):
    """The root ``cli`` callback must not call ``maybe_prompt_for_consent``
    when the invoked path is ``nat configure telemetry``."""
    from click.testing import CliRunner

    monkeypatch.setattr("sys.argv", ["nat", "configure", "telemetry", "--status"])

    prompt_calls: list[bool] = []

    def fake_prompt():
        prompt_calls.append(True)

    monkeypatch.setattr(entrypoint, "maybe_prompt_for_consent", fake_prompt)

    runner = CliRunner()
    result = runner.invoke(entrypoint.cli, ["configure", "telemetry", "--status"])
    # The configure subcommand prints to stdout; we just need the prompt
    # not to have fired.
    assert result.exit_code == 0, result.output
    assert prompt_calls == []  # prompt SUPPRESSED for configure-telemetry


def test_root_callback_prompts_for_other_commands(monkeypatch):
    """Any non-configure-telemetry path must still trigger
    ``maybe_prompt_for_consent`` (whose own short-circuits then decide
    whether to actually display anything)."""
    from click.testing import CliRunner

    monkeypatch.setattr("sys.argv", ["nat", "info", "list-components"])

    prompt_calls: list[bool] = []

    def fake_prompt():
        prompt_calls.append(True)

    monkeypatch.setattr(entrypoint, "maybe_prompt_for_consent", fake_prompt)

    runner = CliRunner()
    runner.invoke(entrypoint.cli, ["info", "list-components"])
    assert prompt_calls == [True]  # prompt invoked (will short-circuit internally)
