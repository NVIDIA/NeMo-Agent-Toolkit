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
"""``nat configure telemetry`` — manage the user's telemetry consent decision.

Three modes:

- ``--enable``: persist consent as enabled.
- ``--disable``: persist consent as disabled.
- ``--status`` (default if no flag): print the current effective state and
  the source that determined it (env var, persisted file, or default).
"""
from __future__ import annotations

import os

import click

from nat.utils.telemetry.consent import ConsentState
from nat.utils.telemetry.consent import _consent_file_path
from nat.utils.telemetry.consent import read_persisted_consent
from nat.utils.telemetry.consent import write_persisted_consent


@click.command(name="telemetry", help="Enable, disable, or inspect NAT telemetry consent.")
@click.option("--enable", "action", flag_value="enable", help="Enable anonymous telemetry.")
@click.option("--disable", "action", flag_value="disable", help="Disable anonymous telemetry.")
@click.option("--status", "action", flag_value="status", default=True, help="Show current state (default).")
def telemetry_command(action: str) -> None:
    """Manage NAT telemetry consent."""
    if action == "enable":
        _persist_and_verify(ConsentState.ENABLED)
        click.echo(f"Telemetry enabled. Persisted to {_consent_file_path()}.")
        _warn_on_env_var_override()
        return
    if action == "disable":
        _persist_and_verify(ConsentState.DISABLED)
        click.echo(f"Telemetry disabled. Persisted to {_consent_file_path()}.")
        _warn_on_env_var_override()
        return

    # Status (default)
    env_value = os.getenv("NAT_TELEMETRY_ENABLED")
    persisted = read_persisted_consent()
    if env_value is not None:
        active = env_value.strip().lower() in ("1", "true", "yes")
        click.echo(f"Effective: {'enabled' if active else 'disabled'} (source: NAT_TELEMETRY_ENABLED={env_value!r})")
        if persisted != ConsentState.NEVER_ASKED:
            click.echo(f"  Persisted decision ({persisted.value}) is being overridden by the env var.")
        return
    if persisted == ConsentState.ENABLED:
        click.echo(f"Effective: enabled (source: {_consent_file_path()})")
        return
    if persisted == ConsentState.DISABLED:
        click.echo(f"Effective: disabled (source: {_consent_file_path()})")
        return
    click.echo("Effective: disabled (source: default — no decision recorded yet)")
    click.echo("Run 'nat configure telemetry --enable' or '--disable' to record a decision,")
    click.echo("or any other 'nat' command interactively to be prompted.")


def _persist_and_verify(state: ConsentState) -> None:
    """Persist ``state`` and read it back to confirm it actually landed.

    ``write_persisted_consent`` swallows write failures by design (the
    interactive consent flow can tolerate a re-prompt next run). The
    explicit ``nat configure telemetry --enable | --disable`` path
    cannot tolerate that: a failed write would leave the user
    confidently believing they opted out while the next invocation
    still emits. Verify the readback and surface a hard error if the
    state did not land.

    A readback mismatch usually means filesystem permission issues, a
    full disk, or an env var overriding the persisted decision (we
    don't fail on env override — that's reported separately by
    ``_warn_on_env_var_override``; we only fail on outright persistence
    failure).
    """
    write_persisted_consent(state)
    actual = read_persisted_consent()
    if actual != state:
        raise click.ClickException(f"Failed to persist telemetry consent to {_consent_file_path()}. "
                                   f"Expected {state.value!r}, file reads {actual.value!r}. "
                                   "Check filesystem permissions and disk space; your previous "
                                   "decision is unchanged.")


def _warn_on_env_var_override() -> None:
    """If NAT_TELEMETRY_ENABLED is set, the persisted decision is overridden.

    Tell the user so they aren't surprised when their next `nat run`
    doesn't behave as they just configured.
    """
    env_value = os.getenv("NAT_TELEMETRY_ENABLED")
    if env_value is not None:
        click.echo(
            f"Note: NAT_TELEMETRY_ENABLED is set to {env_value!r} in your environment "
            "and will override the persisted decision until unset.",
            err=True,
        )
