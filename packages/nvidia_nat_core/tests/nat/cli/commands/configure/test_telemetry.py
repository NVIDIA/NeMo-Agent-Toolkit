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
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from nat.cli.commands.configure.telemetry import telemetry_command
from nat.utils.telemetry.consent import ConsentState
from nat.utils.telemetry.consent import read_persisted_consent
from nat.utils.telemetry.consent import write_persisted_consent


@pytest.fixture
def consent_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "telemetry.toml"
    monkeypatch.setenv("NAT_TELEMETRY_CONSENT_FILE", str(path))
    return path


def test_enable_persists_consent_as_enabled(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NAT_TELEMETRY_ENABLED", raising=False)
    runner = CliRunner()
    result = runner.invoke(telemetry_command, ["--enable"])
    assert result.exit_code == 0
    assert "Telemetry enabled" in result.output
    assert read_persisted_consent() == ConsentState.ENABLED


def test_disable_persists_consent_as_disabled(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NAT_TELEMETRY_ENABLED", raising=False)
    runner = CliRunner()
    result = runner.invoke(telemetry_command, ["--disable"])
    assert result.exit_code == 0
    assert "Telemetry disabled" in result.output
    assert read_persisted_consent() == ConsentState.DISABLED


def test_status_default_when_no_decision(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NAT_TELEMETRY_ENABLED", raising=False)
    runner = CliRunner()
    result = runner.invoke(telemetry_command, [])
    assert result.exit_code == 0
    assert "no decision recorded" in result.output
    # Hint about how to fix it
    assert "nat configure telemetry --enable" in result.output


def test_status_reports_persisted_enabled(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NAT_TELEMETRY_ENABLED", raising=False)
    write_persisted_consent(ConsentState.ENABLED)
    runner = CliRunner()
    result = runner.invoke(telemetry_command, ["--status"])
    assert result.exit_code == 0
    assert "Effective: enabled" in result.output


def test_status_reports_persisted_disabled(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NAT_TELEMETRY_ENABLED", raising=False)
    write_persisted_consent(ConsentState.DISABLED)
    runner = CliRunner()
    result = runner.invoke(telemetry_command, ["--status"])
    assert result.exit_code == 0
    assert "Effective: disabled" in result.output


def test_env_var_overrides_persisted_in_status(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    """If both the env var and the persisted file disagree, env var wins
    AND the user is told about the override."""
    write_persisted_consent(ConsentState.DISABLED)
    monkeypatch.setenv("NAT_TELEMETRY_ENABLED", "true")
    runner = CliRunner()
    result = runner.invoke(telemetry_command, [])
    assert result.exit_code == 0
    assert "Effective: enabled" in result.output
    assert "NAT_TELEMETRY_ENABLED" in result.output
    assert "overridden" in result.output


def test_enable_warns_when_env_var_will_override(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    """Even if the user runs --enable, an opposing env var still wins —
    we must warn them so they aren't surprised."""
    monkeypatch.setenv("NAT_TELEMETRY_ENABLED", "false")
    runner = CliRunner()
    result = runner.invoke(telemetry_command, ["--enable"])
    assert result.exit_code == 0
    assert read_persisted_consent() == ConsentState.ENABLED
    assert "NAT_TELEMETRY_ENABLED" in result.output
    assert "override" in result.output


def test_disable_fails_loudly_when_persistence_silently_drops(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    """If ``write_persisted_consent`` silently swallows a write failure
    (e.g. read-only filesystem), the configure subcommand must NOT
    falsely report success — that would leave a previously-enabled user
    confidently believing they opted out while the next invocation still
    emits."""
    from nat.cli.commands.configure import telemetry as telemetry_module

    def fake_write(_state):
        # Simulate a swallowed failure: pretend the write happened, but
        # the persisted state remains whatever was there before
        # (NEVER_ASKED here, since the consent_file fixture starts empty).
        pass

    monkeypatch.setattr(telemetry_module, "write_persisted_consent", fake_write)
    monkeypatch.delenv("NAT_TELEMETRY_ENABLED", raising=False)

    runner = CliRunner()
    result = runner.invoke(telemetry_command, ["--disable"])
    assert result.exit_code != 0, "Configure must fail when persistence didn't take effect"
    assert "Failed to persist telemetry consent" in result.output
    assert "your previous decision is unchanged" in result.output.lower()


def test_enable_fails_loudly_when_persistence_silently_drops(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    """Symmetric to the --disable case."""
    from nat.cli.commands.configure import telemetry as telemetry_module

    monkeypatch.setattr(telemetry_module, "write_persisted_consent", lambda _state: None)
    monkeypatch.delenv("NAT_TELEMETRY_ENABLED", raising=False)

    runner = CliRunner()
    result = runner.invoke(telemetry_command, ["--enable"])
    assert result.exit_code != 0
    assert "Failed to persist telemetry consent" in result.output
