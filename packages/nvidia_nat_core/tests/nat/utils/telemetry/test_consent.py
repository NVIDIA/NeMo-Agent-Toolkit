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

from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from nat.utils.telemetry import consent
from nat.utils.telemetry.consent import ConsentState


@pytest.fixture
def consent_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the consent module at a tempdir-scoped file."""
    path = tmp_path / "telemetry.toml"
    monkeypatch.setenv("NAT_TELEMETRY_CONSENT_FILE", str(path))
    return path


# ----------------------------------------------------------------- persistence


def test_read_persisted_consent_returns_never_asked_when_file_missing(consent_file: Path):
    assert consent.read_persisted_consent() == ConsentState.NEVER_ASKED


def test_write_then_read_roundtrip_enabled(consent_file: Path):
    consent.write_persisted_consent(ConsentState.ENABLED)
    assert consent_file.exists()
    assert consent.read_persisted_consent() == ConsentState.ENABLED


def test_write_then_read_roundtrip_disabled(consent_file: Path):
    consent.write_persisted_consent(ConsentState.DISABLED)
    assert consent.read_persisted_consent() == ConsentState.DISABLED


def test_write_never_asked_is_noop(consent_file: Path):
    consent.write_persisted_consent(ConsentState.NEVER_ASKED)
    assert not consent_file.exists()


def test_persisted_file_content_includes_metadata(consent_file: Path):
    consent.write_persisted_consent(ConsentState.ENABLED)
    text = consent_file.read_text()
    assert 'consent = "enabled"' in text
    assert "consented_at = " in text
    assert f'prompt_version = "{consent.PROMPT_VERSION}"' in text


def test_read_handles_malformed_file(consent_file: Path):
    consent_file.write_text("this is not valid toml [[[")
    assert consent.read_persisted_consent() == ConsentState.NEVER_ASKED


def test_read_handles_unrecognized_consent_value(consent_file: Path):
    consent_file.write_text('[telemetry]\nconsent = "maybe"\n')
    assert consent.read_persisted_consent() == ConsentState.NEVER_ASKED


def test_write_failure_is_silent(monkeypatch: pytest.MonkeyPatch):
    """A read-only path should not crash the CLI on write."""
    monkeypatch.setenv("NAT_TELEMETRY_CONSENT_FILE", "/proc/never-writable/telemetry.toml")
    # Should not raise
    consent.write_persisted_consent(ConsentState.ENABLED)


# ---------------------------------------------------------------- TTY detection


def test_is_interactive_session_returns_false_when_stdout_not_tty():
    with patch("sys.stdout") as stdout, patch("sys.stdin"):
        stdout.isatty.return_value = False
        assert consent.is_interactive_session() is False


def test_is_interactive_session_returns_false_when_stdin_not_tty():
    with patch("sys.stdout") as stdout, patch("sys.stdin") as stdin:
        stdout.isatty.return_value = True
        stdin.isatty.return_value = False
        assert consent.is_interactive_session() is False


def test_is_interactive_session_returns_true_only_when_both_ttys():
    with patch("sys.stdout") as stdout, patch("sys.stdin") as stdin:
        stdout.isatty.return_value = True
        stdin.isatty.return_value = True
        assert consent.is_interactive_session() is True


# --------------------------------------------------------------- initial state


def test_resolve_initial_consent_env_var_overrides_persisted(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    """Env var beats persisted file."""
    consent.write_persisted_consent(ConsentState.DISABLED)
    monkeypatch.setenv("NAT_TELEMETRY_ENABLED", "true")
    assert consent.resolve_initial_consent() is True


def test_resolve_initial_consent_uses_persisted_when_no_env(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NAT_TELEMETRY_ENABLED", raising=False)
    consent.write_persisted_consent(ConsentState.ENABLED)
    assert consent.resolve_initial_consent() is True


def test_resolve_initial_consent_defaults_off_with_no_signal(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    """No env var, no persisted file → default OFF (not opted-in until prompted)."""
    monkeypatch.delenv("NAT_TELEMETRY_ENABLED", raising=False)
    assert consent.resolve_initial_consent() is False


# ---------------------------------------------------------- maybe_prompt logic


def test_maybe_prompt_short_circuits_on_env_var(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    """Env var means user already chose; no prompt."""
    monkeypatch.setenv("NAT_TELEMETRY_ENABLED", "false")
    with patch.object(consent, "prompt_user") as mock_prompt:
        consent.maybe_prompt_for_consent()
        mock_prompt.assert_not_called()


def test_maybe_prompt_short_circuits_on_persisted_decision(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NAT_TELEMETRY_ENABLED", raising=False)
    consent.write_persisted_consent(ConsentState.ENABLED)
    with patch.object(consent, "prompt_user") as mock_prompt:
        consent.maybe_prompt_for_consent()
        mock_prompt.assert_not_called()


def test_maybe_prompt_short_circuits_on_non_interactive(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    """Headless context should never prompt — and telemetry stays off."""
    monkeypatch.delenv("NAT_TELEMETRY_ENABLED", raising=False)
    with patch.object(consent, "is_interactive_session", return_value=False), \
         patch.object(consent, "prompt_user") as mock_prompt:
        consent.maybe_prompt_for_consent()
        mock_prompt.assert_not_called()


def test_maybe_prompt_runs_prompt_when_interactive_and_undecided(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NAT_TELEMETRY_ENABLED", raising=False)
    with patch.object(consent, "is_interactive_session", return_value=True), \
         patch.object(consent, "prompt_user", return_value=ConsentState.ENABLED) as mock_prompt:
        consent.maybe_prompt_for_consent()
        mock_prompt.assert_called_once()
    # User said yes — persisted to file
    assert consent.read_persisted_consent() == ConsentState.ENABLED


def test_maybe_prompt_updates_live_telemetry_enabled_flag(consent_file: Path, monkeypatch: pytest.MonkeyPatch):
    """Critical: after the prompt, the rest of this same invocation must
    honor the user's decision without a process restart."""
    from nat.utils.telemetry import config as telemetry_config

    monkeypatch.delenv("NAT_TELEMETRY_ENABLED", raising=False)

    # Force the initial state to disabled so we can observe the flip
    original_value = telemetry_config.TELEMETRY_ENABLED
    telemetry_config.TELEMETRY_ENABLED = False
    try:
        with patch.object(consent, "is_interactive_session", return_value=True), \
             patch.object(consent, "prompt_user", return_value=ConsentState.ENABLED):
            consent.maybe_prompt_for_consent()
        assert telemetry_config.TELEMETRY_ENABLED is True
    finally:
        telemetry_config.TELEMETRY_ENABLED = original_value


# -------------------------------------------------------------- prompt_user


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("y", ConsentState.ENABLED),
        ("Y", ConsentState.ENABLED),
        ("yes", ConsentState.ENABLED),
        ("YES", ConsentState.ENABLED),
        ("", ConsentState.ENABLED),  # default yes on enter ([Y/n])
        ("   ", ConsentState.ENABLED),  # whitespace-only line is also "just Enter"
        ("n", ConsentState.DISABLED),
        ("N", ConsentState.DISABLED),
        ("no", ConsentState.DISABLED),
        ("NO", ConsentState.DISABLED),
        ("garbage", ConsentState.DISABLED),
    ])
def test_prompt_user_interprets_answers(answer: str, expected: ConsentState):
    with patch("builtins.input", return_value=answer), \
         patch("sys.stderr", new_callable=StringIO):
        assert consent.prompt_user() == expected


def test_prompt_user_handles_eof_as_disabled():
    with patch("builtins.input", side_effect=EOFError), \
         patch("sys.stderr", new_callable=StringIO):
        assert consent.prompt_user() == ConsentState.DISABLED


def test_prompt_user_handles_keyboard_interrupt_as_disabled():
    with patch("builtins.input", side_effect=KeyboardInterrupt), \
         patch("sys.stderr", new_callable=StringIO):
        assert consent.prompt_user() == ConsentState.DISABLED


def test_prompt_text_lists_what_is_collected_and_not_collected():
    """Privacy contract: the prompt explicitly tells the user both what
    we collect and what we do NOT collect. If you change this, expect a
    privacy review."""
    text = consent.render_prompt()
    assert "We collect" in text
    assert "Command name" in text
    assert "duration" in text
    assert "Python version" in text
    assert "We do NOT collect" in text
    assert "Command arguments" in text
    assert "user-supplied" in text or "user input" in text
    assert "NAT_TELEMETRY_ENABLED" in text
    assert "nat configure telemetry" in text


def test_prompt_text_uses_default_yes_bracket():
    """Pressing Enter must accept (default-yes). The visual ``[Y/n]`` cue
    has to match the parsing in ``prompt_user``; this test ties them
    together so a future flip back to default-no is impossible to do
    halfway."""
    assert "[Y/n]" in consent.render_prompt()
