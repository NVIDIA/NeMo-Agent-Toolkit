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
"""First-run consent prompt for NAT telemetry.

Order of precedence for whether telemetry is active:

1. ``NAT_TELEMETRY_ENABLED`` environment variable, if set (any value).
2. Persisted consent file at ``~/.config/nat/telemetry.toml``.
3. Interactive prompt — only if both stdin and stdout are TTYs.
4. Default OFF, in non-interactive contexts (CI, cron, daemons).

The prompt is shown at most once per machine: the user's answer is
persisted, and subsequent invocations use the persisted value silently.
The prompt explicitly tells the user what is collected, what is not
collected, and how to change their decision later.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import UTC
from datetime import datetime
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

# Override path used by tests; production callers should not set this.
_CONSENT_FILE_ENV_VAR = "NAT_TELEMETRY_CONSENT_FILE"

PROMPT_VERSION = "1.0"
"""Bump if we materially change the prompt language. The persisted decision
records which prompt version the user saw, so we can re-prompt on
substantive changes (e.g. new categories of data collected)."""

_TRUTHY = ("1", "true", "yes")


class ConsentState(StrEnum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    NEVER_ASKED = "never_asked"


def _consent_file_path() -> Path:
    """Resolve the consent file location.

    Honors ``NAT_TELEMETRY_CONSENT_FILE`` for tests; falls back to
    ``~/.config/nat/telemetry.toml`` for production use.
    """
    override = os.getenv(_CONSENT_FILE_ENV_VAR)
    if override:
        return Path(override)
    return Path.home() / ".config" / "nat" / "telemetry.toml"


def read_persisted_consent() -> ConsentState:
    """Read the user's persisted consent decision, if any.

    Returns ``ConsentState.NEVER_ASKED`` if the file is missing, malformed,
    or contains an unrecognized value.
    """
    path = _consent_file_path()
    if not path.exists():
        return ConsentState.NEVER_ASKED
    try:
        import tomllib
        with path.open("rb") as f:
            data = tomllib.load(f)
        consent = data.get("telemetry", {}).get("consent")
        if consent in (ConsentState.ENABLED.value, ConsentState.DISABLED.value):
            return ConsentState(consent)
    except Exception:  # noqa: BLE001 - defensive; never let consent reading break the CLI
        logger.debug("Failed to read consent file at %s", path, exc_info=True)
    return ConsentState.NEVER_ASKED


def write_persisted_consent(state: ConsentState) -> None:
    """Persist the user's consent decision.

    Writes a small TOML file at the resolved consent path. Silently ignores
    write failures (filesystem permission errors, full disk, etc.) — the
    next interactive run will simply re-prompt.
    """
    if state == ConsentState.NEVER_ASKED:
        return
    path = _consent_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        content = ("# NeMo Agent Toolkit telemetry consent.\n"
                   "# To change, run `nat configure telemetry --enable | --disable`,\n"
                   "# or set the NAT_TELEMETRY_ENABLED environment variable.\n"
                   "[telemetry]\n"
                   f'consent = "{state.value}"\n'
                   f'consented_at = "{timestamp}"\n'
                   f'prompt_version = "{PROMPT_VERSION}"\n')
        path.write_text(content)
    except Exception:  # noqa: BLE001 - defensive; never let consent writing break the CLI
        logger.debug("Failed to write consent file at %s", path, exc_info=True)


def is_interactive_session() -> bool:
    """Whether we should attempt to prompt the user.

    Requires both stdin and stdout to be TTYs. Pipes, redirects, CI
    runners, daemons, and other headless contexts return False — in those
    cases we never prompt, and telemetry defaults to OFF.
    """
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:  # noqa: BLE001
        return False


def render_prompt() -> str:
    """The user-facing consent prompt.

    Kept inline (not in a separate file) so tests can assert on its
    contents and reviewers see any wording change in PR diffs.
    """
    return ("\n"
            "===========================================================\n"
            "  NeMo Agent Toolkit — anonymous telemetry\n"
            "===========================================================\n"
            "  We collect aggregate, anonymous CLI usage data to help us\n"
            "  prioritize features and fix bugs.\n"
            "\n"
            "  We collect:\n"
            "    - Command name (e.g. 'run', 'serve', 'evaluate')\n"
            "    - Outcome (success / failure / interrupted) and duration\n"
            "    - Exception class name on failure (no message)\n"
            "    - Python version, NAT version, CPU architecture\n"
            "\n"
            "  We do NOT collect:\n"
            "    - Command arguments, file paths, or config contents\n"
            "    - Workflow / function / tool / model names\n"
            "    - Hostnames, usernames, IP addresses, or any user input\n"
            "\n"
            "  Change anytime:\n"
            "    nat configure telemetry --enable | --disable | --status\n"
            "    or set NAT_TELEMETRY_ENABLED=true|false in your environment.\n"
            "===========================================================\n"
            "Allow anonymous telemetry? [Y/n]: ")


def prompt_user() -> ConsentState:
    """Display the consent prompt and read the user's answer.

    Returns ENABLED on an explicit ``y`` / ``yes`` or on an empty line
    (just pressing Enter, matching the ``[Y/n]`` default). Returns
    DISABLED on ``n`` / ``no``, on any other input, or on EOF /
    KeyboardInterrupt. The decision is always persisted by the caller,
    so a hostile interrupt is treated as "no thanks" rather than
    re-prompting indefinitely.
    """
    try:
        sys.stderr.write(render_prompt())
        sys.stderr.flush()
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return ConsentState.DISABLED
    if answer in ("", "y", "yes"):
        return ConsentState.ENABLED
    return ConsentState.DISABLED


def resolve_initial_consent() -> bool:
    """Resolve telemetry state at module import time, without prompting.

    Used by ``config.TELEMETRY_ENABLED`` so a process that imports the
    telemetry package without going through the CLI entrypoint (e.g. a
    library user) gets a sensible default. Order:

    1. ``NAT_TELEMETRY_ENABLED`` env var, if set.
    2. Persisted consent file.
    3. Default OFF — until the CLI prompt or env var resolves it.
    """
    env = os.getenv("NAT_TELEMETRY_ENABLED")
    if env is not None:
        return env.strip().lower() in _TRUTHY
    return read_persisted_consent() == ConsentState.ENABLED


def maybe_prompt_for_consent() -> None:
    """Run the first-run consent prompt if needed.

    Called by the CLI entrypoint group callback. No-op when:

    - ``NAT_TELEMETRY_ENABLED`` env var is set (the user opted via env).
    - A persisted consent decision already exists.
    - The session is non-interactive (no TTY on stdin or stdout).

    Otherwise: print the prompt, read the user's answer, persist the
    decision, and update the live ``TELEMETRY_ENABLED`` flag so the rest
    of this same invocation honors the choice.
    """
    if os.getenv("NAT_TELEMETRY_ENABLED") is not None:
        return
    if read_persisted_consent() != ConsentState.NEVER_ASKED:
        return
    if not is_interactive_session():
        return
    state = prompt_user()
    write_persisted_consent(state)
    # Update the live module-level flag so the rest of this invocation
    # respects the user's choice without waiting for a process restart.
    from nat.utils.telemetry import config as _config
    _config.TELEMETRY_ENABLED = state == ConsentState.ENABLED
