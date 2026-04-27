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
"""CLI-side telemetry plumbing.

This module exists so that telemetry concerns stay out of `entrypoint.py` and
`main.py`. The integration is a thin pair of helpers:

- :func:`record_invocation_start` — called by the Click group callback, stashes
  a session ID and timestamp on the Click context.
- :func:`emit_command_event` — called by the entry-point wrapper after the CLI
  exits (success, interrupt, or failure), constructs and enqueues a
  :class:`CliCommandEvent`.

All public functions swallow exceptions: telemetry must never disrupt the host
CLI invocation.
"""
from __future__ import annotations

import logging
import platform
import sys
import time
import uuid
from typing import TYPE_CHECKING

from nat.utils.telemetry import TELEMETRY_ENABLED
from nat.utils.telemetry import CliCommandEvent
from nat.utils.telemetry import NATTelemetryHandler
from nat.utils.telemetry import TaskStatusEnum

if TYPE_CHECKING:
    import click

logger = logging.getLogger(__name__)

_CTX_SESSION_ID = "telemetry_session_id"
_CTX_START_TIME = "telemetry_start_time"
_CTX_COMMAND = "telemetry_command"


def record_invocation_start(ctx: click.Context) -> None:
    """Stash the session ID, start time, and invoked command on ``ctx.obj``.

    Safe to call unconditionally; does nothing meaningful when telemetry is
    disabled, but keeping the bookkeeping unconditional makes the call sites
    simpler.
    """
    try:
        ctx_dict = ctx.ensure_object(dict)
        ctx_dict[_CTX_SESSION_ID] = uuid.uuid4().hex
        ctx_dict[_CTX_START_TIME] = time.monotonic()
        ctx_dict[_CTX_COMMAND] = ctx.invoked_subcommand
    except Exception:  # noqa: BLE001
        logger.debug("record_invocation_start failed", exc_info=True)


def emit_command_event(
    ctx_obj: dict | None,
    *,
    task_status: TaskStatusEnum,
    exit_code: int,
    error_class: str | None = None,
) -> None:
    """Build and enqueue a :class:`CliCommandEvent` for the just-finished call.

    Parameters
    ----------
    ctx_obj:
        The mutable dict that was used as Click's ``obj``. Reads back the
        identifiers stashed by :func:`record_invocation_start`. May be
        ``None`` or empty if Click never reached the group callback (e.g.
        argument parse error before any subcommand resolved); we still emit
        an event in that case with ``command="unknown"``.
    task_status:
        Outcome of the invocation.
    exit_code:
        Process exit code we are about to return.
    error_class:
        Exception class name on failure (no message). ``None`` otherwise.
    """
    if not TELEMETRY_ENABLED:
        return

    try:
        ctx_obj = ctx_obj or {}
        session_id: str = ctx_obj.get(_CTX_SESSION_ID) or uuid.uuid4().hex
        start_time: float | None = ctx_obj.get(_CTX_START_TIME)
        command: str = ctx_obj.get(_CTX_COMMAND) or "unknown"
        subcommand = _resolve_subcommand()

        if start_time is not None:
            duration_ms = int((time.monotonic() - start_time) * 1000)
        else:
            duration_ms = -1

        event = CliCommandEvent(
            command=command,
            subcommand=subcommand,
            task_status=task_status,
            duration_ms=duration_ms,
            exit_code=exit_code,
            error_class=error_class,
            python_version=platform.python_version(),
        )

        with NATTelemetryHandler(session_id=session_id) as handler:
            handler.enqueue(event)
    except Exception:  # noqa: BLE001 - never let telemetry break the CLI
        logger.debug("emit_command_event failed", exc_info=True)


def _resolve_subcommand() -> str | None:
    """Best-effort recovery of the second-level command from ``sys.argv``.

    Click does not expose nested ``invoked_subcommand`` from the root context.
    We look at the post-program tokens and pick the first non-flag token after
    the top-level command, which is good enough for usage analytics. Returns
    ``None`` if we can't identify one.
    """
    try:
        argv = sys.argv[1:]
        # Skip top-level flags like --log-level INFO before the first command.
        i = 0
        while i < len(argv) and argv[i].startswith("-"):
            # Skip the option's value if present and non-flag.
            if "=" in argv[i] or i + 1 >= len(argv) or argv[i + 1].startswith("-"):
                i += 1
            else:
                i += 2
        # First non-flag token is the top-level command; the next non-flag is
        # the subcommand if any.
        if i >= len(argv):
            return None
        i += 1  # skip top-level command
        while i < len(argv) and argv[i].startswith("-"):
            if "=" in argv[i] or i + 1 >= len(argv) or argv[i + 1].startswith("-"):
                i += 1
            else:
                i += 2
        if i < len(argv) and not argv[i].startswith("-"):
            return argv[i]
        return None
    except Exception:  # noqa: BLE001
        return None
