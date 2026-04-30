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

import click

from nat.utils.telemetry import CliCommandEvent
from nat.utils.telemetry import NATTelemetryHandler
from nat.utils.telemetry import TaskStatusEnum
from nat.utils.telemetry import config as _telemetry_config

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_CTX_SESSION_ID = "telemetry_session_id"
_CTX_START_TIME = "telemetry_start_time"
_CTX_COMMAND = "telemetry_command"
_CTX_ROOT_COMMAND = "telemetry_root_command"


def record_invocation_start(ctx: click.Context) -> None:
    """Stash the session ID, start time, and invoked command on ``ctx.obj``.

    Safe to call unconditionally; does nothing meaningful when telemetry is
    disabled, but keeping the bookkeeping unconditional makes the call sites
    simpler.

    Also stashes the root Click group so that :func:`_resolve_subcommand` can
    validate any second-level token against the set of *actually registered*
    subcommand names. This is the privacy boundary that prevents user-supplied
    positional arguments (file paths, workflow names, queries) from leaking
    into the telemetry payload.
    """
    try:
        ctx_dict = ctx.ensure_object(dict)
        ctx_dict[_CTX_SESSION_ID] = uuid.uuid4().hex
        ctx_dict[_CTX_START_TIME] = time.monotonic()
        ctx_dict[_CTX_COMMAND] = ctx.invoked_subcommand
        ctx_dict[_CTX_ROOT_COMMAND] = ctx.find_root().command
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
    if not _telemetry_config.TELEMETRY_ENABLED:
        return

    try:
        ctx_obj = ctx_obj or {}
        session_id: str = ctx_obj.get(_CTX_SESSION_ID) or uuid.uuid4().hex
        start_time: float | None = ctx_obj.get(_CTX_START_TIME)
        command: str = ctx_obj.get(_CTX_COMMAND) or "unknown"
        root_command = ctx_obj.get(_CTX_ROOT_COMMAND)
        subcommand = _resolve_subcommand(root_command, command)

        if start_time is not None:
            duration_ms = int((time.monotonic() - start_time) * 1000)
        else:
            duration_ms = -1

        # The schema requires non-nullable strings; fall back to "undefined"
        # (the schema-wide sentinel) when we don't have a concrete value.
        event = CliCommandEvent(
            command=command,
            subcommand=subcommand or "undefined",
            task_status=task_status,
            duration_ms=duration_ms,
            exit_code=exit_code,
            error_class=error_class or "undefined",
            python_version=platform.python_version(),
        )

        with NATTelemetryHandler(session_id=session_id) as handler:
            handler.enqueue(event)
    except Exception:  # noqa: BLE001 - never let telemetry break the CLI
        logger.debug("emit_command_event failed", exc_info=True)


def _resolve_subcommand(root_command: object | None, top_level: str) -> str | None:
    """Recover the second-level command name, validated against the registered
    Click command tree.

    Click does not expose nested ``invoked_subcommand`` from the root context,
    so we scan ``sys.argv`` for a candidate token. Crucially, we only return a
    token if it matches the name of an *actually registered* subcommand of the
    top-level group. This is the privacy boundary: positional arguments
    (file paths, workflow names, free-form queries) cannot match a registered
    subcommand name and therefore cannot leak.

    Returns ``None`` when:

    - The root command isn't a Click group (callable from non-CLI contexts).
    - ``top_level`` isn't a registered subcommand of the root group.
    - The resolved top-level command is itself a leaf (not a group), so no
      second level is possible.
    - No argv token matches a registered subcommand of that group.
    """
    try:
        if not isinstance(root_command, click.Group):
            return None
        sub = root_command.commands.get(top_level)
        if not isinstance(sub, click.Group):
            return None
        registered: set[str] = set(sub.commands.keys())
        for token in sys.argv[1:]:
            if token in registered:
                return token
        return None
    except Exception:  # noqa: BLE001
        return None
