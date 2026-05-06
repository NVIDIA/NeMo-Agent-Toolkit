# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# The purpose of this function is to allow loading the current directory as a module. This allows relative imports and
# more specifically `..common` to function correctly
def run_cli() -> int | None:
    """Process entrypoint for the ``nat`` console script.

    Bootstraps ``sys.path`` so the ``nat`` package can be imported from a
    source checkout, then delegates to the Click ``cli`` group with
    ``standalone_mode=False`` so the wrapper sees real exceptions
    (``KeyboardInterrupt``, :class:`click.Abort`,
    :class:`click.ClickException`) rather than the generic
    :class:`SystemExit` that standalone mode produces. Each branch
    records a single ``nat_cli_command`` telemetry event via
    :func:`nat.cli.telemetry_hook.emit_command_event` before re-raising
    or calling :func:`sys.exit`.

    Side effects:
        - Sets ``TRANSFORMERS_VERBOSITY=error`` in the process environment.
        - Appends the ``packages/.../src`` parent directory to ``sys.path``.
        - Replicates Click's standalone-mode user-facing UX: prints the
          ``"Aborted!"`` line on Ctrl-C / :class:`click.Abort`; calls
          ``ClickException.show()`` on usage errors.
        - Emits exactly one telemetry event per invocation (success,
          failure, or interrupted), gated by the user's persisted
          consent decision and the ``NAT_TELEMETRY_ENABLED`` env var.

    Returns:
        On the **success path**, the integer exit code if the invoked
        Click callback returned an int (Click's convention for "exit
        with this code" without raising); otherwise ``None`` (treated
        as exit 0 by the console-script wrapper's ``sys.exit(...)``).
        On every **non-success path**, the function does not return —
        :class:`SystemExit` is re-raised after telemetry emission so
        the host process exits with the appropriate status code.

    Raises:
        SystemExit: Re-raised after a telemetry event is emitted, so the
            host process exits with the appropriate status code (0 on
            success / ``--help``, 1 on uncaught exception or
            :class:`click.Abort`, 2 on :class:`click.UsageError`, 130 on
            ``KeyboardInterrupt``, ``exc.exit_code`` for other
            :class:`click.ClickException` subclasses).
    """
    import os
    import sys

    # Suppress warnings from transformers
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    if (parent_dir not in sys.path):
        sys.path.append(parent_dir)

    import click

    from nat.cli.entrypoint import cli
    from nat.cli.telemetry_hook import emit_command_event
    from nat.utils.telemetry import TaskStatusEnum

    ctx_obj: dict = {}

    # Run Click with ``standalone_mode=False`` so the real exceptions
    # (KeyboardInterrupt, click.Abort, click.ClickException) reach this
    # wrapper *before* Click rewrites them as a generic ``SystemExit``. With
    # standalone mode, every exit looks the same to us and we'd record
    # ``error_class=None`` and the wrong exit_code for Ctrl-C / bad usage.
    # In exchange we replicate Click's standalone-mode UX explicitly:
    # ``ClickException.show()``, the "Aborted!" line, and the right exit
    # codes per case.
    cli_result = None
    try:
        cli_result = cli(
            obj=ctx_obj,
            auto_envvar_prefix='NAT',
            show_default=True,
            prog_name="nat",
            standalone_mode=False,
        )
    except KeyboardInterrupt as exc:
        emit_command_event(
            ctx_obj,
            task_status=TaskStatusEnum.INTERRUPTED,
            exit_code=130,
            error_class=type(exc).__name__,
        )
        sys.stderr.write("\nAborted!\n")
        sys.exit(130)
    except click.Abort as exc:
        # ``click.Abort`` is raised programmatically (e.g. ``ctx.abort()``).
        # Standalone mode prints "Aborted!" and exits 1.
        emit_command_event(
            ctx_obj,
            task_status=TaskStatusEnum.INTERRUPTED,
            exit_code=1,
            error_class=type(exc).__name__,
        )
        sys.stderr.write("Aborted!\n")
        sys.exit(1)
    except click.ClickException as exc:
        # ``UsageError``, ``BadParameter``, ``MissingParameter``, ``NoSuchOption``,
        # etc. Standalone mode calls ``exc.show()`` and exits with ``exc.exit_code``.
        emit_command_event(
            ctx_obj,
            task_status=TaskStatusEnum.FAILURE,
            exit_code=exc.exit_code,
            error_class=type(exc).__name__,
        )
        exc.show()
        sys.exit(exc.exit_code)
    except SystemExit as exc:
        # A command callback called ``sys.exit(...)`` directly, or Click's
        # ``--help`` / ``--version`` short-circuits (which use ``ctx.exit()``
        # → ``sys.exit(0)`` regardless of standalone mode).
        raw_code = exc.code
        ec_class: str | None
        if raw_code is None or raw_code == 0:
            ts: TaskStatusEnum = TaskStatusEnum.SUCCESS
            ec = 0
            ec_class = None
        elif isinstance(raw_code, int):
            ts = TaskStatusEnum.FAILURE
            ec = raw_code
            ec_class = type(exc).__name__
        else:
            ts = TaskStatusEnum.FAILURE
            ec = 1
            ec_class = type(exc).__name__
        emit_command_event(ctx_obj, task_status=ts, exit_code=ec, error_class=ec_class)
        raise
    except BaseException as exc:  # noqa: BLE001 - we always re-raise
        emit_command_event(
            ctx_obj,
            task_status=TaskStatusEnum.FAILURE,
            exit_code=1,
            error_class=type(exc).__name__,
        )
        raise
    else:
        # Successful return from a non-standalone Click invocation.
        # Honor an int return value from the invoked callback as the
        # process exit code: that's the Click convention for "exit with
        # this code" without raising. Anything else (None, lists from
        # chained commands, arbitrary objects) is treated as exit 0.
        # A non-zero int signals failure even though no exception was
        # raised — record FAILURE so analytics can spot it.
        exit_code = cli_result if isinstance(cli_result, int) else 0
        task_status = TaskStatusEnum.SUCCESS if exit_code == 0 else TaskStatusEnum.FAILURE
        emit_command_event(
            ctx_obj,
            task_status=task_status,
            exit_code=exit_code,
            error_class=None,
        )
        # Return the int so the ``nat`` console-script wrapper's
        # ``sys.exit(run_cli())`` exits the process with the right code.
        return exit_code


if __name__ == '__main__':
    run_cli()
