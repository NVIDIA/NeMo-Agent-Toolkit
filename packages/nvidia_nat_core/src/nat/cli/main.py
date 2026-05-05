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
def run_cli():
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
    try:
        cli(
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
        emit_command_event(
            ctx_obj,
            task_status=TaskStatusEnum.SUCCESS,
            exit_code=0,
            error_class=None,
        )


if __name__ == '__main__':
    run_cli()
