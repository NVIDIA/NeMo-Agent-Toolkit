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

    from nat.cli.entrypoint import cli
    from nat.cli.telemetry_hook import emit_command_event
    from nat.utils.telemetry import TaskStatusEnum

    ctx_obj: dict = {}
    task_status: TaskStatusEnum = TaskStatusEnum.SUCCESS
    exit_code: int = 0
    error_class: str | None = None

    try:
        cli(obj=ctx_obj, auto_envvar_prefix='NAT', show_default=True, prog_name="nat")
    except SystemExit as exc:
        # Click's standalone_mode=True converts every exit (success, --help,
        # bad usage, raised exception) into a SystemExit. The code attribute
        # tells us success vs. failure.
        raw_code = exc.code
        if raw_code is None or raw_code == 0:
            exit_code = 0
        elif isinstance(raw_code, int):
            exit_code = raw_code
            task_status = TaskStatusEnum.FAILURE
        else:
            exit_code = 1
            task_status = TaskStatusEnum.FAILURE
        emit_command_event(
            ctx_obj,
            task_status=task_status,
            exit_code=exit_code,
            error_class=error_class,
        )
        raise
    except KeyboardInterrupt:
        emit_command_event(
            ctx_obj,
            task_status=TaskStatusEnum.INTERRUPTED,
            exit_code=130,
            error_class=None,
        )
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
        # Reached only if Click is invoked with standalone_mode=False; in the
        # standard configuration above the success path lands in the SystemExit
        # branch with code 0.
        emit_command_event(
            ctx_obj,
            task_status=TaskStatusEnum.SUCCESS,
            exit_code=0,
            error_class=None,
        )


if __name__ == '__main__':
    run_cli()
