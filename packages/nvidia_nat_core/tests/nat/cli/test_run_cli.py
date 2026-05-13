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
"""Tests for ``nat.cli.main.run_cli`` exit-handling and telemetry.

The wrapper runs Click with ``standalone_mode=False`` so it can see the
real exception types before Click rewrites them as ``SystemExit``. These
tests verify that each exception branch records the right
``task_status`` / ``exit_code`` / ``error_class`` on the telemetry event
and exits with the expected process code.
"""
from __future__ import annotations

from unittest.mock import patch

import click
import pytest

from nat.cli import main as main_module
from nat.utils.telemetry import TaskStatusEnum


def _run_with_cli_raising(exc: BaseException | None = None, return_value=None):
    """Invoke ``run_cli`` with ``cli`` patched to raise ``exc`` (or return ``return_value``).

    Returns ``(emit_calls, sys_exit_code, raised)`` where ``emit_calls``
    is a list of kwargs each call to ``emit_command_event`` saw,
    ``sys_exit_code`` is the integer passed to ``sys.exit`` (or ``None``
    if the wrapper used ``raise`` instead), and ``raised`` is whichever
    exception escaped (or ``None`` on a clean run).
    """
    emit_calls: list[dict] = []

    def fake_cli(*_args, **_kwargs):
        if exc is not None:
            raise exc
        return return_value

    def fake_emit(_ctx, **kwargs):
        emit_calls.append(kwargs)

    raised: BaseException | None = None
    sys_exit_code: int | None = None

    with patch("nat.cli.entrypoint.cli", fake_cli), \
         patch("nat.cli.telemetry_hook.emit_command_event", fake_emit):
        try:
            main_module.run_cli()
        except SystemExit as se:
            sys_exit_code = se.code if isinstance(se.code, int) else (1 if se.code else 0)
            raised = se
        except BaseException as e:  # noqa: BLE001
            raised = e

    return emit_calls, sys_exit_code, raised


def test_run_cli_success_records_success(capsys):
    emits, code, raised = _run_with_cli_raising(exc=None)
    assert raised is None
    assert code is None  # didn't raise SystemExit on the success path
    assert emits == [{
        "task_status": TaskStatusEnum.SUCCESS,
        "exit_code": 0,
        "error_class": None,
    }]


def test_run_cli_callback_returning_zero_int_records_success(capsys):
    """Callback explicitly returning 0 — same as returning None: SUCCESS."""
    emits, _, raised = _run_with_cli_raising(exc=None, return_value=0)
    assert raised is None
    assert emits == [{
        "task_status": TaskStatusEnum.SUCCESS,
        "exit_code": 0,
        "error_class": None,
    }]


def test_run_cli_callback_returning_nonzero_int_records_failure(capsys):
    """Click's ``standalone_mode=False`` returns the callback's return
    value verbatim. A non-zero int signals "exit with this code"
    (Click's convention) — record FAILURE and propagate the code, even
    though no exception was raised."""
    emits, _, raised = _run_with_cli_raising(exc=None, return_value=5)
    assert raised is None
    assert emits == [{
        "task_status": TaskStatusEnum.FAILURE,
        "exit_code": 5,
        "error_class": None,
    }]


def test_run_cli_callback_returning_non_int_treated_as_success(capsys):
    """Non-int return values (lists from chained commands, arbitrary
    objects, strings) are not exit codes — fall back to SUCCESS / 0."""
    for return_value in ([None, None], "ok", {"status": "ok"}):
        emits, _, raised = _run_with_cli_raising(exc=None, return_value=return_value)
        assert raised is None, f"unexpected raise for return_value={return_value!r}"
        assert emits == [{
            "task_status": TaskStatusEnum.SUCCESS,
            "exit_code": 0,
            "error_class": None,
        }], f"wrong telemetry for return_value={return_value!r}"


def test_run_cli_returns_int_for_console_script_wrapper(capsys):
    """The ``nat`` console-script wrapper does ``sys.exit(run_cli())`` —
    so on the int-return path, ``run_cli`` must return that int (not
    ``None``) so the process exits with the right code."""
    # Capture run_cli's actual return value by bypassing the helper's
    # SystemExit translation.
    from unittest.mock import patch as _patch
    with _patch("nat.cli.entrypoint.cli", lambda *a, **kw: 7), \
         _patch("nat.cli.telemetry_hook.emit_command_event", lambda *a, **kw: None):
        ret = main_module.run_cli()
    assert ret == 7


def test_run_cli_keyboard_interrupt_records_interrupted_and_exits_130(capsys):
    emits, code, raised = _run_with_cli_raising(exc=KeyboardInterrupt())
    assert isinstance(raised, SystemExit)
    assert code == 130
    assert emits == [{
        "task_status": TaskStatusEnum.INTERRUPTED,
        "exit_code": 130,
        "error_class": "KeyboardInterrupt",
    }]
    captured = capsys.readouterr()
    assert "Aborted!" in captured.err


def test_run_cli_click_abort_records_interrupted_and_exits_1(capsys):
    emits, code, raised = _run_with_cli_raising(exc=click.Abort())
    assert isinstance(raised, SystemExit)
    assert code == 1
    assert emits == [{
        "task_status": TaskStatusEnum.INTERRUPTED,
        "exit_code": 1,
        "error_class": "Abort",
    }]
    captured = capsys.readouterr()
    assert "Aborted!" in captured.err


def test_run_cli_click_usage_error_records_failure_and_calls_show(capsys):
    emits, code, raised = _run_with_cli_raising(exc=click.UsageError("missing arg"))
    assert isinstance(raised, SystemExit)
    # ``UsageError.exit_code`` is 2 by Click convention.
    assert code == 2
    assert emits == [{
        "task_status": TaskStatusEnum.FAILURE,
        "exit_code": 2,
        "error_class": "UsageError",
    }]
    # exc.show() writes the usage error message to stderr.
    captured = capsys.readouterr()
    assert "missing arg" in captured.err


def test_run_cli_click_bad_parameter_uses_its_exit_code(capsys):
    """``BadParameter`` is also a ``ClickException``; exit code from the exc."""
    exc = click.BadParameter("bad value")
    emits, code, raised = _run_with_cli_raising(exc=exc)
    assert isinstance(raised, SystemExit)
    assert code == exc.exit_code
    assert emits == [{
        "task_status": TaskStatusEnum.FAILURE,
        "exit_code": exc.exit_code,
        "error_class": "BadParameter",
    }]


def test_run_cli_generic_exception_records_failure_and_reraises():
    emits, _, raised = _run_with_cli_raising(exc=ValueError("boom"))
    assert isinstance(raised, ValueError)
    assert emits == [{
        "task_status": TaskStatusEnum.FAILURE,
        "exit_code": 1,
        "error_class": "ValueError",
    }]


def test_run_cli_callback_sys_exit_zero_records_success():
    """Clean sys.exit(0) → SUCCESS with no error_class (no error to classify)."""
    emits, code, raised = _run_with_cli_raising(exc=SystemExit(0))
    assert isinstance(raised, SystemExit)
    assert code == 0
    assert emits == [{
        "task_status": TaskStatusEnum.SUCCESS,
        "exit_code": 0,
        "error_class": None,
    }]


def test_run_cli_callback_sys_exit_nonzero_records_failure_with_systemexit_class():
    """A callback that calls ``sys.exit(5)`` is a programmatic failure;
    error_class should be ``"SystemExit"`` so analytics can distinguish
    callback-driven exits from silent failures."""
    emits, code, raised = _run_with_cli_raising(exc=SystemExit(5))
    assert isinstance(raised, SystemExit)
    assert code == 5
    assert emits == [{
        "task_status": TaskStatusEnum.FAILURE,
        "exit_code": 5,
        "error_class": "SystemExit",
    }]


def test_run_cli_callback_sys_exit_string_message_records_failure_with_code_1():
    """``sys.exit("error message")`` produces ``SystemExit(code='error message')``;
    map non-int codes to FAILURE/1 (matching standalone-mode behavior) and
    still record the SystemExit class for visibility."""
    emits, code, _raised = _run_with_cli_raising(exc=SystemExit("oops"))
    assert code == 1
    assert emits == [{
        "task_status": TaskStatusEnum.FAILURE,
        "exit_code": 1,
        "error_class": "SystemExit",
    }]


@pytest.mark.parametrize(
    "exc_factory,expected_status,expected_class",
    [
        # Order check: KeyboardInterrupt comes before BaseException catch-all.
        (lambda: KeyboardInterrupt(), TaskStatusEnum.INTERRUPTED, "KeyboardInterrupt"),
        # click.Abort takes precedence over generic ClickException catch.
        (lambda: click.Abort(), TaskStatusEnum.INTERRUPTED, "Abort"),
        # ClickException subclass uses its own name, not the parent's.
        (lambda: click.NoSuchOption("--bogus"), TaskStatusEnum.FAILURE, "NoSuchOption"),
    ])
def test_run_cli_exception_routing_priority(exc_factory, expected_status, expected_class):
    emits, _, _ = _run_with_cli_raising(exc=exc_factory())
    assert len(emits) == 1
    assert emits[0]["task_status"] == expected_status
    assert emits[0]["error_class"] == expected_class
