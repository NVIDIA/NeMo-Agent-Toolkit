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

from unittest.mock import MagicMock
from unittest.mock import patch

from nat.cli import telemetry_hook
from nat.utils.telemetry import config as config_module
from nat.utils.telemetry.events import CliCommandEvent
from nat.utils.telemetry.events import TaskStatusEnum


def test_emit_command_event_constructs_expected_event():
    ctx_obj = {
        "telemetry_session_id": "abc123",
        "telemetry_start_time": 0.0,
        "telemetry_command": "run",
    }
    captured: list[CliCommandEvent] = []

    handler_instance = MagicMock()
    handler_instance.__enter__ = MagicMock(return_value=handler_instance)
    handler_instance.__exit__ = MagicMock(return_value=False)
    handler_instance.enqueue.side_effect = lambda ev: captured.append(ev)

    with patch.object(config_module, "TELEMETRY_ENABLED", True), \
         patch.object(telemetry_hook, "NATTelemetryHandler", return_value=handler_instance) as mock_cls, \
         patch("time.monotonic", return_value=2.5):
        telemetry_hook.emit_command_event(
            ctx_obj,
            task_status=TaskStatusEnum.SUCCESS,
            exit_code=0,
        )

    mock_cls.assert_called_once()
    assert mock_cls.call_args.kwargs["session_id"] == "abc123"

    assert len(captured) == 1
    event = captured[0]
    assert event.command == "run"
    assert event.task_status == TaskStatusEnum.SUCCESS
    assert event.exit_code == 0
    assert event.duration_ms == 2500  # (2.5 - 0.0) * 1000
    # Per D-19 the schema requires strings; emit_command_event coerces None -> "undefined".
    assert event.error_class == "undefined"


def test_emit_command_event_skipped_when_telemetry_disabled():
    with patch.object(config_module, "TELEMETRY_ENABLED", False), \
         patch.object(telemetry_hook, "NATTelemetryHandler") as mock_cls:
        telemetry_hook.emit_command_event(
            {},
            task_status=TaskStatusEnum.SUCCESS,
            exit_code=0,
        )
    mock_cls.assert_not_called()


def test_emit_command_event_failure_includes_error_class():
    handler_instance = MagicMock()
    handler_instance.__enter__ = MagicMock(return_value=handler_instance)
    handler_instance.__exit__ = MagicMock(return_value=False)
    captured: list[CliCommandEvent] = []
    handler_instance.enqueue.side_effect = lambda ev: captured.append(ev)

    ctx_obj = {
        "telemetry_session_id": "sid",
        "telemetry_start_time": 0.0,
        "telemetry_command": "evaluate",
    }

    with patch.object(config_module, "TELEMETRY_ENABLED", True), \
         patch.object(telemetry_hook, "NATTelemetryHandler", return_value=handler_instance):
        telemetry_hook.emit_command_event(
            ctx_obj,
            task_status=TaskStatusEnum.FAILURE,
            exit_code=1,
            error_class="ValueError",
        )

    assert captured[0].task_status == TaskStatusEnum.FAILURE
    assert captured[0].error_class == "ValueError"
    assert captured[0].exit_code == 1


def test_emit_command_event_handles_missing_ctx_obj():
    handler_instance = MagicMock()
    handler_instance.__enter__ = MagicMock(return_value=handler_instance)
    handler_instance.__exit__ = MagicMock(return_value=False)
    captured: list[CliCommandEvent] = []
    handler_instance.enqueue.side_effect = lambda ev: captured.append(ev)

    with patch.object(config_module, "TELEMETRY_ENABLED", True), \
         patch.object(telemetry_hook, "NATTelemetryHandler", return_value=handler_instance):
        telemetry_hook.emit_command_event(
            None,
            task_status=TaskStatusEnum.FAILURE,
            exit_code=2,
        )

    assert captured[0].command == "unknown"
    assert captured[0].duration_ms == -1


def test_emit_command_event_swallows_handler_errors():
    """A bug or exception inside telemetry must never propagate."""

    class Boom:

        def __enter__(self):
            raise RuntimeError("internal telemetry bug")

        def __exit__(self, *a):
            return False

    with patch.object(config_module, "TELEMETRY_ENABLED", True), \
         patch.object(telemetry_hook, "NATTelemetryHandler", return_value=Boom()):
        # Should not raise.
        telemetry_hook.emit_command_event(
            {"telemetry_command": "run"},
            task_status=TaskStatusEnum.SUCCESS,
            exit_code=0,
        )


def test_resolve_subcommand_picks_first_non_flag_after_command(monkeypatch):
    monkeypatch.setattr("sys.argv", ["nat", "info", "list-components"])
    assert telemetry_hook._resolve_subcommand() == "list-components"


def test_resolve_subcommand_skips_top_level_flags(monkeypatch):
    monkeypatch.setattr("sys.argv", ["nat", "--log-level", "DEBUG", "info", "list-components", "--filter", "x"])
    assert telemetry_hook._resolve_subcommand() == "list-components"


def test_resolve_subcommand_returns_none_when_only_top_level(monkeypatch):
    monkeypatch.setattr("sys.argv", ["nat", "run"])
    assert telemetry_hook._resolve_subcommand() is None
