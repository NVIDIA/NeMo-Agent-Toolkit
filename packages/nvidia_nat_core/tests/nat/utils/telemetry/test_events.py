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

from typing import ClassVar

import pytest

from nat.utils.telemetry.events import CliCommandEvent
from nat.utils.telemetry.events import TaskStatusEnum
from nat.utils.telemetry.events import TelemetryEvent


def test_telemetry_event_subclass_must_define_event_name():
    with pytest.raises(TypeError, match="must define '_event_name'"):

        class BadEvent(TelemetryEvent):  # noqa: F841 - intentional definition for the test
            pass


def test_telemetry_event_subclass_with_event_name_succeeds():

    class GoodEvent(TelemetryEvent):
        _event_name: ClassVar[str] = "good_event"

    assert GoodEvent._event_name == "good_event"
    assert GoodEvent._schema_version == "1.0"


def test_cli_command_event_serializes_with_camel_case_aliases():
    event = CliCommandEvent(
        command="run",
        subcommand=None,
        task_status=TaskStatusEnum.SUCCESS,
        duration_ms=1234,
        exit_code=0,
        error_class=None,
        python_version="3.11.7",
    )
    dumped = event.model_dump(by_alias=True)

    assert dumped["command"] == "run"
    assert dumped["taskStatus"] == "success"
    assert dumped["deploymentType"] in ("library", "api", "undefined")
    assert dumped["durationMs"] == 1234
    assert dumped["exitCode"] == 0
    assert dumped["errorClass"] is None
    assert dumped["pythonVersion"] == "3.11.7"
    assert event._event_name == "nat_cli_command"


def test_cli_command_event_supports_failure_with_error_class():
    event = CliCommandEvent(
        command="evaluate",
        task_status=TaskStatusEnum.FAILURE,
        exit_code=1,
        error_class="ValueError",
        python_version="3.11.7",
    )
    dumped = event.model_dump(by_alias=True)
    assert dumped["taskStatus"] == "failure"
    assert dumped["errorClass"] == "ValueError"


def test_cli_command_event_rejects_negative_duration_below_minus_one():
    with pytest.raises(ValueError):
        CliCommandEvent(
            command="run",
            task_status=TaskStatusEnum.SUCCESS,
            duration_ms=-2,
            exit_code=0,
            python_version="3.11.7",
        )
