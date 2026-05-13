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

from datetime import UTC
from datetime import datetime

import pytest

from nat.utils.telemetry.events import CliCommandEvent
from nat.utils.telemetry.events import TaskStatusEnum
from nat.utils.telemetry.payload import QueuedEvent
from nat.utils.telemetry.payload import build_payload


def _sample_queued_event() -> QueuedEvent:
    event = CliCommandEvent(
        command="run",
        task_status=TaskStatusEnum.SUCCESS,
        duration_ms=10,
        exit_code=0,
        python_version="3.11.7",
    )
    return QueuedEvent(event=event, timestamp=datetime.now(UTC))


def test_build_payload_has_no_pii_fields_populated():
    payload = build_payload([_sample_queued_event()], source_client_version="0.1.0", session_id="sess-123")

    pii_fields = (
        "deviceId",
        "deviceMake",
        "deviceModel",
        "deviceOS",
        "deviceOSVersion",
        "deviceType",
        "externalUserId",
        "userId",
        "idpId",
        "integrationId",
        "browserType",
        "productName",
        "productVersion",
    )
    for field in pii_fields:
        assert payload[field] == "undefined", f"{field} must be 'undefined' but got {payload[field]!r}"

    for gdpr_field in ("deviceGdprBehOptIn",
                       "deviceGdprFuncOptIn",
                       "deviceGdprTechOptIn",
                       "gdprBehOptIn",
                       "gdprFuncOptIn",
                       "gdprTechOptIn"):
        assert payload[gdpr_field] == "None"


def test_build_payload_carries_session_and_version():
    payload = build_payload([_sample_queued_event()], source_client_version="9.9.9", session_id="my-session")
    assert payload["sessionId"] == "my-session"
    assert payload["clientVer"] == "9.9.9"
    # clientId is shared with the NeMo Usage Telemetry project (D-18); NAT events
    # are distinguished by nemoSource = "agent_toolkit" rather than a separate clientId.
    assert payload["clientId"] == "184482118588404"
    assert payload["eventSchemaVer"] == "1.5"


def test_build_payload_contains_one_event_per_queued():
    e1 = _sample_queued_event()
    e2 = _sample_queued_event()
    payload = build_payload([e1, e2], source_client_version="0.1.0")
    assert len(payload["events"]) == 2
    assert payload["events"][0]["name"] == "nat_cli_command"
    assert payload["events"][0]["parameters"]["command"] == "run"


def test_build_payload_rejects_empty_events():
    with pytest.raises(ValueError):
        build_payload([], source_client_version="0.1.0")
