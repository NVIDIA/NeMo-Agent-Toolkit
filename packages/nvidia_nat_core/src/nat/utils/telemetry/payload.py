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
"""Wire envelope construction for telemetry payloads.

Privacy: every potentially identifying field in the envelope is hardcoded to
``"undefined"``. The only fields populated from runtime data are the version
string, CPU architecture, and the events themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from typing import Any

from nat.utils.telemetry.config import CLIENT_ID
from nat.utils.telemetry.config import CPU_ARCHITECTURE
from nat.utils.telemetry.config import NAT_TELEMETRY_VERSION
from nat.utils.telemetry.events import TelemetryEvent


@dataclass
class QueuedEvent:
    """In-memory wrapper around an event awaiting flush."""

    event: TelemetryEvent
    timestamp: datetime
    retry_count: int = 0


def _iso_timestamp(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def build_payload(
    events: list[QueuedEvent],
    *,
    source_client_version: str,
    session_id: str = "undefined",
) -> dict[str, Any]:
    """Build a wire envelope for a batch of queued events.

    All identity-bearing fields are hardcoded to ``"undefined"``; do not change
    them without a privacy review. Only the events themselves and the harmless
    metadata (version, CPU arch) carry runtime information.
    """
    if not events:
        raise ValueError("build_payload requires at least one event")

    return {
        "browserType":
            "undefined",
        "clientId":
            CLIENT_ID,
        "clientType":
            "Native",
        "clientVariant":
            "Release",
        "clientVer":
            source_client_version,
        "cpuArchitecture":
            CPU_ARCHITECTURE,
        "deviceGdprBehOptIn":
            "None",
        "deviceGdprFuncOptIn":
            "None",
        "deviceGdprTechOptIn":
            "None",
        "deviceId":
            "undefined",
        "deviceMake":
            "undefined",
        "deviceModel":
            "undefined",
        "deviceOS":
            "undefined",
        "deviceOSVersion":
            "undefined",
        "deviceType":
            "undefined",
        "eventProtocol":
            "1.6",
        "eventSchemaVer":
            events[0].event._schema_version,
        "eventSysVer":
            NAT_TELEMETRY_VERSION,
        "externalUserId":
            "undefined",
        "gdprBehOptIn":
            "None",
        "gdprFuncOptIn":
            "None",
        "gdprTechOptIn":
            "None",
        "idpId":
            "undefined",
        "integrationId":
            "undefined",
        "productName":
            "undefined",
        "productVersion":
            "undefined",
        "sentTs":
            _iso_timestamp(),
        "sessionId":
            session_id,
        "userId":
            "undefined",
        "events": [{
            "ts": _iso_timestamp(queued.timestamp),
            "parameters": queued.event.model_dump(by_alias=True),
            "name": queued.event._event_name,
        } for queued in events],
    }
