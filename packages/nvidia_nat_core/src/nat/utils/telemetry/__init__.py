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
"""Opt-out runtime telemetry for the NAT CLI.

See `nat.utils.telemetry.config` for environment variables (defaults: opt-out
disabled, telemetry enabled). To turn telemetry off for a single invocation:

    NAT_TELEMETRY_ENABLED=false nat run ...

To inspect events locally without making network calls:

    NAT_TELEMETRY_ENDPOINT=stdout nat run ...
"""
from __future__ import annotations

from nat.utils.telemetry.config import TELEMETRY_ENABLED
from nat.utils.telemetry.consent import ConsentState
from nat.utils.telemetry.consent import maybe_prompt_for_consent
from nat.utils.telemetry.consent import read_persisted_consent
from nat.utils.telemetry.consent import write_persisted_consent
from nat.utils.telemetry.events import CliCommandEvent
from nat.utils.telemetry.events import NemoSourceEnum
from nat.utils.telemetry.events import TaskStatusEnum
from nat.utils.telemetry.events import TelemetryEvent
from nat.utils.telemetry.handler import NATTelemetryHandler

__all__ = [
    "CliCommandEvent",
    "ConsentState",
    "NATTelemetryHandler",
    "NemoSourceEnum",
    "TELEMETRY_ENABLED",
    "TaskStatusEnum",
    "TelemetryEvent",
    "maybe_prompt_for_consent",
    "read_persisted_consent",
    "write_persisted_consent",
]
