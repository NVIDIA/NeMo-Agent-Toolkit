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
"""Environment-driven configuration for NAT CLI telemetry.

All values are evaluated once at import time. Changing an environment variable
after `nat.utils.telemetry` has been imported has no effect on the current
process.

Environment variables
---------------------
- ``NAT_TELEMETRY_ENABLED`` (default: ``true``): master opt-out switch.
  Accepts ``1``/``true``/``yes`` (case-insensitive); anything else disables.
- ``NAT_TELEMETRY_ENDPOINT`` (default: empty): destination for telemetry
  payloads. When empty, no HTTP request is issued; events are still built and
  validated, which keeps the path exercised during local development. The
  literal value ``stdout`` is treated as a debug sink that writes JSON-line
  payloads to stderr instead of issuing HTTP POSTs.
- ``NAT_DEPLOYMENT_TYPE`` (default: ``library``): tagged into events.
  Validated against :class:`DeploymentTypeEnum` at import; an invalid value
  raises ``ValueError``.
- ``NAT_SESSION_PREFIX`` (default: unset): optional prefix prepended to
  every session ID. Useful for tagging dev/CI runs.
- ``NAT_TELEMETRY_DRY_RUN`` (default: ``false``): when truthy, payloads are
  built and logged but no HTTP request is issued. Independent of the endpoint
  override.
"""
from __future__ import annotations

import os
import platform
from enum import StrEnum

NAT_TELEMETRY_VERSION = "nat-telemetry/1.0"
"""Identifier embedded in every event envelope as ``eventSysVer``."""

CLIENT_ID = "nvidia-nat-cli"
"""Stable identifier for the NAT CLI client. Sent as ``clientId``."""

CPU_ARCHITECTURE = platform.uname().machine
"""Captured once at import; reported as ``cpuArchitecture`` in payloads."""

# Default ingest endpoint for NAT.
#
# Intentionally blank: an owned ingest URL has not been provisioned yet. With
# no endpoint configured, the handler builds and validates payloads but does
# not issue any HTTP request, which keeps the code path exercised for local
# development. Set ``NAT_TELEMETRY_ENDPOINT=stdout`` to inspect payloads, or
# point at a real endpoint once one is available.
DEFAULT_NAT_TELEMETRY_ENDPOINT = ""

STDOUT_ENDPOINT_SENTINEL = "stdout"
"""When ``NAT_TELEMETRY_ENDPOINT`` equals this value, payloads are written to
stderr as JSON lines instead of POSTed."""

_TRUTHY = ("1", "true", "yes")


def _is_truthy(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY


TELEMETRY_ENABLED: bool = _is_truthy(os.getenv("NAT_TELEMETRY_ENABLED"), default=True)
"""Master opt-out flag. Read at import time; the only source of truth."""

NAT_TELEMETRY_ENDPOINT: str = os.getenv("NAT_TELEMETRY_ENDPOINT", DEFAULT_NAT_TELEMETRY_ENDPOINT)
"""Resolved telemetry endpoint. May be a URL or :data:`STDOUT_ENDPOINT_SENTINEL`."""

NAT_TELEMETRY_DRY_RUN: bool = _is_truthy(os.getenv("NAT_TELEMETRY_DRY_RUN"), default=False)
"""When true, payloads are logged but no HTTP request is made."""

SESSION_PREFIX: str | None = os.getenv("NAT_SESSION_PREFIX") or None


class DeploymentTypeEnum(StrEnum):
    LIBRARY = "library"
    API = "api"
    UNDEFINED = "undefined"


_deployment_type_raw = os.getenv("NAT_DEPLOYMENT_TYPE", DeploymentTypeEnum.LIBRARY.value).strip().lower()
try:
    DEPLOYMENT_TYPE: DeploymentTypeEnum = DeploymentTypeEnum(_deployment_type_raw)
except ValueError:
    _valid = [e.value for e in DeploymentTypeEnum]
    raise ValueError(f"Invalid NAT_DEPLOYMENT_TYPE: {_deployment_type_raw!r}. Must be one of: {_valid}") from None
