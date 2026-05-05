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
"""Configuration for NAT CLI telemetry.

Most values are evaluated once at import time. The ``TELEMETRY_ENABLED``
flag is module-level mutable: it starts at the value derived from the
environment / persisted consent, and the CLI entrypoint may flip it after
the first-run consent prompt resolves.

Resolution order for ``TELEMETRY_ENABLED`` (lazy: import-time + prompt):
    1. ``NAT_TELEMETRY_ENABLED`` env var, if set.
    2. Persisted consent at ``~/.config/nat/telemetry.toml``.
    3. ``False`` — until the CLI's first-run prompt resolves to either
       ENABLED (TTY user said yes) or DISABLED (TTY user said no, or
       non-interactive session).

Environment variables
---------------------
- ``NAT_TELEMETRY_ENABLED`` (no default — first-run prompt or persisted
  consent decides): master opt-out switch. Accepts ``1``/``true``/``yes``
  (case-insensitive); anything else disables.
- ``NAT_TELEMETRY_ENDPOINT`` (default:
  ``https://events.telemetry.data.nvidia.com/v1.1/events/json``):
  destination for telemetry payloads. The default points at the shared
  NeMo Usage Telemetry production ingest. Set to the empty string to
  build payloads locally without issuing any HTTP request, or to the
  literal value ``stdout`` to write JSON-line payloads to stderr for
  inspection.
- ``NAT_SESSION_PREFIX`` (default: unset): optional prefix prepended to
  every session ID. Useful for tagging dev/CI runs.
- ``NAT_TELEMETRY_DRY_RUN`` (default: ``false``): when truthy, payloads
  are built and logged but no HTTP request is issued.
- ``NAT_TELEMETRY_CONSENT_FILE`` (default:
  ``~/.config/nat/telemetry.toml``): override for the persisted consent
  file location. Primarily a testing hook.
"""
from __future__ import annotations

import os
import platform

NAT_TELEMETRY_VERSION = "nat-telemetry/1.0"
"""Identifier embedded in every event envelope as ``eventSysVer``."""

CLIENT_ID = "184482118588404"
"""Stable identifier for the NAT CLI client. Sent as ``clientId``.

Shared with the NeMo Usage Telemetry project — NAT events are tagged by
``nemoSource = "agent_toolkit"`` to distinguish them from sibling NeMo
products' events at query time."""

CPU_ARCHITECTURE = platform.uname().machine
"""Captured once at import; reported as ``cpuArchitecture`` in payloads."""

# Default ingest endpoint for NAT.
#
# Mirrors the production URL used by ``nemo-telemetry/telemetry.py`` (the
# canonical reference handler for the shared NeMo Usage Telemetry project).
# Sending is still gated behind opt-in consent (``TELEMETRY_ENABLED``); the
# default endpoint only matters once the user has consented.
#
# Override paths:
# - ``NAT_TELEMETRY_ENDPOINT=""`` — build and validate payloads locally
#   without issuing any HTTP request. Useful for offline development.
# - ``NAT_TELEMETRY_ENDPOINT=stdout`` — write JSON-line payloads to stderr
#   for inspection. Useful for verifying the wire shape.
# - any other URL — point at a custom ingest (e.g. UAT for stage testing).
DEFAULT_NAT_TELEMETRY_ENDPOINT = "https://events.telemetry.data.nvidia.com/v1.1/events/json"

STDOUT_ENDPOINT_SENTINEL = "stdout"
"""When ``NAT_TELEMETRY_ENDPOINT`` equals this value, payloads are written to
stderr as JSON lines instead of POSTed."""

_TRUTHY = ("1", "true", "yes")


def _is_truthy(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY


def _resolve_initial_telemetry_enabled() -> bool:
    """Resolve the initial value at import time.

    Defers persisted-consent reading to ``consent.resolve_initial_consent``
    so all the policy rules live in one place. The CLI entrypoint may
    later flip this flag via the consent prompt; consumers (handler,
    telemetry_hook) re-read the live value rather than caching it.
    """
    from nat.utils.telemetry.consent import resolve_initial_consent
    return resolve_initial_consent()


TELEMETRY_ENABLED: bool = _resolve_initial_telemetry_enabled()
"""Master opt-out flag. Initialized at import; may be flipped by the
first-run consent prompt during a CLI invocation. Consumers should access
this attribute on the module rather than ``from … import TELEMETRY_ENABLED``
so the live value is honored."""

NAT_TELEMETRY_ENDPOINT: str = os.getenv("NAT_TELEMETRY_ENDPOINT", DEFAULT_NAT_TELEMETRY_ENDPOINT)
"""Resolved telemetry endpoint. May be a URL or :data:`STDOUT_ENDPOINT_SENTINEL`."""

NAT_TELEMETRY_DRY_RUN: bool = _is_truthy(os.getenv("NAT_TELEMETRY_DRY_RUN"), default=False)
"""When true, payloads are logged but no HTTP request is made."""

SESSION_PREFIX: str | None = os.getenv("NAT_SESSION_PREFIX") or None
