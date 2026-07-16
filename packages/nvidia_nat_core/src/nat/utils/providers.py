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
"""Process-wide provider hooks for generated identifiers and timestamps.

The runtime stamps workflow runs, intermediate steps, spans, and function invocations with freshly generated
UUIDs and wall-clock timestamps. By default these come from :func:`uuid.uuid4` and :func:`time.time`, which is
correct for normal execution but makes two otherwise identical runs produce different identifiers and timings.

Integrations that need reproducible runs can install their own providers. Examples include record/replay style
testing, golden-file trace comparison, and integrations with runtimes that re-execute workflow code and require
identifiers to be stable across re-executions.

Both hooks are process-wide and opt-in. When no provider is installed, behavior is unchanged. The setters return
the previously installed provider so callers can restore it.

The id provider must return strings in canonical UUID form. Integer identifiers, such as the OpenTelemetry-style
128-bit trace ids and 64-bit span ids, are derived from the id provider by parsing the returned string, so a
value that is not a valid UUID string raises :class:`ValueError` at generation time.
"""

import time
import uuid
from collections.abc import Callable

IdProvider = Callable[[], str]
"""Zero-argument callable returning a new identifier as a canonical UUID string."""

TimeProvider = Callable[[], float]
"""Zero-argument callable returning the current time in fractional seconds since the Unix epoch."""


def default_id_provider() -> str:
    """Return a random UUID4 string. This is the default id provider."""
    return str(uuid.uuid4())


def default_time_provider() -> float:
    """Return the current wall-clock time in fractional seconds. This is the default time provider."""
    return time.time()


class _ProviderState:
    """Mutable holder for the process-wide providers. Mutated only through the module-level setters."""

    def __init__(self) -> None:
        self.id_provider: IdProvider = default_id_provider
        self.time_provider: TimeProvider = default_time_provider


_state = _ProviderState()


def set_id_provider(provider: IdProvider) -> IdProvider:
    """Install ``provider`` as the process-wide id provider.

    Args:
        provider: Zero-argument callable returning a new identifier as a canonical UUID string.

    Returns:
        IdProvider: The previously installed id provider, so callers can restore it.
    """
    previous = _state.id_provider
    _state.id_provider = provider
    return previous


def get_id_provider() -> IdProvider:
    """Return the currently installed id provider."""
    return _state.id_provider


def set_time_provider(provider: TimeProvider) -> TimeProvider:
    """Install ``provider`` as the process-wide time provider.

    Args:
        provider: Zero-argument callable returning the current time in fractional seconds since the Unix epoch.

    Returns:
        TimeProvider: The previously installed time provider, so callers can restore it.
    """
    previous = _state.time_provider
    _state.time_provider = provider
    return previous


def get_time_provider() -> TimeProvider:
    """Return the currently installed time provider."""
    return _state.time_provider


def generate_id() -> str:
    """Return a new identifier string from the installed id provider."""
    return _state.id_provider()


def generate_trace_id() -> int:
    """Return a new 128-bit integer id derived from the installed id provider.

    With the default provider this is equivalent to ``uuid.uuid4().int``.

    Raises:
        ValueError: If the installed id provider generated a UUID whose derived trace ID is zero.
    """
    trace_id = uuid.UUID(generate_id()).int
    if trace_id == 0:
        raise ValueError("The installed id provider generated a UUID with an all-zero trace ID")
    return trace_id


def generate_span_id() -> int:
    """Return a new 64-bit integer id derived from the installed id provider.

    With the default provider this is equivalent to ``uuid.uuid4().int >> 64``.

    Raises:
        ValueError: If the installed id provider generated a UUID whose derived span ID is zero.
    """
    span_id = uuid.UUID(generate_id()).int >> 64
    if span_id == 0:
        raise ValueError("The installed id provider generated a UUID with an all-zero span ID")
    return span_id


def current_time() -> float:
    """Return the current time in fractional seconds from the installed time provider."""
    return _state.time_provider()


def current_time_ns() -> int:
    """Return the current time in integer nanoseconds from the installed time provider."""
    return int(current_time() * 1e9)
