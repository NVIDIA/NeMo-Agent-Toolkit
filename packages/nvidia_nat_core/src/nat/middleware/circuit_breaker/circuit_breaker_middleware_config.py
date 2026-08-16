# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Configuration for circuit breaker middleware."""

from __future__ import annotations

from pydantic import Field

from nat.middleware.dynamic.dynamic_middleware_config import DynamicMiddlewareConfig


class CircuitBreakerMiddlewareConfig(DynamicMiddlewareConfig, name="circuit_breaker"):
    """Configuration for circuit breaker middleware.

    Attributes:
        failure_threshold: Number of consecutive failures required to trip the circuit breaker.
        cooldown_period: Time in seconds to wait in OPEN state before probing.
        half_open_success_threshold: Consecutive successful probes required to recover to CLOSED.
        probe_timeout: Optional timeout in seconds for probe calls in HALF_OPEN state.
        circuit_breaker_message: Optional custom message returned when short-circuited.
    """

    failure_threshold: int = Field(
        default=3,
        gt=0,
        description="Number of consecutive failures required to trip the circuit breaker into OPEN state.",
    )

    cooldown_period: float = Field(
        default=60.0,
        gt=0,
        description="Time in seconds to wait in OPEN state before transitioning to HALF_OPEN to probe availability.",
    )

    half_open_success_threshold: int = Field(
        default=1,
        gt=0,
        description=(
            "Number of consecutive successful probe calls in HALF_OPEN state required to recover to CLOSED state."),
    )

    probe_timeout: float | None = Field(
        default=None,
        gt=0,
        description="Optional timeout in seconds for probe calls in HALF_OPEN state.",
    )

    circuit_breaker_message: str | None = Field(
        default=None,
        description=("Optional custom message returned when calls are "
                     "short-circuited while the circuit breaker is OPEN."),
    )
