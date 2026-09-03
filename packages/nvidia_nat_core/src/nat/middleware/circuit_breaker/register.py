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
"""Registration for circuit breaker middleware."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nat.builder.builder import Builder

from nat.cli.register_workflow import register_middleware
from nat.middleware.circuit_breaker.circuit_breaker_middleware import CircuitBreakerMiddleware
from nat.middleware.circuit_breaker.circuit_breaker_middleware_config import CircuitBreakerMiddlewareConfig


@register_middleware(config_type=CircuitBreakerMiddlewareConfig)
async def circuit_breaker_middleware(
    config: CircuitBreakerMiddlewareConfig,
    builder: Builder,
) -> AsyncGenerator[CircuitBreakerMiddleware, None]:
    """Build a circuit breaker middleware from configuration.

    Args:
        config: The circuit breaker middleware configuration
        builder: The workflow builder

    Yields:
        A configured circuit breaker middleware instance
    """
    yield CircuitBreakerMiddleware(config=config, builder=builder)
