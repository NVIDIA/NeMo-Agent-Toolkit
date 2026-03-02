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
"""Registration for timeout middleware."""

from collections.abc import AsyncGenerator

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_middleware
from nat.middleware.timeout.timeout_middleware import TimeoutMiddleware
from nat.middleware.timeout.timeout_middleware_config import TimeoutMiddlewareConfig


@register_middleware(config_type=TimeoutMiddlewareConfig)
async def timeout_middleware(
    config: TimeoutMiddlewareConfig,
    builder: Builder,
) -> AsyncGenerator[TimeoutMiddleware, None]:
    """Build a timeout middleware from configuration.

    Args:
        config: The timeout middleware configuration
        builder: The workflow builder

    Yields:
        A configured timeout middleware instance
    """
    yield TimeoutMiddleware(config=config, builder=builder)
