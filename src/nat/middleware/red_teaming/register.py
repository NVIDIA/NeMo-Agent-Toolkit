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
"""Registration module for red teaming middleware."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_middleware
from nat.middleware.red_teaming.red_teaming_middleware import RedTeamingMiddleware
from nat.middleware.red_teaming.red_teaming_middleware_config import RedTeamingMiddlewareConfig


@register_middleware(config_type=RedTeamingMiddlewareConfig)
async def red_teaming_middleware(
    config: RedTeamingMiddlewareConfig,
    builder: Builder,
) -> AsyncGenerator[RedTeamingMiddleware, None]:
    """Build a red teaming middleware from configuration.

    Args:
        config: The red teaming middleware configuration
        builder: The workflow builder (unused but required by component pattern)

    Yields:
        A configured red teaming middleware instance
    """
    yield RedTeamingMiddleware(attack_payload=config.attack_payload,
                               target_function_or_group=config.target_function_or_group,
                               payload_placement=config.payload_placement,
                               target_location=config.target_location,
                               target_field=config.target_field,
                               target_field_resolution_strategy=config.target_field_resolution_strategy,
                               call_limit=config.call_limit)
