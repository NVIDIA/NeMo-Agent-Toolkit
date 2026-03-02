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
"""Configuration for timeout middleware."""

from __future__ import annotations

from pydantic import Field

from nat.middleware.dynamic.dynamic_middleware_config import DynamicMiddlewareConfig


class TimeoutMiddlewareConfig(DynamicMiddlewareConfig, name="timeout"):
    """Configuration for timeout middleware.
    """

    timeout: float = Field(
        description="Timeout in seconds for all calls intercepted by this middleware instance.",
        gt=0,
    )

    timeout_message: str | None = Field(
        default=None,
        description="Additional message appended to the TimeoutError raised on expiry.",
    )
