# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Registration module for built-in middleware."""

from __future__ import annotations

from nat.cli.register_workflow import register_middleware
from nat.middleware.cache_middleware import CacheMiddleware
from nat.middleware.cache_middleware import CacheMiddlewareConfig


@register_middleware(config_type=CacheMiddlewareConfig)
async def cache_middleware(config: CacheMiddlewareConfig, builder):
    """Build a cache middleware from configuration.

    Args:
        config: The cache middleware configuration
        builder: The workflow builder (unused but required by component pattern)

    Yields:
        A configured cache middleware instance
    """
    yield CacheMiddleware(enabled_mode=config.enabled_mode, similarity_threshold=config.similarity_threshold)
