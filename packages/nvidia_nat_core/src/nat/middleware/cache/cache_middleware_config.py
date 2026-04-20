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
"""Configuration for cache middleware."""

from typing import Literal

from pydantic import Field

from nat.data_models.middleware import FunctionMiddlewareBaseConfig
from nat.middleware.cache.cache_middleware import _DEFAULT_MAX_CACHE_ENTRIES


class CacheMiddlewareConfig(FunctionMiddlewareBaseConfig, name="cache"):
    """Configuration for cache middleware.

    The cache middleware memoizes function outputs based on input similarity,
    with support for both exact and fuzzy matching.

    Args:
        enabled_mode: Controls when caching is active:
            - "always": Cache is always enabled
            - "eval": Cache only active when Context.is_evaluating is True
        similarity_threshold: Float in [0, 1.0] for input matching:
            - 1.0: Exact string matching (fastest, recommended)
            - < 1.0: Fuzzy matching via difflib. Note that difflib is
              quadratic in the worst case, so large caches with low
              thresholds may have a performance cost. Values near 0
              increase the risk of cache collisions where different
              inputs return the same cached response.
        max_entries: Upper bound on cached entries. When exceeded, the
            least-recently-used entry is evicted. Must be a positive int;
            defaults to _DEFAULT_MAX_CACHE_ENTRIES.
    """

    enabled_mode: Literal["always", "eval"] = Field(
        default="eval", description="When caching is enabled: 'always' or 'eval' (only during evaluation)")
    similarity_threshold: float = Field(
        default=1.0,
        ge=0,
        le=1.0,
        description=(
            "Similarity threshold in [0, 1.0]. Use 1.0 for exact matching (recommended). "
            "Lower values enable fuzzy matching via difflib; note that difflib is quadratic "
            "in the worst case, so large caches with low thresholds may have a performance "
            "cost. Values near 0 increase the risk of cache collisions where different "
            "inputs return the same cached response."),
    )
    max_entries: int = Field(
        default=_DEFAULT_MAX_CACHE_ENTRIES,
        ge=1,
        description=("Maximum number of cache entries before LRU eviction. Must be >= 1. "
                     "Prevents memory-exhaustion DoS from unbounded cache growth under "
                     "sustained unique inputs."),
    )
