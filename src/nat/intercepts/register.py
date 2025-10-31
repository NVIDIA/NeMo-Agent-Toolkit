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

"""Registration module for built-in function intercepts."""

from typing import Literal

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function_intercept
from nat.data_models.function_intercept import FunctionInterceptBaseConfig
from nat.intercepts.cache_intercept import CacheIntercept


class CacheInterceptConfig(FunctionInterceptBaseConfig, name="cache"):
    """Configuration for cache intercept middleware.

    The cache intercept memoizes function outputs based on input similarity,
    with support for both exact and fuzzy matching.

    Args:
        enabled_mode: Controls when caching is active:
            - "always": Cache is always enabled
            - "eval": Cache only active when Context.is_evaluating is True
        similarity_threshold: Float between 0 and 1 for input matching:
            - 1.0: Exact string matching (fastest)
            - < 1.0: Fuzzy matching using difflib similarity
    """

    enabled_mode: Literal["always", "eval"] = Field(
        default="eval",
        description="When caching is enabled: 'always' or 'eval' (only during evaluation)"
    )
    similarity_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Similarity threshold between 0 and 1. Use 1.0 for exact matching"
    )


@register_function_intercept(config_type=CacheInterceptConfig)
async def cache_intercept(config: CacheInterceptConfig, builder: Builder):
    """Build a cache intercept from configuration.

    Args:
        config: The cache intercept configuration
        builder: The workflow builder (unused but required by component pattern)

    Yields:
        A configured cache intercept instance
    """
    yield CacheIntercept(
        enabled_mode=config.enabled_mode,
        similarity_threshold=config.similarity_threshold
    )
