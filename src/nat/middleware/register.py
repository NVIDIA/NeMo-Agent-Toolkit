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
from nat.middleware.defense_middleware_content_guard import ContentSafetyGuardMiddleware
from nat.middleware.defense_middleware_content_guard import ContentSafetyGuardMiddlewareConfig
from nat.middleware.defense_middleware_output_verifier import OutputVerifierMiddleware
from nat.middleware.defense_middleware_output_verifier import OutputVerifierMiddlewareConfig
from nat.middleware.defense_middleware_pii import PIIDefenseMiddleware
from nat.middleware.defense_middleware_pii import PIIDefenseMiddlewareConfig
from nat.middleware.red_teaming_middleware import RedTeamingMiddleware
from nat.middleware.red_teaming_middleware_config import RedTeamingMiddlewareConfig


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

@register_middleware(config_type=RedTeamingMiddlewareConfig)
async def red_teaming_middleware(config: RedTeamingMiddlewareConfig, builder):
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
                               target_field=config.target_field)

@register_middleware(config_type=ContentSafetyGuardMiddlewareConfig)
async def content_safety_guard_middleware(config: ContentSafetyGuardMiddlewareConfig, builder):
    """Build a Content Safety Guard middleware from configuration.

    Args:
        config: The content safety guard middleware configuration
        builder: The workflow builder used to resolve the LLM

    Yields:
        A configured Content Safety Guard middleware instance
    """
    # Pass the builder and config, LLM will be loaded lazily
    yield ContentSafetyGuardMiddleware(config=config, builder=builder)

@register_middleware(config_type=OutputVerifierMiddlewareConfig)
async def output_verifier_middleware(config: OutputVerifierMiddlewareConfig, builder):
    """Build an Output Verifier middleware from configuration.

    Args:
        config: The Output Verifier middleware configuration
        builder: The workflow builder used to resolve the LLM

    Yields:
        A configured Output Verifier middleware instance
    """
    # Pass the builder and config, LLM will be loaded lazily
    yield OutputVerifierMiddleware(config=config, builder=builder)

@register_middleware(config_type=PIIDefenseMiddlewareConfig)
async def pii_defense_middleware(config: PIIDefenseMiddlewareConfig, builder):
    """Build a PII Defense middleware from configuration.

    Args:
        config: The PII Defense middleware configuration
        builder: The workflow builder (not used for PII defense)

    Yields:
        A configured PII Defense middleware instance
    """
    # Pass the builder and config, Presidio will be loaded lazily
    yield PIIDefenseMiddleware(config=config, builder=builder)

