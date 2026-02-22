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
"""Registration module for defense middleware."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_middleware
from nat.middleware.defense.defense_middleware_content_guard import ContentSafetyGuardMiddleware
from nat.middleware.defense.defense_middleware_content_guard import ContentSafetyGuardMiddlewareConfig
from nat.middleware.defense.defense_middleware_output_verifier import OutputVerifierMiddleware
from nat.middleware.defense.defense_middleware_output_verifier import OutputVerifierMiddlewareConfig
from nat.middleware.defense.defense_middleware_pii import PIIDefenseMiddleware
from nat.middleware.defense.defense_middleware_pii import PIIDefenseMiddlewareConfig


@register_middleware(config_type=ContentSafetyGuardMiddlewareConfig)
async def content_safety_guard_middleware(
    config: ContentSafetyGuardMiddlewareConfig,
    builder: Builder,
) -> AsyncGenerator[ContentSafetyGuardMiddleware, None]:
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
async def output_verifier_middleware(
    config: OutputVerifierMiddlewareConfig,
    builder: Builder,
) -> AsyncGenerator[OutputVerifierMiddleware, None]:
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
async def pii_defense_middleware(
    config: PIIDefenseMiddlewareConfig,
    builder: Builder,
) -> AsyncGenerator[PIIDefenseMiddleware, None]:
    """Build a PII Defense middleware from configuration.

    Args:
        config: The PII Defense middleware configuration
        builder: The workflow builder (not used for PII defense)

    Yields:
        A configured PII Defense middleware instance
    """
    # Pass the builder and config, Presidio will be loaded lazily
    yield PIIDefenseMiddleware(config=config, builder=builder)
