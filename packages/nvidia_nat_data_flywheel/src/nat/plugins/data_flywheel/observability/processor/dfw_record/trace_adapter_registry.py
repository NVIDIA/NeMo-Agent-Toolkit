# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

from nat.plugins.data_flywheel.observability.processor.dfw_record.adapters import TraceSourceAdapter
from nat.plugins.data_flywheel.observability.schema.trace_source import TraceSource

logger = logging.getLogger(__name__)


class TraceAdapterRegistry:
    """Registry for managing trace source adapters."""

    _adapters: dict[str, TraceSourceAdapter] = {}

    @classmethod
    def register_adapter(cls, adapter: TraceSourceAdapter):
        """Register a new adapter."""
        cls._adapters[adapter.framework_identifier] = adapter
        logger.debug("Registered adapter for framework: %s", adapter.framework_identifier)

    @classmethod
    def unregister_adapter(cls, framework_identifier: str) -> bool:
        """Unregister an adapter by framework identifier.

        Args:
            framework_identifier: The framework identifier to unregister

        Returns:
            True if adapter was found and removed, False otherwise
        """
        if framework_identifier in cls._adapters:
            cls._adapters.pop(framework_identifier)
            logger.debug("Unregistered adapter for framework: %s", framework_identifier)
            return True
        logger.warning("Attempted to unregister non-existent adapter: %s", framework_identifier)
        return False

    @classmethod
    def get_adapter(cls, trace_source: TraceSource) -> TraceSourceAdapter | None:
        """Get the appropriate adapter for a trace source."""
        # Input validation: Ensure required fields are present and valid
        if not trace_source.source.framework or not trace_source.source.provider:
            logger.warning("Invalid trace source: missing framework ('%s') or provider ('%s')",
                           trace_source.source.framework,
                           trace_source.source.provider)
            return None
        framework_provider = f"{trace_source.source.framework}_{trace_source.source.provider}"
        return cls._adapters.get(framework_provider)

    @classmethod
    def list_supported_frameworks(cls) -> list[str]:
        """List all supported framework identifiers."""
        return list(cls._adapters.keys())


def register_span_adapter(adapter: TraceSourceAdapter):
    """Register a custom adapter globally."""
    TraceAdapterRegistry.register_adapter(adapter)


def unregister_span_adapter(framework_identifier: str) -> bool:
    """Unregister an adapter by framework identifier.

    Args:
        framework_identifier: The framework identifier to unregister

    Returns:
        True if adapter was found and removed, False otherwise
    """
    return TraceAdapterRegistry.unregister_adapter(framework_identifier)
