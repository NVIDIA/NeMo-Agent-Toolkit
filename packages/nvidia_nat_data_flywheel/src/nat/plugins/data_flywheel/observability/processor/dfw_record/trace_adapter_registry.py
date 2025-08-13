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

import logging
from typing import Any
from typing import TypeVar

from nat.plugins.data_flywheel.observability.processor.dfw_record.adapters import TraceSourceAdapter
from nat.plugins.data_flywheel.observability.schema.trace_source import TraceSource

OutputT = TypeVar("OutputT")

logger = logging.getLogger(__name__)


class TraceAdapterRegistry:
    """Registry for managing trace source adapters.

    Supports multiple output types while maintaining backward compatibility.
    Stores adapters in format: {framework_identifier: {output_type_name: adapter}}
    """

    _adapters: dict[str, dict[str, TraceSourceAdapter[Any]]] = {}

    @classmethod
    def register_adapter(cls, adapter: TraceSourceAdapter[Any]):
        """Register a new adapter.

        The output type is automatically determined from the adapter's output_type property.

        Args:
            adapter: The adapter to register
        """
        framework_id = adapter.framework_identifier
        output_type_name = adapter.output_type.__name__

        if framework_id not in cls._adapters:
            cls._adapters[framework_id] = {}

        cls._adapters[framework_id][output_type_name] = adapter
        logger.debug("Registered adapter for framework: %s, output type: %s", framework_id, output_type_name)

    @classmethod
    def unregister_adapter(cls, framework_identifier: str, output_type: type[Any] | None = None) -> bool:
        """Unregister an adapter by framework identifier and output type.

        Args:
            framework_identifier: The framework identifier to unregister
            output_type: The output type to unregister (if None, removes all adapters
                        for this framework)

        Returns:
            True if adapter was found and removed, False otherwise
        """
        if framework_identifier not in cls._adapters:
            logger.warning("Attempted to unregister non-existent framework: %s", framework_identifier)
            return False

        if output_type is None:
            # Remove all adapters for this framework
            cls._adapters.pop(framework_identifier)
            logger.debug("Unregistered all adapters for framework: %s", framework_identifier)
            return True

        # Remove specific output type adapter
        output_type_name = output_type.__name__
        if output_type_name in cls._adapters[framework_identifier]:
            cls._adapters[framework_identifier].pop(output_type_name)

            # Clean up empty framework entry
            if not cls._adapters[framework_identifier]:
                cls._adapters.pop(framework_identifier)

            logger.debug("Unregistered adapter for framework: %s, output type: %s",
                         framework_identifier,
                         output_type_name)
            return True

        logger.warning("Attempted to unregister non-existent adapter: %s:%s", framework_identifier, output_type_name)
        return False

    @classmethod
    def get_adapter(cls, trace_source: TraceSource, output_type: type[OutputT]) -> TraceSourceAdapter[OutputT] | None:
        """Get the appropriate adapter for a trace source and output type.

        Args:
            trace_source: The trace source to find an adapter for
            output_type: The desired output type
        """
        # Input validation: Ensure required fields are present and valid
        if not trace_source.source.framework or not trace_source.source.provider:
            logger.warning("Invalid trace source: missing framework ('%s') or provider ('%s')",
                           trace_source.source.framework,
                           trace_source.source.provider)
            return None

        framework_provider = f"{trace_source.source.framework}_{trace_source.source.provider}"
        output_type_name = output_type.__name__

        framework_adapters = cls._adapters.get(framework_provider)
        if framework_adapters:
            return framework_adapters.get(output_type_name)  # type: ignore
        return None

    @classmethod
    def list_supported_frameworks(cls) -> list[str]:
        """List all supported framework identifiers."""
        return list(cls._adapters.keys())

    @classmethod
    def list_supported_output_types(cls, framework_identifier: str | None = None) -> list[str]:
        """List all supported output types.

        Args:
            framework_identifier: If provided, list output types for this framework only
        """
        if framework_identifier:
            framework_adapters = cls._adapters.get(framework_identifier, {})
            return list(framework_adapters.keys())
        else:
            # Return all unique output types across all frameworks
            output_types = set()
            for framework_adapters in cls._adapters.values():
                output_types.update(framework_adapters.keys())
            return list(output_types)

    @classmethod
    def list_adapters(cls) -> dict[str, dict[str, str]]:
        """List all registered adapters in a structured format.

        Returns:
            Dict in format: {framework_id: {output_type: adapter_class_name}}
        """
        result = {}
        for framework_id, framework_adapters in cls._adapters.items():
            result[framework_id] = {
                output_type: adapter.__class__.__name__
                for output_type, adapter in framework_adapters.items()
            }
        return result


def register_span_adapter(adapter: TraceSourceAdapter[Any]):
    """Register a custom adapter globally.

    The output type is automatically determined from the adapter's output_type property.

    Args:
        adapter: The adapter to register
    """
    TraceAdapterRegistry.register_adapter(adapter)


def unregister_span_adapter(framework_identifier: str, output_type: type[Any] | None = None) -> bool:
    """Unregister an adapter by framework identifier and output type.

    Args:
        framework_identifier: The framework identifier to unregister
        output_type: The output type to unregister (if None, removes all adapters
                    for this framework)

    Returns:
        True if adapter was found and removed, False otherwise
    """
    return TraceAdapterRegistry.unregister_adapter(framework_identifier, output_type)
