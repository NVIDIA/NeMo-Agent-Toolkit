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
from collections.abc import Callable
from functools import reduce
from typing import Any

from nat.plugins.data_flywheel.observability.schema.trace_container import TraceContainer

logger = logging.getLogger(__name__)


class TraceAdapterRegistry:
    """Registry for trace source to target type conversions.

    Maintains schema detection through Pydantic unions while enabling dynamic registration
    of converter functions for different trace source types.
    """

    _registered_types: dict[type, dict[type, Callable]] = {}  # source_type -> {target_type -> converter}
    _union_cache: Any = None

    @classmethod
    def register_adapter(cls, trace_source_model: type) -> Callable[[Callable], Callable]:
        """Register adapter with a trace source Pydantic model.

        The model defines the schema for union-based detection, allowing automatic
        schema matching without explicit framework/provider specification.

        Args:
            trace_source_model (type): Pydantic model class that defines the trace source schema
                               (e.g., OpenAITraceSource, NIMTraceSource, CustomTraceSource)

        Returns:
            Callable: Decorator function that registers the converter
        """

        def decorator(func):
            return_type = func.__annotations__.get('return')

            # Validate return type annotation exists and is meaningful
            if return_type is None:
                raise ValueError(f"Converter function '{func.__name__}' must have a return type annotation.\n"
                                 f"Example: def {func.__name__}(trace: TraceContainer) -> DFWESRecord:")

            # Initialize nested dict if needed
            if trace_source_model not in cls._registered_types:
                cls._registered_types[trace_source_model] = {}

            # Store converter: source_type -> target_type -> converter_func
            cls._registered_types[trace_source_model][return_type] = func

            # Immediately rebuild union and update TraceContainer model
            cls._rebuild_union()

            logger.debug("Registered %s -> %s converter",
                         trace_source_model.__name__,
                         getattr(return_type, '__name__', str(return_type)))
            return func

        return decorator

    @classmethod
    def convert(cls, trace_container: TraceContainer, to_type: type) -> Any:
        """Convert trace to target type using registered converter function.

        Args:
            trace_container (TraceContainer): TraceContainer with source data to convert
            to_type (type): Target type to convert to

        Returns:
            Converted object of to_type

        Raises:
            ValueError: If no converter is registered for source->target combination
        """
        source_type = type(trace_container.source)

        # Look up converter: source_type -> target_type -> converter_func
        source_converters = cls._registered_types.get(source_type, {})
        converter = source_converters.get(to_type)

        if not converter:
            available_targets = list(source_converters.keys()) if source_converters else []
            available_target_names = [getattr(t, '__name__', str(t)) for t in available_targets]
            raise ValueError(
                f"No converter from {source_type.__name__} to {getattr(to_type, '__name__', str(to_type))}. "
                f"Available targets: {available_target_names}")

        return converter(trace_container)

    @classmethod
    def get_adapter(cls, trace_container: TraceContainer, to_type: type) -> Callable | None:
        """Get the converter function for a given trace source and target type.

        Args:
            trace_container (TraceContainer): TraceContainer with source data
            to_type (type): Target type to convert to

        Returns:
            Converter function if registered, None if not found
        """
        source_type = type(trace_container.source)
        return cls._registered_types.get(source_type, {}).get(to_type)

    @classmethod
    def get_current_union(cls) -> type:
        """Get the current source union with all registered source types.

        Returns:
            type: Union type containing all registered trace source types
        """
        if cls._union_cache is None:
            cls._rebuild_union()
        return cls._union_cache

    @classmethod
    def _rebuild_union(cls):
        """Rebuild the union with all registered trace source types."""

        # Get all registered source types (dictionary keys)
        all_schema_types = set(cls._registered_types.keys())

        # Create union from source types (used for Pydantic schema detection)
        if len(all_schema_types) == 0:
            # No types registered yet - use Any as permissive fallback
            cls._union_cache = Any
        elif len(all_schema_types) == 1:
            cls._union_cache = next(iter(all_schema_types))
        else:
            # Sort types by name to ensure consistent order
            sorted_types = sorted(all_schema_types, key=lambda t: t.__name__)
            # Create Union from multiple types using reduce
            cls._union_cache = reduce(lambda a, b: a | b, sorted_types)

        logger.debug("Rebuilt source union with %d registered source types: %s",
                     len(all_schema_types), [t.__name__ for t in all_schema_types])

        # Update TraceContainer model with new union
        cls._update_trace_source_model()

    @classmethod
    def _update_trace_source_model(cls):
        """Update the TraceContainer model to use the current dynamic union."""
        try:
            # Update the source field annotation to use current union
            if hasattr(TraceContainer, '__annotations__'):
                TraceContainer.__annotations__['source'] = cls._union_cache

                # Force Pydantic to rebuild the model with new annotations
                TraceContainer.model_rebuild()
                logger.debug("Updated TraceContainer model with new union type")
        except Exception as e:
            logger.warning("Failed to update TraceContainer model: %s", e)

    @classmethod
    def unregister_adapter(cls, source_type: type, target_type: type) -> bool:
        """Unregister a specific adapter.

        Args:
            source_type (type): The trace source type
            target_type (type): The target conversion type

        Returns:
            bool: True if adapter was found and removed, False if not found
        """
        if source_type not in cls._registered_types:
            return False

        target_converters = cls._registered_types[source_type]
        if target_type not in target_converters:
            return False

        # Remove the specific converter
        del target_converters[target_type]

        # Clean up empty source entry
        if not target_converters:
            del cls._registered_types[source_type]

        # Rebuild union since registered types changed
        cls._rebuild_union()

        logger.debug("Unregistered %s -> %s converter",
                     source_type.__name__,
                     getattr(target_type, '__name__', str(target_type)))
        return True

    @classmethod
    def unregister_all_adapters(cls, source_type: type) -> int:
        """Unregister all adapters for a given source type.

        Args:
            source_type (type): The trace source type to remove all converters for

        Returns:
            int: Number of converters removed
        """
        if source_type not in cls._registered_types:
            return 0

        removed_count = len(cls._registered_types[source_type])
        del cls._registered_types[source_type]

        # Rebuild union since registered types changed
        cls._rebuild_union()

        logger.debug("Unregistered all %d converters for %s", removed_count, source_type.__name__)
        return removed_count

    @classmethod
    def clear_registry(cls) -> int:
        """Clear all registered adapters. Useful for testing cleanup.

        Returns:
            int: Total number of converters removed
        """
        total_removed = sum(len(converters) for converters in cls._registered_types.values())
        cls._registered_types.clear()
        cls._union_cache = None

        # Rebuild union (will be empty now)
        cls._rebuild_union()

        logger.debug("Cleared registry - removed %d total converters", total_removed)
        return total_removed

    @classmethod
    def list_registered_types(cls) -> dict[type, dict[type, Callable]]:
        """List all registered conversions: source_type -> {target_type -> converter}.

        Returns:
            dict[type, dict[type, Callable]]: Nested dict mapping source types to their available target conversions
        """
        return cls._registered_types


# Convenience functions for adapter management
register_adapter = TraceAdapterRegistry.register_adapter
unregister_adapter = TraceAdapterRegistry.unregister_adapter
unregister_all_adapters = TraceAdapterRegistry.unregister_all_adapters
clear_registry = TraceAdapterRegistry.clear_registry
