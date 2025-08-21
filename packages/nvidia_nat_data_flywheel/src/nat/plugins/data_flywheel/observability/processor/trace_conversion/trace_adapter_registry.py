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
from nat.utils.type_converter import GlobalTypeConverter

logger = logging.getLogger(__name__)


class TraceAdapterRegistry:
    """Registry for dynamically creating concrete TraceContainer types and updating unions.

    Maintains schema detection through Pydantic unions while enabling dynamic registration.
    """

    _registered_types: dict[type, type] = {}  # model_class -> concrete_type
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

            # Create unique concrete type for GlobalTypeConverter distinction
            type_name = f"{trace_source_model.__name__}TraceContainer"
            concrete_type = type(type_name, (TraceContainer, ),
                                 {
                                     '_source_model': trace_source_model,
                                     '__module__': func.__module__,
                                     '__qualname__': type_name,
                                 })

            # Store the mapping
            cls._registered_types[trace_source_model] = concrete_type

            # Immediately rebuild union and update TraceContainer model
            cls._rebuild_union()

            # Create converter function with concrete type signature
            def typed_converter(trace_source: concrete_type):
                """Schema-specific converter with unique input type."""
                return func(trace_source)

            # Set proper function metadata
            typed_converter.__name__ = f"convert_{trace_source_model.__name__}"
            typed_converter.__qualname__ = typed_converter.__name__
            typed_converter.__annotations__ = {
                'trace_source': concrete_type, 'return': func.__annotations__.get('return', type(None))
            }

            # Register with GlobalTypeConverter
            GlobalTypeConverter.register_converter(typed_converter)
            logger.debug("Registered converter: %s for model %s with type %s",
                         typed_converter.__name__,
                         trace_source_model.__name__,
                         concrete_type.__name__)

            return func

        return decorator

    @classmethod
    def get_current_union(cls) -> type:
        """Get the current source union with all registered types.

        Returns:
            type: Union type containing all registered concrete types plus original types
        """
        if cls._union_cache is None:
            cls._rebuild_union()
        return cls._union_cache

    @classmethod
    def _rebuild_union(cls):
        """Rebuild the union with all registered schema models."""

        # Start empty - all types added through registration
        all_schema_types = set()

        # Add all registered schema models for union detection
        all_schema_types.update(cls._registered_types.keys())

        # Create union from schema types (these are used for schema detection)
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

        logger.debug("Rebuilt source union with %d registered schema types: %s",
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
    def create_dynamic_instance(cls, trace_container: TraceContainer) -> TraceContainer:
        """Convert a TraceContainer to the appropriate dynamic type.

        Args:
            trace_container (TraceContainer): The base TraceContainer instance

        Returns:
            TraceContainer: The dynamic type instance

        Raises:
            ValueError: If no adapter is registered for the trace source schema type
            RuntimeError: If dynamic type creation fails
        """
        # Check if we have a registered dynamic type for this trace source schema
        if trace_container.source.__class__ in cls._registered_types:
            dynamic_type = cls._registered_types[trace_container.source.__class__]
            logger.debug("Converting to dynamic type %s for %s",
                         dynamic_type.__name__,
                         trace_container.__class__.__name__)

            # Create instance of the dynamic type with the same data
            try:
                return dynamic_type(source=trace_container.source, span=trace_container.span)
            except Exception as e:
                logger.error("Failed to create dynamic type %s: %s", dynamic_type.__name__, e)
                raise RuntimeError(
                    f"Dynamic type creation failed for {trace_container.__class__.__name__}: {dynamic_type.__name__}"
                ) from e
        else:
            logger.error("No dynamic type registered for schema %s", trace_container.source.__class__.__name__)
            raise ValueError(f"No adapter registered for schema {trace_container.source.__class__.__name__}")

    @classmethod
    def list_registered_types(cls) -> dict[type, type]:
        """List all registered trace source schema types and their concrete container types.

        Returns:
            dict[type, type]: Dict mapping trace source model types to concrete container types
        """
        return cls._registered_types


# Convenience function for adapter registration
register_adapter = TraceAdapterRegistry.register_adapter
