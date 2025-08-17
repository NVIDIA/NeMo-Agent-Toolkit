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
from functools import reduce
from typing import Any

from nat.plugins.data_flywheel.observability.schema.trace_container import TraceContainer
from nat.utils.type_converter import GlobalTypeConverter

logger = logging.getLogger(__name__)


class TraceAdapterRegistry:
    """Registry for dynamically creating concrete TraceContainer types and updating unions.

    Maintains schema detection through Pydantic unions while enabling dynamic registration.
    """

    _registered_types: dict[tuple[str, str], type] = {}  # (framework, provider) -> concrete_type
    _registered_models: dict[type, tuple[str, str]] = {}  # model_class -> (framework, provider)
    _union_cache: Any = None

    @classmethod
    def register_adapter(cls, trace_source_model: type):
        """Register adapter with a trace source Pydantic model.

        The model defines the schema for union-based detection and provides
        framework/provider information.

        Args:
            trace_source_model: Pydantic model class that defines the trace source schema
                               (e.g., OpenAITraceContainer, NIMTraceContainer, CustomTraceContainer)

        Returns:
            Decorator function that registers the converter
        """

        def decorator(func):
            # Extract framework and provider from the model
            framework, provider = cls._extract_framework_provider(trace_source_model)

            # Create unique concrete type for GlobalTypeConverter distinction
            type_name = f"{framework.title()}{provider.title()}TraceContainer"
            concrete_type = type(
                type_name, (TraceContainer, ),
                {
                    '_framework': framework,
                    '_provider': provider,
                    '_source_model': trace_source_model,
                    '__module__': func.__module__,
                    '__qualname__': type_name,
                })

            # Store the mapping
            cls._registered_types[(framework, provider)] = concrete_type
            cls._registered_models[trace_source_model] = (framework, provider)

            # Immediately rebuild union and update TraceContainer model
            cls._rebuild_union()

            # Create converter function with concrete type signature
            def typed_converter(trace_source: concrete_type):
                """Schema-specific converter with unique input type."""
                return func(trace_source)

            # Set proper function metadata
            typed_converter.__name__ = f"convert_{framework}_{provider}"
            typed_converter.__qualname__ = typed_converter.__name__
            typed_converter.__annotations__ = {
                'trace_source': concrete_type, 'return': func.__annotations__.get('return', type(None))
            }

            # Register with GlobalTypeConverter
            GlobalTypeConverter.register_converter(typed_converter)
            logger.debug("Registered converter: %s for model %s (%s+%s) with type %s",
                         typed_converter.__name__,
                         trace_source_model.__name__,
                         framework,
                         provider,
                         type_name)

            return func

        return decorator

    @classmethod
    def _extract_framework_provider(cls, model: type) -> tuple[str, str]:
        """Extract framework and provider from a trace source model.

        Args:
            model: Pydantic model class

        Returns:
            Tuple of (framework, provider)
        """
        # Try to get default values from the model fields
        extracted_framework = "unknown"
        extracted_provider = "unknown"

        if hasattr(model, 'model_fields'):
            # Pydantic v2 approach
            if 'framework' in model.model_fields:  # type: ignore
                fw_field = model.model_fields['framework']  # type: ignore
                if hasattr(fw_field, 'default') and fw_field.default is not None:
                    extracted_framework = str(fw_field.default).lower().replace('llmframeworkenum.', '')

            if 'provider' in model.model_fields:  # type: ignore
                prov_field = model.model_fields['provider']  # type: ignore
                if hasattr(prov_field, 'default') and prov_field.default is not None:
                    extracted_provider = getattr(prov_field.default, 'value', str(prov_field.default))

        logger.debug("Extracted framework='%s', provider='%s' from %s",
                     extracted_framework,
                     extracted_provider,
                     model.__name__)
        return extracted_framework, extracted_provider

    @classmethod
    def get_current_union(cls):
        """Get the current source union with all registered types.

        Returns:
            Union type containing all registered concrete types plus original types
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
        all_schema_types.update(cls._registered_models.keys())

        # Create union from schema types (these are used for schema detection)
        if len(all_schema_types) == 0:
            # No types registered yet - use Any as permissive fallback
            cls._union_cache = Any
        elif len(all_schema_types) == 1:
            cls._union_cache = next(iter(all_schema_types))
        else:
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
    def create_dynamic_instance(cls, trace_source: TraceContainer, framework: str, provider: str) -> TraceContainer:
        """Convert a TraceContainer to the appropriate dynamic type if registered.

        Args:
            trace_source: The base TraceContainer instance
            framework: Framework name
            provider: Provider name

        Returns:
            TraceContainer: Either the dynamic type instance or the original trace_source
        """
        # Check if we have a registered dynamic type for this framework+provider
        if (framework, provider) in cls._registered_types:
            dynamic_type = cls._registered_types[(framework, provider)]
            logger.debug("Converting to dynamic type %s for %s+%s", dynamic_type.__name__, framework, provider)

            # Create instance of the dynamic type with the same data
            try:
                return dynamic_type(source=trace_source.source, span=trace_source.span)
            except Exception as e:
                logger.warning("Failed to create dynamic type %s: %s", dynamic_type.__name__, e)
                return trace_source
        else:
            logger.debug("No dynamic type registered for %s+%s", framework, provider)
            return trace_source

    @classmethod
    def list_registered_types(cls) -> dict[tuple[str, str], str]:
        """List all registered framework+provider combinations.

        Returns:
            Dict mapping (framework, provider) tuples to type names
        """
        return {key: type_obj.__name__ for key, type_obj in cls._registered_types.items()}


# Convenience function for the new approach
register_adapter = TraceAdapterRegistry.register_adapter
