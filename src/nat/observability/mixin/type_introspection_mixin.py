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

import inspect
import logging
import types
from functools import lru_cache
from typing import Any
from typing import TypeVar
from typing import get_args
from typing import get_origin

from pydantic import BaseModel
from pydantic import ValidationError
from pydantic import create_model
from pydantic.fields import FieldInfo

from nat.utils.type_utils import DecomposedType

logger = logging.getLogger(__name__)


class TypeIntrospectionMixin:
    """Hybrid mixin class providing type introspection capabilities for generic classes.

    This mixin combines the DecomposedType class utilities with MRO traversal
    to properly handle complex inheritance chains like HeaderRedactionProcessor or ProcessingExporter.
    """

    def _extract_types_from_signature_method(self) -> tuple[type[Any], type[Any]] | None:
        """Extract input/output types from the signature method.

        This method looks for a signature method (either defined via _signature_method class
        attribute or discovered generically) and extracts input/output types from
        its method signature.

        Returns:
            tuple[type[Any], type[Any]] | None: (input_type, output_type) or None if not found.
        """
        # First, try to get the signature method name from the class
        signature_method_name = getattr(self.__class__, '_signature_method', None)

        # If not defined, try to discover it generically
        if not signature_method_name:
            signature_method_name = self._discover_signature_method()

        if not signature_method_name:
            return None

        # Get the method and inspect its signature
        try:
            method = getattr(self, signature_method_name)
            sig = inspect.signature(method)

            # Find the first parameter that's not 'self'
            params = list(sig.parameters.values())
            input_param = None
            for param in params:
                if param.name != 'self':
                    input_param = param
                    break

            if not input_param or input_param.annotation == inspect.Parameter.empty:
                return None

            # Get return type
            return_annotation = sig.return_annotation
            if return_annotation == inspect.Signature.empty:
                return None

            input_type = input_param.annotation
            output_type = return_annotation

            # Resolve any TypeVars if needed (including nested ones)
            if isinstance(input_type, TypeVar) or isinstance(
                    output_type, TypeVar) or self._contains_typevar(input_type) or self._contains_typevar(output_type):
                # Try to resolve using the MRO approach as fallback
                typevar_mapping = self._build_typevar_mapping()
                input_type = self._resolve_typevar_recursively(input_type, typevar_mapping)
                output_type = self._resolve_typevar_recursively(output_type, typevar_mapping)

            # Only return if we have concrete types
            if not isinstance(input_type, TypeVar) and not isinstance(output_type, TypeVar):
                return input_type, output_type

        except (AttributeError, TypeError) as e:
            logger.debug("Failed to extract types from signature method '%s': %s", signature_method_name, e)

        return None

    def _discover_signature_method(self) -> str | None:
        """Discover any method suitable for type introspection.

        Looks for any method with the signature pattern: method(self, param: Type) -> ReturnType
        Any method matching this pattern is functionally equivalent for type introspection purposes.

        Returns:
            str | None: Method name or None if not found
        """
        # Look through all methods to find ones that match the input/output pattern
        candidates = []

        for cls in self.__class__.__mro__:
            for name, method in inspect.getmembers(cls, inspect.isfunction):
                # Skip private methods except dunder methods
                if name.startswith('_') and not name.startswith('__'):
                    continue

                # Skip methods that were defined in TypeIntrospectionMixin
                if hasattr(method, '__qualname__') and 'TypeIntrospectionMixin' in method.__qualname__:
                    logger.debug("Skipping method '%s' defined in TypeIntrospectionMixin", name)
                    continue

                # Let signature analysis determine suitability - method names don't matter
                try:
                    sig = inspect.signature(method)
                    params = list(sig.parameters.values())

                    # Look for methods with exactly one non-self parameter and a return annotation
                    non_self_params = [p for p in params if p.name != 'self']
                    if (len(non_self_params) == 1 and non_self_params[0].annotation != inspect.Parameter.empty
                            and sig.return_annotation != inspect.Signature.empty):

                        # Prioritize abstract methods
                        is_abstract = getattr(method, '__isabstractmethod__', False)
                        candidates.append((name, is_abstract, cls))

                except (TypeError, ValueError) as e:
                    logger.debug("Failed to inspect signature of method '%s': %s", name, e)

        if not candidates:
            logger.debug("No candidates found for signature method")
            return None

        # Any method with the right signature will work for type introspection
        # Prioritize abstract methods if available, otherwise use the first valid one
        candidates.sort(key=lambda x: not x[1])  # Abstract methods first
        return candidates[0][0]

    def _resolve_typevar_recursively(self, type_arg: Any, typevar_mapping: dict[TypeVar, type[Any]]) -> Any:
        """Recursively resolve TypeVars within complex types.

        Args:
            type_arg (Any): The type argument to resolve (could be a TypeVar, generic type, etc.)
            typevar_mapping (dict[TypeVar, type[Any]]): Current mapping of TypeVars to concrete types

        Returns:
            Any: The resolved type with all TypeVars substituted
        """
        # If it's a TypeVar, resolve it
        if isinstance(type_arg, TypeVar):
            return typevar_mapping.get(type_arg, type_arg)

        # If it's a generic type, decompose and resolve its arguments
        try:
            decomposed = DecomposedType(type_arg)
            if decomposed.is_generic and decomposed.args:
                # Recursively resolve all type arguments
                resolved_args = []
                for arg in decomposed.args:
                    resolved_arg = self._resolve_typevar_recursively(arg, typevar_mapping)
                    resolved_args.append(resolved_arg)

                # Reconstruct the generic type with resolved arguments
                if decomposed.origin:
                    return decomposed.origin[tuple(resolved_args)]

        except (TypeError, AttributeError) as e:
            # If we can't decompose or reconstruct, return as-is
            logger.debug("Failed to decompose or reconstruct type '%s': %s", type_arg, e)

        return type_arg

    def _contains_typevar(self, type_arg: Any) -> bool:
        """Check if a type contains any TypeVars (including nested ones).

        Args:
            type_arg (Any): The type to check

        Returns:
            bool: True if the type contains any TypeVars
        """
        if isinstance(type_arg, TypeVar):
            return True

        try:
            decomposed = DecomposedType(type_arg)
            if decomposed.is_generic and decomposed.args:
                return any(self._contains_typevar(arg) for arg in decomposed.args)
        except (TypeError, AttributeError) as e:
            logger.debug("Failed to decompose or reconstruct type '%s': %s", type_arg, e)

        return False

    def _build_typevar_mapping(self) -> dict[TypeVar, type[Any]]:
        """Build TypeVar to concrete type mapping from MRO traversal.

        Returns:
            dict[TypeVar, type[Any]]: Mapping of TypeVars to concrete types
        """
        typevar_mapping = {}

        # First, check if the instance has concrete type arguments from __orig_class__
        # This handles cases like BatchingProcessor[str]() where we need to map T -> str
        orig_class = getattr(self, '__orig_class__', None)
        if orig_class:
            class_origin = get_origin(orig_class)
            class_args = get_args(orig_class)
            class_params = getattr(class_origin, '__parameters__', None)

            if class_args and class_params:
                # Map class-level TypeVars to their concrete arguments
                for param, arg in zip(class_params, class_args):
                    typevar_mapping[param] = arg

        # Then traverse the MRO to build the complete mapping
        for cls in self.__class__.__mro__:
            for base in getattr(cls, '__orig_bases__', []):
                decomposed_base = DecomposedType(base)

                if (decomposed_base.is_generic and decomposed_base.origin
                        and hasattr(decomposed_base.origin, '__parameters__')):
                    type_params = decomposed_base.origin.__parameters__
                    # Map each TypeVar to its concrete argument
                    for param, arg in zip(type_params, decomposed_base.args):
                        if param not in typevar_mapping:  # Keep the most specific mapping
                            # If arg is also a TypeVar, try to resolve it
                            if isinstance(arg, TypeVar) and arg in typevar_mapping:
                                typevar_mapping[param] = typevar_mapping[arg]
                            else:
                                typevar_mapping[param] = arg

        return typevar_mapping

    def _extract_instance_types_from_mro(self) -> tuple[type[Any], type[Any]] | None:
        """Extract Generic[InputT, OutputT] types by traversing the MRO.

        This handles complex inheritance chains by looking for the base
        class and resolving TypeVars through the inheritance hierarchy.

        Returns:
            tuple[type[Any], type[Any]] | None: (input_type, output_type) or None if not found
        """
        # Use the centralized TypeVar mapping
        typevar_mapping = self._build_typevar_mapping()

        # Now find the first generic base with exactly 2 parameters, starting from the base classes
        # This ensures we get the fundamental input/output types rather than specialized ones
        for cls in reversed(self.__class__.__mro__):
            for base in getattr(cls, '__orig_bases__', []):
                decomposed_base = DecomposedType(base)

                # Look for any generic with exactly 2 parameters (likely InputT, OutputT pattern)
                if decomposed_base.is_generic and len(decomposed_base.args) == 2:
                    input_type = decomposed_base.args[0]
                    output_type = decomposed_base.args[1]

                    # Resolve TypeVars to concrete types using recursive resolution
                    input_type = self._resolve_typevar_recursively(input_type, typevar_mapping)
                    output_type = self._resolve_typevar_recursively(output_type, typevar_mapping)

                    # Only return if we have concrete types (not TypeVars)
                    if not isinstance(input_type, TypeVar) and not isinstance(output_type, TypeVar):
                        return input_type, output_type

        return None

    @lru_cache
    def _extract_input_output_types(self) -> tuple[type[Any], type[Any]]:
        """Extract both input and output types using available approaches.

        Returns:
            tuple[type[Any], type[Any]]: (input_type, output_type)

        Raises:
            ValueError: If types cannot be extracted
        """
        # First try the signature-based approach
        result = self._extract_types_from_signature_method()
        if result:
            return result

        # Fallback to MRO-based approach for complex inheritance
        result = self._extract_instance_types_from_mro()
        if result:
            return result

        raise ValueError(f"Could not extract input/output types from {self.__class__.__name__}. "
                         f"Ensure class inherits from a generic like Processor[InputT, OutputT] "
                         f"or has a signature method with type annotations")

    @property
    def input_type(self) -> type[Any]:
        """Get the input type of the instance.

        Returns:
            type[Any]: The input type
        """
        return self._extract_input_output_types()[0]

    @property
    def output_type(self) -> type[Any]:
        """Get the output type of the instance.

        Returns:
            type[Any]: The output type
        """
        return self._extract_input_output_types()[1]

    @lru_cache
    def _get_union_info(self, type_obj: type[Any]) -> tuple[bool, tuple[type, ...] | None]:
        """Get union information for a type.

        Args:
            type_obj (type[Any]): The type to analyze

        Returns:
            tuple[bool, tuple[type, ...] | None]: (is_union, union_types_or_none)
        """
        decomposed = DecomposedType(type_obj)
        return decomposed.is_union, decomposed.args if decomposed.is_union else None

    @property
    def has_union_input(self) -> bool:
        """Check if the input type is a union type.

        Returns:
            bool: True if the input type is a union type, False otherwise
        """
        return self._get_union_info(self.input_type)[0]

    @property
    def has_union_output(self) -> bool:
        """Check if the output type is a union type.

        Returns:
            bool: True if the output type is a union type, False otherwise
        """
        return self._get_union_info(self.output_type)[0]

    @property
    def input_union_types(self) -> tuple[type, ...] | None:
        """Get the individual types in an input union.

        Returns:
            tuple[type, ...] | None: The individual types in an input union or None if not found
        """
        return self._get_union_info(self.input_type)[1]

    @property
    def output_union_types(self) -> tuple[type, ...] | None:
        """Get the individual types in an output union.

        Returns:
            tuple[type, ...] | None: The individual types in an output union or None if not found
        """
        return self._get_union_info(self.output_type)[1]

    def is_compatible_with_input(self, source_type: type) -> bool:
        """Check if a source type is compatible with this instance's input type.

        Uses Pydantic-based type compatibility checking for strict type matching.
        This focuses on proper type relationships rather than batch compatibility.

        Args:
            source_type (type): The source type to check

        Returns:
            bool: True if the source type is compatible with the input type, False otherwise
        """
        return self._is_pydantic_type_compatible(source_type, self.input_type)

    def is_output_compatible_with(self, target_type: type) -> bool:
        """Check if this instance's output type is compatible with a target type.

        Uses Pydantic-based type compatibility checking for strict type matching.
        This focuses on proper type relationships rather than batch compatibility.

        Args:
            target_type (type): The target type to check

        Returns:
            bool: True if the output type is compatible with the target type, False otherwise
        """
        return self._is_pydantic_type_compatible(self.output_type, target_type)

    def _is_pydantic_type_compatible(self, source_type: type, target_type: type) -> bool:
        """Check strict type compatibility without batch compatibility hacks.

        This focuses on proper type relationships: exact matches and subclass relationships.

        Args:
            source_type (type): The source type to check
            target_type (type): The target type to check compatibility with

        Returns:
            bool: True if types are compatible, False otherwise
        """
        # Direct equality check (most common case)
        if source_type == target_type:
            return True

        # Subclass relationship check
        try:
            if issubclass(source_type, target_type):
                return True
        except TypeError:
            # Generic types can't use issubclass, they're only compatible if equal
            logger.debug("Generic type %s cannot be used with issubclass, they're only compatible if equal",
                         source_type)

        return False

    @lru_cache
    def _get_input_validator(self) -> type[BaseModel]:
        """Create a Pydantic model for validating input types.

        Returns:
            type[BaseModel]: The Pydantic model for validating input types
        """
        input_type = self.input_type
        return create_model(f"{self.__class__.__name__}InputValidator", input=(input_type, FieldInfo()))

    @lru_cache
    def _get_output_validator(self) -> type[BaseModel]:
        """Create a Pydantic model for validating output types.

        Returns:
            type[BaseModel]: The Pydantic model for validating output types
        """
        output_type = self.output_type
        return create_model(f"{self.__class__.__name__}OutputValidator", output=(output_type, FieldInfo()))

    def validate_input_type(self, item: Any) -> bool:
        """Validate that an item matches the expected input type using Pydantic.

        Args:
            item (Any): The item to validate

        Returns:
            bool: True if the item matches the input type, False otherwise
        """
        try:
            validator = self._get_input_validator()
            validator(input=item)
            return True
        except ValidationError:
            logger.warning("Item %s is not compatible with input type %s", item, self.input_type)
            return False

    def validate_output_type(self, item: Any) -> bool:
        """Validate that an item matches the expected output type using Pydantic.

        Args:
            item (Any): The item to validate

        Returns:
            bool: True if the item matches the output type, False otherwise
        """
        try:
            validator = self._get_output_validator()
            validator(output=item)
            return True
        except ValidationError:
            logger.warning("Item %s is not compatible with output type %s", item, self.output_type)
            return False

    @lru_cache
    def extract_non_optional_type(self, type_obj: type | types.UnionType) -> Any:
        """Extract the non-None type from Optional[T] or Union[T, None] types.

        This is useful when you need to pass a type to a system that doesn't
        understand Optional types (like registries that expect concrete types).

        Args:
            type_obj (type | types.UnionType): The type to extract from (could be Optional[T] or Union[T, None])

        Returns:
            Any: The actual type without None, or the original type if not a union with None
        """
        decomposed = DecomposedType(type_obj)  # type: ignore[arg-type]
        if decomposed.is_optional:
            return decomposed.get_optional_type().type
        return type_obj
