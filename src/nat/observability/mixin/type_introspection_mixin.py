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

from functools import lru_cache
from typing import Any
from typing import get_args
from typing import get_origin


class TypeIntrospectionMixin:
    """Mixin class providing type introspection capabilities for generic classes.

    This mixin extracts type information from generic class definitions,
    allowing classes to determine their InputT and OutputT types at runtime.
    """

    def _find_generic_types(self) -> tuple[type[Any], type[Any]] | None:
        """
        Recursively search through the inheritance hierarchy to find generic type parameters.

        This method handles cases where a class inherits from a generic parent class,
        resolving the concrete types through the inheritance chain.

        Returns:
            tuple[type[Any], type[Any]] | None: (input_type, output_type) if found, None otherwise
        """
        # First, try to find types directly in this class's __orig_bases__
        for base_cls in getattr(self.__class__, '__orig_bases__', []):
            base_cls_args = get_args(base_cls)

            # Direct case: MyClass[InputT, OutputT]
            if len(base_cls_args) >= 2:
                return base_cls_args[0], base_cls_args[1]

            # Indirect case: MyClass[SomeGeneric[ConcreteType]]
            # Need to resolve the generic parent's types
            if len(base_cls_args) == 1:
                base_origin = get_origin(base_cls)
                if base_origin and hasattr(base_origin, '__orig_bases__'):
                    # Look at the parent's generic definition
                    for parent_base in getattr(base_origin, '__orig_bases__', []):
                        parent_args = get_args(parent_base)
                        if len(parent_args) >= 2:
                            # Found the pattern: ParentClass[T, list[T]]
                            # Substitute T with our concrete type
                            concrete_type = base_cls_args[0]
                            input_type = self._substitute_type_var(parent_args[0], concrete_type)
                            output_type = self._substitute_type_var(parent_args[1], concrete_type)
                            return input_type, output_type

        return None

    def _substitute_type_var(self, type_expr: Any, concrete_type: type) -> type[Any]:
        """
        Substitute TypeVar in a type expression with a concrete type.

        Args:
            type_expr: The type expression potentially containing TypeVars
            concrete_type: The concrete type to substitute

        Returns:
            The type expression with TypeVars substituted
        """
        from typing import TypeVar

        # If it's a TypeVar, substitute it
        if isinstance(type_expr, TypeVar):
            return concrete_type

        # If it's a generic type like list[T], substitute the args
        origin = get_origin(type_expr)
        args = get_args(type_expr)

        if origin and args:
            # Recursively substitute in the arguments
            new_args = tuple(self._substitute_type_var(arg, concrete_type) for arg in args)
            # Reconstruct the generic type
            return origin[new_args]

        # Otherwise, return as-is
        return type_expr

    @property
    @lru_cache
    def input_type(self) -> type[Any]:
        """
        Get the input type of the class. The input type is determined by the generic parameters of the class.

        For example, if a class is defined as `MyClass[list[int], str]`, the `input_type` is `list[int]`.

        Returns
        -------
        type[Any]
            The input type specified in the generic parameters

        Raises
        ------
        ValueError
            If the input type cannot be determined from the class definition
        """
        types = self._find_generic_types()
        if types:
            return types[0]

        raise ValueError(f"Could not find input type for {self.__class__.__name__}")

    @property
    @lru_cache
    def output_type(self) -> type[Any]:
        """
        Get the output type of the class. The output type is determined by the generic parameters of the class.

        For example, if a class is defined as `MyClass[list[int], str]`, the `output_type` is `str`.

        Returns
        -------
        type[Any]
            The output type specified in the generic parameters

        Raises
        ------
        ValueError
            If the output type cannot be determined from the class definition
        """
        types = self._find_generic_types()
        if types:
            return types[1]

        raise ValueError(f"Could not find output type for {self.__class__.__name__}")

    @property
    @lru_cache
    def input_class(self) -> type:
        """
        Get the python class of the input type. This is the class that can be used to check if a value is an
        instance of the input type. It removes any generic or annotation information from the input type.

        For example, if the input type is `list[int]`, the `input_class` is `list`.

        Returns
        -------
        type
            The python type of the input type
        """
        input_origin = get_origin(self.input_type)

        if input_origin is None:
            return self.input_type

        return input_origin

    @property
    @lru_cache
    def output_class(self) -> type:
        """
        Get the python class of the output type. This is the class that can be used to check if a value is an
        instance of the output type. It removes any generic or annotation information from the output type.

        For example, if the output type is `list[int]`, the `output_class` is `list`.

        Returns
        -------
        type
            The python type of the output type
        """
        output_origin = get_origin(self.output_type)

        if output_origin is None:
            return self.output_type

        return output_origin
