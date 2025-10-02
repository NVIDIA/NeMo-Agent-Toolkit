# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import collections
import collections.abc
import inspect
import json
import os
import sys
import types
import typing
from functools import lru_cache
from typing import TypeAlias

from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model
from pydantic_core import PydanticUndefined

# Mimic the `StrPath` type alias from the `typeshed` package. We can't import it directly because it's not available at
# runtime and causes problems
StrPath: TypeAlias = str | os.PathLike[str]

ClassInfo: TypeAlias = type | types.UnionType | tuple["ClassInfo", ...]


# utility for check if string is a valid json string
def is_valid_json(string):
    try:
        input_str = string.replace("'", "\"")
        json.loads(input_str)
        return True
    except json.JSONDecodeError:
        return False


# A compatibility layer for typing.override decorator.
# In Python >= 3.12, it uses the built-in typing.override decorator
# In Python < 3.12, it acts as a no-op decorator
if sys.version_info >= (3, 12):
    from typing import override
else:

    def override(func):
        return func


class DecomposedType:

    def __init__(self, original: type):

        if (inspect.Signature.empty == original):
            original = types.NoneType

        self.type = original

    @property
    @lru_cache
    def origin(self):
        """
        Get the origin of the current type using `typing.get_origin`. For example, if the current type is `list[int]`,
        the origin would be `list`.

        Returns
        -------
        type
            The origin of the current type.
        """

        return typing.get_origin(self.type)

    @property
    @lru_cache
    def args(self):
        """
        Get the arguments of the current type using `typing.get_args`. For example, if the current type is `list[int,
        str]`, the arguments would be `[int, str]`.

        Returns
        -------
        tuple[type]
            The arguments of the current type.
        """

        return typing.get_args(self.type)

    @property
    @lru_cache
    def root(self):
        """
        Get the root type of the current type. This is the type without any annotations or async generators.

        Returns
        -------
        type
            The root type of the current type.
        """

        return self.origin if self.origin is not None else self.type

    @property
    @lru_cache
    def is_empty(self):
        """
        Check if the current type is eqivalent to `NoneType`.

        Returns
        -------
        bool
            True if the current type is `NoneType`, False otherwise.
        """
        return self.type is types.NoneType

    @property
    @lru_cache
    def is_class(self):
        """
        Check if the current type is a class using `inspect.isclass`. For example, `list[int]` would return False, but
        `list` would return True.

        Returns
        -------
        bool
            True if the current type is a class, False otherwise.
        """

        return inspect.isclass(self.type)

    @property
    @lru_cache
    def is_generic(self):
        """
        Check if the current type is a generic using `typing.GenericMeta`. For example, `list[int]` would return True,
        but `list` would return False.

        Returns
        -------
        bool
            True if the current type is a generic, False otherwise.
        """

        return self.origin is not None

    @property
    @lru_cache
    def is_annotated(self):
        """
        Check if the current type is an annotated type using `typing.Annotated`. For example, `Annotated[int, str]`
        would return True, but `int` would return False.

        Returns
        -------
        bool
            True if the current type is an annotated type, False otherwise.
        """

        return self.origin is typing.Annotated

    @property
    @lru_cache
    def is_union(self):
        """
        Check if the current type is a union type using `typing.Union`. For example, `Union[int, str]` would return
        True, but `int` would return False.

        Returns
        -------
        bool
            True if the current type is a union type, False otherwise.
        """

        return self.origin in (typing.Union, types.UnionType)

    @property
    @lru_cache
    def is_async_generator(self):
        """
        Check if the current type is an async generator type. For example, `AsyncGenerator[int]` would return True,
        but `int` would return False.

        Returns
        -------
        bool
            True if the current type is an async generator type, False otherwise.
        """

        return self.origin in (
            typing.AsyncGenerator,
            collections.abc.AsyncGenerator,
            types.AsyncGeneratorType,
        )

    @property
    @lru_cache
    def is_optional(self):
        """
        Check if the current type is an optional type. For example, `Optional[int]` and `int | None` would return True,
        but `int` would return False.

        Returns
        -------
        bool
            True if the current type is an optional type, False otherwise.
        """

        return self.is_union and types.NoneType in self.args

    @property
    @lru_cache
    def has_base_type(self):
        """
        Check if the current type has a base type, ignoring any annotations or async generators.
        """

        return self.is_annotated or self.is_async_generator

    def get_optional_type(self) -> "DecomposedType":
        """
        If the current type is optional, return the type that is not `NoneType`. If the current type is not optional,
        raise a `ValueError`.

        Returns
        -------
        DecomposedType
            The optional type that is not `NoneType`.

        Raises
        ------
        ValueError
            If the current type is not optional.
        ValueError
            If the current type is optional but has more than one argument that is not `NoneType`.
        """

        if (not self.is_optional):
            raise ValueError(f"Type {self.type} is not optional.")

        remaining_args = tuple(arg for arg in self.args if arg is not types.NoneType)

        if (len(remaining_args) > 1):
            return DecomposedType(typing.Union[*remaining_args])
        if (len(remaining_args) == 1):
            return DecomposedType(remaining_args[0])

        raise ValueError(f"Type {self.type} is not optional.")

    def get_annotated_type(self) -> "DecomposedType":
        """
        If the current type is annotated, return the annotated type. If the current type is not annotated, raise a
        `ValueError`.

        Returns
        -------
        DecomposedType
            The annotated type.

        Raises
        ------
        ValueError
            If the current type is not annotated.
        """

        if (not self.is_annotated):
            raise ValueError(f"Type {self.type} is not annotated.")

        return DecomposedType(self.args[0])

    def get_async_generator_type(self) -> "DecomposedType":
        """
        If the current type is an async generator, return the async generator type. If the current type is not an async
        generator, raise a `ValueError`.

        Returns
        -------
        DecomposedType
            The async generator type.

        Raises
        ------
        ValueError
            If the current type is not an async generator.
        """

        if (not self.is_async_generator):
            raise ValueError(f"Type {self.type} is not an async generator.")

        return DecomposedType(self.args[0])

    def get_base_type(self) -> "DecomposedType":
        """
        Returns the base type of the current type, ignoring any annotations or async generators.

        Returns
        -------
        DecomposedType
            The base type of the current type.
        """

        base_type = self

        while (base_type.has_base_type):
            if (base_type.is_annotated):
                base_type = base_type.get_annotated_type()
            elif (base_type.is_async_generator):
                base_type = base_type.get_async_generator_type()

        return base_type

    def is_subtype(self, class_or_tuple: ClassInfo) -> bool:
        """
        Check if the current type is a subtype of the specified class or tuple of classes similar to `issubclass`.

        Parameters
        ----------
        class_or_tuple : ClassInfo
            The class or tuple of classes to check if the current type is a subtype of.

        Returns
        -------
        bool
            True if the current type is a subtype of the specified class or tuple of classes, False otherwise
        """

        if (isinstance(class_or_tuple, tuple)):
            return any(issubclass(self.root, DecomposedType(cls).root) for cls in class_or_tuple)

        return issubclass(self.root, DecomposedType(class_or_tuple).root)

    def is_instance(self, instance: typing.Any) -> bool:
        """
        Check if the current type is an instance of the specified instance similar to `isinstance`.

        Parameters
        ----------
        instance : typing.Any
            The instance to check if the current type is an instance of.

        Returns
        -------
        bool
            True if the current type is an instance of the specified instance, False otherwise
        """

        return isinstance(instance, self.get_base_type().root)

    def get_pydantic_schema(self,
                            converters: list[collections.abc.Callable] | None = None) -> type[BaseModel] | type[None]:
        """
        Get the Pydantic schema for the current type.

        Parameters
        ----------
        converters : list[Callable], optional
            A list of converters to append new converts to, by default None

        Returns
        -------
        type[BaseModel]
            The Pydantic schema for the current type.
        """

        if (converters is None):
            converters = []

        if (self.has_base_type):
            return self.get_base_type().get_pydantic_schema(converters=converters)

        if (self.type == types.NoneType):
            return types.NoneType

        if (self.is_class and issubclass(self.type, BaseModel)):
            return self.type

        schema = create_model("OutputArgsSchema", value=(self.type, Field(default=PydanticUndefined)))

        def _convert_to_cls(schema_in: schema) -> self.type:
            return schema_in.value

        def _convert_to_schema(cls_in: self.type) -> schema:
            return schema.model_validate({"value": cls_in})

        converters.append(_convert_to_cls)
        converters.append(_convert_to_schema)

        return schema

    @staticmethod
    def extract_generic_parameters_from_class(target_class: type,
                                              expected_param_count: int | None = None) -> tuple[type, ...]:
        """
        Extract generic type parameters from a class's inheritance chain.

        This method searches through __orig_bases__ to find generic parameters,
        which is useful for classes that inherit from generic base classes.

        Parameters
        ----------
        target_class : type
            The class to extract parameters from
        expected_param_count : int | None, optional
            Expected number of parameters. If specified, only matches with this count are considered.

        Returns
        -------
        tuple[type, ...]
            Tuple of generic type parameters found

        Raises
        ------
        ValueError
            If no generic parameters matching the expected count are found

        Examples
        --------
        >>> class MyClass(SomeGeneric[int, str, bool]):
        ...     pass
        >>> DecomposedType.extract_generic_parameters_from_class(MyClass, 3)
        (int, str, bool)
        """
        for base_cls in getattr(target_class, '__orig_bases__', []):
            base_cls_args = typing.get_args(base_cls)

            if expected_param_count is None or len(base_cls_args) == expected_param_count:
                if base_cls_args:  # Only return if we actually found parameters
                    return base_cls_args

        if expected_param_count is not None:
            raise ValueError(
                f"Could not find generic parameters with count {expected_param_count} for class {target_class}")
        raise ValueError(f"Could not find any generic parameters for class {target_class}")

    @staticmethod
    def is_type_compatible(source_type: type, target_type: type) -> bool:
        """
        Check if a source type is compatible with a target type.

        This handles direct compatibility and special cases like batch compatibility
        where list[T] can be compatible with targets that expect T.

        Parameters
        ----------
        source_type : type
            The source type to check
        target_type : type
            The target type to check compatibility with

        Returns
        -------
        bool
            True if types are compatible, False otherwise
        """
        # Direct compatibility check
        try:
            if issubclass(source_type, target_type):
                return True
        except TypeError:
            # Handle generic types that can't use issubclass
            pass

        # Check direct equality (works for both regular and generic types)
        if source_type == target_type:
            return True

        # Check if source outputs list[T] and target expects T
        source_decomposed = DecomposedType(source_type)
        if source_decomposed.origin is list and source_decomposed.args:
            inner_type = source_decomposed.args[0]
            try:
                if issubclass(inner_type, target_type):
                    return True
            except TypeError:
                # If we can't use issubclass, check type equality
                if inner_type == target_type:
                    return True

        return False
