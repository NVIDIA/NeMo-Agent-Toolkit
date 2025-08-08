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

import typing
from collections.abc import AsyncGenerator
from typing import Generic
from typing import TypeVar

import pytest

from nat.utils.type_utils import DecomposedType

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


class TestExtractGenericParametersFromClass:
    """Tests for DecomposedType.extract_generic_parameters_from_class method."""

    def test_single_parameter_class(self):
        """Test extracting parameters from class with single generic parameter."""

        class MyGeneric(Generic[T]):
            pass

        class MyClass(MyGeneric[int]):
            pass

        result = DecomposedType.extract_generic_parameters_from_class(MyClass)
        assert result == (int, )

    def test_multiple_parameter_class(self):
        """Test extracting parameters from class with multiple generic parameters."""

        class MyGeneric(Generic[T, U, V]):
            pass

        class MyClass(MyGeneric[int, str, bool]):
            pass

        result = DecomposedType.extract_generic_parameters_from_class(MyClass)
        assert result == (int, str, bool)

    def test_expected_param_count_match(self):
        """Test extracting parameters with matching expected count."""

        class MyGeneric(Generic[T, U]):
            pass

        class MyClass(MyGeneric[int, str]):
            pass

        result = DecomposedType.extract_generic_parameters_from_class(MyClass, expected_param_count=2)
        assert result == (int, str)

    def test_expected_param_count_no_match(self):
        """Test error when expected count doesn't match."""

        class MyGeneric(Generic[T, U]):
            pass

        class MyClass(MyGeneric[int, str]):
            pass

        with pytest.raises(ValueError, match="Could not find generic parameters with count 3"):
            DecomposedType.extract_generic_parameters_from_class(MyClass, expected_param_count=3)

    def test_no_generic_parameters(self):
        """Test error when class has no generic parameters."""

        class MyClass:
            pass

        with pytest.raises(ValueError, match="Could not find any generic parameters"):
            DecomposedType.extract_generic_parameters_from_class(MyClass)

    def test_complex_types(self):
        """Test with complex type parameters like list[int]."""

        class MyGeneric(Generic[T, U]):
            pass

        class MyClass(MyGeneric[list[int], dict[str, bool]]):
            pass

        result = DecomposedType.extract_generic_parameters_from_class(MyClass)
        assert result == (list[int], dict[str, bool])

    def test_nested_generics(self):
        """Test with nested generic types."""

        class MyGeneric(Generic[T]):
            pass

        class MyClass(MyGeneric[AsyncGenerator[str]]):
            pass

        result = DecomposedType.extract_generic_parameters_from_class(MyClass)
        assert result == (AsyncGenerator[str], )

    def test_inheritance_chain(self):
        """Test with inheritance chain."""

        class BaseGeneric(Generic[T, U]):
            pass

        class MiddleClass(BaseGeneric[int, str]):
            pass

        # MiddleClass inherits from BaseGeneric[int, str], so it should find those parameters
        result = DecomposedType.extract_generic_parameters_from_class(MiddleClass)
        assert result == (int, str)


class TestIsTypeCompatible:
    """Tests for DecomposedType.is_type_compatible method."""

    def test_direct_compatibility_same_type(self):
        """Test direct compatibility with same types."""
        assert DecomposedType.is_type_compatible(int, int) is True
        assert DecomposedType.is_type_compatible(str, str) is True
        assert DecomposedType.is_type_compatible(list, list) is True

    def test_direct_compatibility_subclass(self):
        """Test direct compatibility with subclass relationship."""

        class Base:
            pass

        class Derived(Base):
            pass

        assert DecomposedType.is_type_compatible(Derived, Base) is True
        assert DecomposedType.is_type_compatible(Base, Derived) is False

    def test_incompatible_types(self):
        """Test incompatible types."""
        assert DecomposedType.is_type_compatible(int, str) is False
        assert DecomposedType.is_type_compatible(list, dict) is False

    def test_batch_compatibility_list_to_element(self):
        """Test batch compatibility: list[T] compatible with T."""
        assert DecomposedType.is_type_compatible(list[int], int) is True
        assert DecomposedType.is_type_compatible(list[str], str) is True
        assert DecomposedType.is_type_compatible(list[dict], dict) is True

    def test_batch_compatibility_with_subclass(self):
        """Test batch compatibility with subclass relationships."""

        class Base:
            pass

        class Derived(Base):
            pass

        assert DecomposedType.is_type_compatible(list[Derived], Base) is True
        assert DecomposedType.is_type_compatible(list[Base], Derived) is False

    def test_batch_incompatibility(self):
        """Test cases where batch compatibility should not apply."""
        assert DecomposedType.is_type_compatible(list[int], str) is False
        assert DecomposedType.is_type_compatible(list[str], int) is False

    def test_non_list_containers(self):
        """Test that batch compatibility only applies to lists."""
        assert DecomposedType.is_type_compatible(set[int], int) is False
        assert DecomposedType.is_type_compatible(tuple[int, ...], int) is False
        assert DecomposedType.is_type_compatible(dict[str, int], int) is False

    def test_generic_type_edge_cases(self):
        """Test edge cases with generic types."""
        # Generic types that can't use issubclass should fall back gracefully
        assert DecomposedType.is_type_compatible(list[int], list[int]) is False  # Generic aliases

    def test_complex_batch_scenarios(self):
        """Test complex batch compatibility scenarios."""

        class CustomClass:
            pass

        class CustomSubclass(CustomClass):
            pass

        # Test with custom classes
        assert DecomposedType.is_type_compatible(list[CustomSubclass], CustomClass) is True
        assert DecomposedType.is_type_compatible(list[CustomClass], CustomSubclass) is False

        # Test with built-in types
        assert DecomposedType.is_type_compatible(list[bool], int) is True  # bool is subclass of int
        assert DecomposedType.is_type_compatible(list[int], bool) is False

    def test_type_equality_fallback(self):
        """Test type equality fallback when issubclass fails."""
        # Create a scenario where issubclass would fail but types are equal
        # This tests the TypeError exception handling

        # For generic types, the method should handle TypeError gracefully
        result = DecomposedType.is_type_compatible(list[typing.Any], typing.Any)
        assert result is True  # Should work via type equality

    def test_empty_list_scenario(self):
        """Test compatibility with empty list scenarios."""
        # list without type parameter
        assert DecomposedType.is_type_compatible(list, int) is False


class TestDecomposedTypeBasics:
    """Basic tests for DecomposedType functionality to ensure core features work."""

    def test_decomposed_type_creation(self):
        """Test basic DecomposedType creation and properties."""
        dt = DecomposedType(list[int])
        assert dt.origin is list
        assert dt.args == (int, )
        assert dt.root is list

    def test_non_generic_type(self):
        """Test DecomposedType with non-generic types."""
        dt = DecomposedType(int)
        assert dt.origin is None
        assert dt.args == ()
        assert dt.root is int

    def test_is_generic_property(self):
        """Test is_generic property."""
        assert DecomposedType(list[int]).is_generic is True
        assert DecomposedType(int).is_generic is False
