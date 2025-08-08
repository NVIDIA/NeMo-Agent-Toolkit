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

from typing import Generic
from typing import TypeVar
from unittest.mock import patch

import pytest

from nat.observability.mixin.type_introspection_mixin import TypeIntrospectionMixin

# Test classes for different generic scenarios

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class DirectGenericClass(TypeIntrospectionMixin, Generic[InputT, OutputT]):
    """Test class with direct generic parameters"""
    pass


class ConcreteDirectClass(DirectGenericClass[list[int], str]):
    """Concrete class inheriting from direct generic class"""
    pass


class ConcreteDirectComplexClass(DirectGenericClass[dict[str, int], list[str]]):
    """Concrete class with complex generic types"""
    pass


T = TypeVar('T')
U = TypeVar('U')


class IndirectGenericParent(TypeIntrospectionMixin, Generic[T, U]):
    """Parent class with indirect generic pattern"""
    pass


class IndirectGenericChild(IndirectGenericParent[int, list[int]]):
    """Child class that should resolve T=int, U=list[int]"""
    pass


class NonGenericClass(TypeIntrospectionMixin):
    """Class without generic parameters for error testing"""
    pass


SingleT = TypeVar('SingleT')


class SingleGenericClass(TypeIntrospectionMixin, Generic[SingleT]):
    """Class with only one generic parameter"""
    pass


class ConcreteSignleGenericClass(SingleGenericClass[str]):
    """Concrete class with single generic parameter"""
    pass


class TestTypeIntrospectionMixin:
    """Test suite for TypeIntrospectionMixin"""

    def test_direct_generic_input_type(self):
        """Test input_type property with direct generic parameters"""
        instance = ConcreteDirectClass()
        assert instance.input_type == list[int]

    def test_direct_generic_output_type(self):
        """Test output_type property with direct generic parameters"""
        instance = ConcreteDirectClass()
        assert instance.output_type == str

    def test_direct_generic_complex_input_type(self):
        """Test input_type with complex generic types"""
        instance = ConcreteDirectComplexClass()
        assert instance.input_type == dict[str, int]

    def test_direct_generic_complex_output_type(self):
        """Test output_type with complex generic types"""
        instance = ConcreteDirectComplexClass()
        assert instance.output_type == list[str]

    def test_indirect_generic_input_type(self):
        """Test input_type property with indirect generic resolution"""
        instance = IndirectGenericChild()
        assert instance.input_type == int

    def test_indirect_generic_output_type(self):
        """Test output_type property with indirect generic resolution"""
        instance = IndirectGenericChild()
        assert instance.output_type == list[int]

    def test_input_class_simple_type(self):
        """Test input_class property with simple type"""
        instance = ConcreteDirectClass()
        assert instance.input_class == list

    def test_input_class_non_generic_type(self):
        """Test input_class property with non-generic type"""
        instance = ConcreteDirectClass()
        # Mock _find_generic_types to return a non-generic input type
        with patch.object(instance, '_find_generic_types', return_value=(str, list[int])):
            # Clear the cache by accessing the property function
            instance.__class__.input_type.fget.cache_clear()
            instance.__class__.input_class.fget.cache_clear()
            assert instance.input_class == str

    def test_output_class_simple_type(self):
        """Test output_class property with simple type"""
        instance = ConcreteDirectClass()
        assert instance.output_class == str

    def test_output_class_generic_type(self):
        """Test output_class property with generic type"""
        instance = ConcreteDirectComplexClass()
        assert instance.output_class == list

    def test_output_class_non_generic_type(self):
        """Test output_class property with non-generic type"""
        instance = ConcreteDirectComplexClass()
        # Mock _find_generic_types to return a non-generic output type
        with patch.object(instance, '_find_generic_types', return_value=(dict[str, int], int)):
            # Clear the cache by accessing the property function
            instance.__class__.output_type.fget.cache_clear()
            instance.__class__.output_class.fget.cache_clear()
            assert instance.output_class == int

    def test_non_generic_class_input_type_error(self):
        """Test that non-generic class raises error for input_type"""
        instance = NonGenericClass()
        with pytest.raises(ValueError, match="Could not find input type for NonGenericClass"):
            _ = instance.input_type

    def test_non_generic_class_output_type_error(self):
        """Test that non-generic class raises error for output_type"""
        instance = NonGenericClass()
        with pytest.raises(ValueError, match="Could not find output type for NonGenericClass"):
            _ = instance.output_type

    def test_single_generic_parameter_error(self):
        """Test that class with single generic parameter raises error"""
        instance = ConcreteSignleGenericClass()
        with pytest.raises(ValueError, match="Could not find input type for ConcreteSignleGenericClass"):
            _ = instance.input_type

    def test_find_generic_types_direct_case(self):
        """Test _find_generic_types method for direct case"""
        instance = ConcreteDirectClass()
        types = instance._find_generic_types()
        assert types == (list[int], str)

    def test_find_generic_types_indirect_case(self):
        """Test _find_generic_types method for indirect case"""
        instance = IndirectGenericChild()
        types = instance._find_generic_types()
        assert types == (int, list[int])

    def test_find_generic_types_no_types(self):
        """Test _find_generic_types method when no types found"""
        instance = NonGenericClass()
        types = instance._find_generic_types()
        assert types is None

    def test_substitute_type_var_with_typevar(self):
        """Test _substitute_type_var method with TypeVar"""
        instance = IndirectGenericChild()
        result = instance._substitute_type_var(T, int)
        assert result == int

    def test_substitute_type_var_with_generic_type(self):
        """Test _substitute_type_var method with generic type containing TypeVar"""
        instance = IndirectGenericChild()
        result = instance._substitute_type_var(list[T], int)
        assert result == list[int]

    def test_substitute_type_var_with_nested_generic(self):
        """Test _substitute_type_var method with nested generic types"""
        instance = IndirectGenericChild()
        NestedT = TypeVar('NestedT')
        result = instance._substitute_type_var(dict[str, list[NestedT]], int)
        assert result == dict[str, list[int]]

    def test_substitute_type_var_with_non_typevar(self):
        """Test _substitute_type_var method with non-TypeVar type"""
        instance = IndirectGenericChild()
        result = instance._substitute_type_var(str, int)
        assert result == str

    def test_substitute_type_var_with_complex_nested_type(self):
        """Test _substitute_type_var method with complex nested type"""
        instance = IndirectGenericChild()
        # Test with Dict[T, list[T]] pattern
        DictT = TypeVar('DictT')
        complex_type = dict[DictT, list[DictT]]
        result = instance._substitute_type_var(complex_type, str)
        assert result == dict[str, list[str]]

    def test_properties_cached(self):
        """Test that properties are cached using lru_cache"""
        instance = ConcreteDirectClass()

        # Access properties multiple times
        input_type1 = instance.input_type
        input_type2 = instance.input_type
        output_type1 = instance.output_type
        output_type2 = instance.output_type
        input_class1 = instance.input_class
        input_class2 = instance.input_class
        output_class1 = instance.output_class
        output_class2 = instance.output_class

        # Verify they return the same objects (cached)
        assert input_type1 is input_type2
        assert output_type1 is output_type2
        assert input_class1 is input_class2
        assert output_class1 is output_class2

    def test_find_generic_types_with_no_orig_bases(self):
        """Test _find_generic_types when class has no __orig_bases__"""
        instance = ConcreteDirectClass()

        # Mock to remove __orig_bases__
        with patch.object(instance.__class__, '__orig_bases__', []):
            types = instance._find_generic_types()
            assert types is None

    def test_find_generic_types_with_single_arg_no_parent_bases(self):
        """Test _find_generic_types with single arg when parent has no __orig_bases__"""

        # Create a mock class structure
        class MockGeneric(Generic[T]):
            pass

        class MockChild(TypeIntrospectionMixin):
            __orig_bases__ = (MockGeneric[int], )

        instance = MockChild()
        types = instance._find_generic_types()
        assert types is None

    def test_edge_case_empty_args(self):
        """Test behavior with empty type arguments"""

        class EmptyArgsClass(TypeIntrospectionMixin):
            __orig_bases__ = (Generic, )  # Generic with no args

        instance = EmptyArgsClass()
        types = instance._find_generic_types()
        assert types is None
