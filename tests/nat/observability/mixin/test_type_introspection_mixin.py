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
from typing import get_args
from typing import get_origin
from unittest.mock import patch

import pytest

from nat.observability.mixin.type_introspection_mixin import TypeIntrospectionMixin

# Test classes for different generic scenarios

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class DirectGenericClass(TypeIntrospectionMixin, Generic[InputT, OutputT]):
    """Test class with direct generic parameters"""


class ConcreteDirectClass(DirectGenericClass[list[int], str]):
    """Concrete class inheriting from direct generic class"""


class ConcreteDirectComplexClass(DirectGenericClass[dict[str, int], list[str]]):
    """Concrete class with complex generic types"""


T = TypeVar('T')
U = TypeVar('U')


class IndirectGenericParent(TypeIntrospectionMixin, Generic[T, U]):
    """Parent class with indirect generic pattern"""


class IndirectGenericChild(IndirectGenericParent[int, list[int]]):
    """Child class that should resolve T=int, U=list[int]"""


class NonGenericClass(TypeIntrospectionMixin):
    """Class without generic parameters for error testing"""


SingleT = TypeVar('SingleT')


class SingleGenericClass(TypeIntrospectionMixin, Generic[SingleT]):
    """Class with only one generic parameter"""


class ConcreteSignleGenericClass(SingleGenericClass[str]):
    """Concrete class with single generic parameter"""


# Test classes for Generic[T] mixed inheritance edge case
DataT = TypeVar('DataT')


class BaseProcessor(TypeIntrospectionMixin, Generic[InputT, OutputT]):
    """Base processor with InputT, OutputT pattern"""


class SpanProcessor(BaseProcessor[str, str]):
    """Span processor with concrete types"""


class ContextualProcessor(SpanProcessor, Generic[DataT]):
    """Processor that mixes SpanProcessor inheritance with Generic[DataT]"""


class ConcreteContextualProcessor(ContextualProcessor[dict[str, int]]):
    """Concrete processor that should find str -> str through MRO traversal"""


class DeepInheritanceBase(TypeIntrospectionMixin, Generic[InputT, OutputT]):
    """Deep inheritance base"""


class DeepInheritanceMiddle(DeepInheritanceBase[int, list[int]]):
    """Middle layer without Generic"""


class DeepInheritanceChild(DeepInheritanceMiddle, Generic[DataT]):
    """Child with Generic[DataT] that should find int -> list[int] deep in MRO"""


class ConcreteDeepInheritance(DeepInheritanceChild[str]):
    """Concrete class testing deep MRO traversal"""


# Test classes for same TypeVar expansion (RedactionProcessor pattern)
# Simulate the exact pattern: Processor[T, T] -> RedactionProcessor[T] -> ConcreteProcessor[Span]
RedactionT = TypeVar('RedactionT')


class MockProcessor(TypeIntrospectionMixin, Generic[InputT, OutputT]):
    """Mock Processor[InputT, OutputT] base class"""


class MockRedactionProcessor(MockProcessor[RedactionT, RedactionT]):
    """Mock RedactionProcessor that inherits from Processor[T, T] - same TypeVar twice"""


class ConcreteMockRedactionProcessor(MockRedactionProcessor[str]):
    """Concrete processor that should expand [str] to [str, str]"""


class TestTypeIntrospectionMixin:
    """Test suite for TypeIntrospectionMixin"""

    def test_direct_generic_input_type(self):
        """Test input_type property with direct generic parameters"""
        instance = ConcreteDirectClass()
        assert instance.input_type == list[int]

    def test_direct_generic_output_type(self):
        """Test output_type property with direct generic parameters"""
        instance = ConcreteDirectClass()
        assert instance.output_type is str

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
        assert instance.input_type is int

    def test_indirect_generic_output_type(self):
        """Test output_type property with indirect generic resolution"""
        instance = IndirectGenericChild()
        assert instance.output_type == list[int]

    def test_pydantic_validation_simple_types(self):
        """Test Pydantic-based validation with simple types"""
        instance = ConcreteDirectClass()

        # Test input validation
        assert instance.validate_input_type([1, 2, 3])
        assert not instance.validate_input_type("not_a_list")

        # Test output validation
        assert instance.validate_output_type("test_string")
        assert not instance.validate_output_type(123)

    def test_pydantic_validation_generic_types(self):
        """Test Pydantic-based validation with generic types"""
        instance = ConcreteDirectComplexClass()

        # Test input validation for dict[str, int]
        assert instance.validate_input_type({"key": 123})
        assert instance.validate_input_type({"key": 456, "another": 789})
        assert not instance.validate_input_type([1, 2, 3])
        assert not instance.validate_input_type({"key": "value"})  # String value, not int

        # Test output validation for list[str]
        assert instance.validate_output_type(["item1", "item2"])
        assert not instance.validate_output_type([1, 2, 3])

    def test_type_compatibility_methods(self):
        """Test type compatibility checking methods"""
        instance = ConcreteDirectClass()

        # Test input compatibility - ConcreteDirectClass has input_type = list[int]
        assert instance.is_compatible_with_input(list[int])  # Exact match should be compatible
        assert not instance.is_compatible_with_input(str)  # Different type should not be compatible

        # Test output compatibility - ConcreteDirectClass has output_type = str
        assert instance.is_output_compatible_with(str)
        assert instance.is_output_compatible_with(object)  # More general should be compatible
        assert not instance.is_output_compatible_with(int)

    def test_strict_type_compatibility(self):
        """Test strict type compatibility without batch compatibility hacks"""

        # Use existing test class that has proper type extraction
        instance = ConcreteDirectClass()  # Has input_type = list[int], output_type = str

        # Test that exact type matches work
        assert instance.is_compatible_with_input(list[int])  # Exact match
        assert instance.is_output_compatible_with(str)  # Exact match

        # Test that mismatched types are not compatible (no batch compatibility)
        assert not instance.is_compatible_with_input(str)  # No batch compatibility: str != list[int]
        assert not instance.is_compatible_with_input(list[str])  # Different generic args
        assert not instance.is_output_compatible_with(int)  # Different types

        # Test subclass compatibility still works
        assert instance.is_output_compatible_with(object)  # str is subclass of object

        # Test that the old batch compatibility behavior is gone
        # (list[T] should NOT be compatible with T anymore in TypeIntrospectionMixin)
        assert not instance._is_pydantic_type_compatible(list[str], str)

    def test_non_generic_class_input_type_error(self):
        """Test that non-generic class raises error for input_type"""
        instance = NonGenericClass()
        with pytest.raises(ValueError, match="Could not extract input/output types from NonGenericClass"):
            _ = instance.input_type

    def test_non_generic_class_output_type_error(self):
        """Test that non-generic class raises error for output_type"""
        instance = NonGenericClass()
        with pytest.raises(ValueError, match="Could not extract input/output types from NonGenericClass"):
            _ = instance.output_type

    def test_single_generic_parameter_error(self):
        """Test that class with single generic parameter raises error"""
        instance = ConcreteSignleGenericClass()
        with pytest.raises(ValueError, match="Could not extract input/output types from ConcreteSignleGenericClass"):
            _ = instance.input_type

    def test_properties_cached(self):
        """Test that properties are cached using lru_cache"""
        instance = ConcreteDirectClass()

        # Access properties multiple times
        input_type1 = instance.input_type
        input_type2 = instance.input_type
        output_type1 = instance.output_type
        output_type2 = instance.output_type

        # Verify they return the same objects (cached)
        assert input_type1 is input_type2
        assert output_type1 is output_type2

        # Test that validation methods work consistently
        assert instance.validate_input_type([1, 2, 3])
        assert instance.validate_input_type([1, 2, 3])  # Should be consistent

    def test_no_orig_bases_error(self):
        """Test behavior when class has no __orig_bases__"""
        instance = ConcreteDirectClass()

        # Mock to remove __orig_bases__
        with patch.object(instance.__class__, '__orig_bases__', []):
            # Clear cache to force re-evaluation
            instance._extract_input_output_types.cache_clear()
            with pytest.raises(ValueError):
                _ = instance.input_type

    def test_single_arg_no_parent_bases_error(self):
        """Test behavior with single arg when parent has no suitable bases"""

        # Create a mock class structure
        class MockGeneric(Generic[T]):
            pass

        class MockChild(TypeIntrospectionMixin):
            __orig_bases__ = (MockGeneric[int], )

        instance = MockChild()
        with pytest.raises(ValueError):
            _ = instance.input_type

    def test_edge_case_empty_args(self):
        """Test behavior with empty type arguments"""

        class EmptyArgsClass(TypeIntrospectionMixin):
            __orig_bases__ = (Generic, )  # Generic with no args

        instance = EmptyArgsClass()
        with pytest.raises(ValueError):
            _ = instance.input_type

    def test_mixed_inheritance_with_generic_skipping(self):
        """Test that Generic[T] bases are skipped in mixed inheritance"""
        instance = ConcreteContextualProcessor()

        # Should find str -> str from SpanProcessor, skipping Generic[DataT]
        assert instance.input_type is str
        assert instance.output_type is str

    def test_mixed_inheritance_behavior(self):
        """Test behavior with mixed inheritance - should work through public interface"""
        instance = ConcreteContextualProcessor()

        # Should find the InputT, OutputT pattern despite Generic[DataT] confusion
        assert instance.input_type is str
        assert instance.output_type is str

    def test_deep_mro_traversal_with_generic_skipping(self):
        """Test MRO traversal works when Generic[T] is present"""
        instance = ConcreteDeepInheritance()

        # Should find int -> list[int] deep in the MRO, skipping Generic[DataT]
        assert instance.input_type is int
        assert instance.output_type == list[int]

    def test_deep_mro_behavior(self):
        """Test behavior with deep MRO traversal"""
        instance = ConcreteDeepInheritance()

        # Should traverse MRO and find the deep inheritance pattern
        assert instance.input_type is int
        assert instance.output_type == list[int]

    def test_generic_bases_are_skipped(self):
        """Test that typing.Generic bases are properly skipped"""
        instance = ConcreteContextualProcessor()

        # Verify that the algorithm doesn't get confused by Generic[DataT]

        # Check that ContextualProcessor has Generic[DataT] in its bases
        contextual_bases = getattr(ContextualProcessor, '__orig_bases__', [])
        has_generic = any(get_origin(base) is Generic for base in contextual_bases)
        assert has_generic, "Test setup should have Generic[DataT] in bases"

        # But type introspection should still work
        assert instance.input_type is str
        assert instance.output_type is str

    def test_mro_traversal_fallback(self):
        """Test that MRO traversal works as fallback when immediate bases don't have 2 args"""
        instance = ConcreteContextualProcessor()

        # The immediate __orig_bases__ of ConcreteContextualProcessor should not have 2 args
        immediate_bases = getattr(instance.__class__, '__orig_bases__', [])
        has_two_args = any(len(get_args(base)) >= 2 for base in immediate_bases)
        assert not has_two_args, "Test setup: immediate bases should not have 2 args"

        # But MRO traversal should find the 2-arg pattern through public interface
        assert instance.input_type is str
        assert instance.output_type is str

    def test_same_typevar_expansion(self):
        """Test that single type argument expands correctly when parent uses same TypeVar twice"""
        # This tests the RedactionProcessor[T, T] -> RedactionProcessor[Span] pattern
        instance = ConcreteMockRedactionProcessor()

        # Verify the test setup: MockRedactionProcessor should inherit from MockProcessor[T, T]
        parent_bases = getattr(MockRedactionProcessor, '__orig_bases__', [])
        assert len(parent_bases) > 0

        # Should find MockProcessor[RedactionT, RedactionT]
        processor_base = parent_bases[0]
        type_args = get_args(processor_base)
        assert len(type_args) == 2
        assert type_args[0] == type_args[1], "MockRedactionProcessor should use same TypeVar for both positions"

        # The key test: single type argument [str] should expand to [str, str]
        # This tests our algorithm's ability to detect same TypeVar pattern and expand correctly
        assert instance.input_type is str
        assert instance.output_type is str


class TestSignatureBasedExtraction:
    """Test signature-based type extraction functionality"""

    def test_signature_method_attribute(self):
        """Test class with _signature_method attribute"""

        class ProcessorWithSignature(TypeIntrospectionMixin):
            _signature_method = 'process'

            async def process(self, item: str) -> int:
                return len(item)

        instance = ProcessorWithSignature()
        assert instance.input_type is str
        assert instance.output_type is int

    def test_discovered_signature_method(self):
        """Test automatic signature method discovery"""

        class ProcessorWithoutAttribute(TypeIntrospectionMixin):

            async def process(self, item: list[str]) -> dict[str, int]:
                return {s: len(s) for s in item}

        instance = ProcessorWithoutAttribute()
        assert instance.input_type == list[str]
        assert instance.output_type == dict[str, int]

    def test_no_type_annotations_fallback(self):
        """Test fallback when signature method has no type annotations"""

        class ProcessorNoAnnotations(TypeIntrospectionMixin, Generic[InputT, OutputT]):

            async def process(self, item):  # No annotations
                return item

        class ConcreteProcessor(ProcessorNoAnnotations[str, int]):
            pass

        instance = ConcreteProcessor()
        # Should fall back to MRO-based approach
        assert instance.input_type is str
        assert instance.output_type is int


class TestUnionTypes:
    """Test union type functionality"""

    def test_union_input_detection(self):
        """Test detection of union input types"""

        class UnionInputClass(TypeIntrospectionMixin):

            async def process(self, item: str | int) -> str:
                return str(item)

        instance = UnionInputClass()
        assert instance.has_union_input is True
        assert instance.has_union_output is False
        input_types = instance.input_union_types
        assert input_types is not None and set(input_types) == {str, int}
        assert instance.output_union_types is None

    def test_union_output_detection(self):
        """Test detection of union output types"""

        class UnionOutputClass(TypeIntrospectionMixin):

            async def process(self, item: str) -> int | float:
                return len(item) if len(item) > 5 else float(len(item))

        instance = UnionOutputClass()
        assert instance.has_union_input is False
        assert instance.has_union_output is True
        assert instance.input_union_types is None
        output_types = instance.output_union_types
        assert output_types is not None and set(output_types) == {int, float}

    def test_no_union_types(self):
        """Test behavior with non-union types"""
        instance = ConcreteDirectClass()
        assert instance.has_union_input is False
        assert instance.has_union_output is False
        assert instance.input_union_types is None
        assert instance.output_union_types is None


class TestCompatibilityMethods:
    """Test type compatibility methods"""

    def test_input_compatibility(self):
        """Test is_compatible_with_input method"""
        instance = ConcreteDirectClass()  # input: list[int]

        # Test that the method works (exact compatibility logic depends on DecomposedType implementation)
        result = instance.is_compatible_with_input(list[int])
        assert isinstance(result, bool)  # Should return a boolean

        # Test with different type
        result2 = instance.is_compatible_with_input(str)
        assert isinstance(result2, bool)

    def test_output_compatibility(self):
        """Test is_output_compatible_with method"""
        instance = ConcreteDirectClass()  # output: str

        assert instance.is_output_compatible_with(str) is True
        assert instance.is_output_compatible_with(object) is True  # More general should be compatible


class TestRecursiveTypeVarResolution:
    """Test recursive TypeVar resolution in deeply nested generics"""

    def test_deeply_nested_generics(self):
        """Test recursive resolution of deeply nested generic types"""

        NestedT = TypeVar('NestedT')

        class DeepGeneric(TypeIntrospectionMixin, Generic[NestedT]):

            async def process(self, item: dict[str, list[NestedT | None]]) -> list[dict[str, NestedT]]:
                # This is a test method, the implementation doesn't need to be perfect
                return [{"result": val} for val in item.get("data", []) if val is not None]

        instance = DeepGeneric()
        # For this test, we're checking that the signature method can handle complex nested types
        # The actual types will be resolved from the method signature, not the TypeVar
        input_type = instance.input_type
        output_type = instance.output_type
        assert get_origin(input_type) is dict
        assert get_origin(output_type) is list

    def test_multiple_typevar_resolution(self):
        """Test resolution when multiple TypeVars are involved"""

        class MultiTypeVar(TypeIntrospectionMixin):

            async def process(self, item: dict[str, list[int]]) -> list[tuple[str, int]]:
                return [(k, v) for k, vals in item.items() for v in vals]

        instance = MultiTypeVar()
        assert instance.input_type == dict[str, list[int]]
        assert instance.output_type == list[tuple[str, int]]


class TestRealWorldPatterns:
    """Test patterns that match real-world usage in the codebase"""

    def test_processor_pattern(self):
        """Test the actual Processor[InputT, OutputT] pattern"""

        class RealProcessor(TypeIntrospectionMixin, Generic[InputT, OutputT]):
            _signature_method = 'process'

            async def process(self, item: InputT) -> OutputT:
                return item  # type: ignore

        class ConcreteRealProcessor(RealProcessor[str, int]):

            async def process(self, item: str) -> int:
                return len(item)

        instance = ConcreteRealProcessor()
        # Should use signature method (concrete types) over MRO (TypeVars)
        assert instance.input_type is str
        assert instance.output_type is int

    def test_redaction_processor_pattern(self):
        """Test the exact RedactionProcessor pattern from the codebase"""

        class MockSpan:
            pass

        # Simulate the exact inheritance pattern
        class RealWorldBaseProcessor(TypeIntrospectionMixin, Generic[InputT, OutputT]):
            pass

        class RedactionProcessor(RealWorldBaseProcessor[RedactionT, RedactionT], Generic[RedactionT]):
            pass

        class SpanRedactionProcessor(RedactionProcessor[MockSpan]):
            pass

        instance = SpanRedactionProcessor()
        # Should resolve single type argument to both input and output
        assert instance.input_type is MockSpan
        assert instance.output_type is MockSpan


class TestExtractNonOptionalType:
    """Test extract_non_optional_type method functionality"""

    def test_extract_from_optional_type_int(self):
        """Test extracting concrete type from int | None"""
        instance = ConcreteDirectClass()

        # Test with int | None
        optional_int = int | None
        result = instance.extract_non_optional_type(optional_int)
        assert result is int

    def test_extract_from_optional_type_str(self):
        """Test extracting concrete type from str | None"""
        instance = ConcreteDirectClass()

        # Test with str | None
        optional_str = str | None
        result = instance.extract_non_optional_type(optional_str)
        assert result is str

    def test_extract_from_optional_complex_type(self):
        """Test extracting concrete type from complex optional types"""
        instance = ConcreteDirectClass()

        # Test with dict[str, int] | None
        optional_dict = dict[str, int] | None
        result = instance.extract_non_optional_type(optional_dict)
        assert result == dict[str, int]

        # Test with list[str] | None
        optional_list = list[str] | None
        result = instance.extract_non_optional_type(optional_list)
        assert result == list[str]

    def test_extract_from_union_with_none_first(self):
        """Test extracting when None is first in union"""
        instance = ConcreteDirectClass()

        # Test with None | str (order shouldn't matter)
        union_type = None | str
        result = instance.extract_non_optional_type(union_type)
        assert result is str

    def test_extract_from_non_optional_type(self):
        """Test that non-optional types are returned unchanged"""
        instance = ConcreteDirectClass()

        # Test with concrete types that are not optional
        assert instance.extract_non_optional_type(int) is int
        assert instance.extract_non_optional_type(str) is str
        assert instance.extract_non_optional_type(dict[str, int]) == dict[str, int]
        assert instance.extract_non_optional_type(list[str]) == list[str]

    def test_extract_from_union_without_none(self):
        """Test that unions without None are returned unchanged"""
        instance = ConcreteDirectClass()

        # Test with union that doesn't include None
        union_type = str | int
        result = instance.extract_non_optional_type(union_type)
        assert result == (str | int)

    def test_extract_from_complex_union_with_none(self):
        """Test extracting from complex union with None"""
        instance = ConcreteDirectClass()

        # Test with multiple types and None
        union_type = str | int | None
        result = instance.extract_non_optional_type(union_type)
        # Should extract the non-None part of the union
        assert result == (str | int)

    def test_extract_from_nested_generic_optional(self):
        """Test extracting from nested generic optional types"""
        instance = ConcreteDirectClass()

        # Test with nested generics
        optional_nested = dict[str, list[int]] | None
        result = instance.extract_non_optional_type(optional_nested)
        assert result == dict[str, list[int]]

        # Test with very complex nested type
        complex_optional = dict[str, list[tuple[str, int]]] | None
        result = instance.extract_non_optional_type(complex_optional)
        assert result == dict[str, list[tuple[str, int]]]

    def test_extract_preserves_original_type_object(self):
        """Test that the extracted type maintains its identity"""
        instance = ConcreteDirectClass()

        # Create a specific type
        custom_type = dict[str, int]
        optional_custom = custom_type | None

        result = instance.extract_non_optional_type(optional_custom)
        assert result == custom_type
        assert result is not optional_custom

    def test_extract_with_direct_types(self):
        """Test extracting with direct types (non-optional)"""
        instance = ConcreteDirectClass()

        # This tests that the method correctly handles direct types
        result = instance.extract_non_optional_type(str)  # Direct type, not optional
        assert result is str
