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

from typing import Any

import pytest

from nat.observability.processor.processor import Processor


class TestProcessorAbstractBehavior:
    """Test the abstract behavior of the Processor class."""

    def test_processor_cannot_be_instantiated_directly(self):
        """Test that Processor cannot be instantiated directly due to abstract method."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class Processor"):
            Processor()

    def test_processor_with_unimplemented_process_method_fails(self):
        """Test that a class inheriting from Processor without implementing process() fails."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class IncompleteProcessor(Processor[str, int]):
                pass

            IncompleteProcessor()


class TestProcessorTypeIntrospection:
    """Test the type introspection capabilities of concrete Processor implementations."""

    def test_simple_type_introspection(self):
        """Test type introspection with simple types."""

        class StringToIntProcessor(Processor[str, int]):

            async def process(self, item: str) -> int:
                return len(item)

        processor = StringToIntProcessor()
        assert processor.input_type is str
        assert processor.output_type is int
        assert processor.input_class is str
        assert processor.output_class is int

    def test_generic_type_introspection(self):
        """Test type introspection with generic types."""

        class ListToStringProcessor(Processor[list[int], str]):

            async def process(self, item: list[int]) -> str:
                return str(item)

        processor = ListToStringProcessor()
        assert processor.input_type == list[int]
        assert processor.output_type is str
        assert processor.input_class is list  # Generic origin is list
        assert processor.output_class is str

    def test_complex_generic_type_introspection(self):
        """Test type introspection with complex generic types."""

        class DictToListProcessor(Processor[dict[str, Any], list[str]]):

            async def process(self, item: dict[str, Any]) -> list[str]:
                return list(item.keys())

        processor = DictToListProcessor()
        assert processor.input_type == dict[str, Any]
        assert processor.output_type == list[str]
        assert processor.input_class is dict
        assert processor.output_class is list

    def test_type_introspection_error_handling(self):
        """Test error handling when type introspection fails."""
        from nat.observability.mixin.type_introspection_mixin import TypeIntrospectionMixin

        # Create a class with TypeIntrospectionMixin but no generic type parameters
        class BadProcessor(TypeIntrospectionMixin):

            async def process(self, item):
                return item

        processor = BadProcessor()

        with pytest.raises(ValueError, match="Could not find input type for BadProcessor"):
            _ = processor.input_type

        with pytest.raises(ValueError, match="Could not find output type for BadProcessor"):
            _ = processor.output_type

    def test_type_introspection_caching(self):
        """Test that type introspection results are cached."""

        class CacheTestProcessor(Processor[str, int]):

            async def process(self, item: str) -> int:
                return len(item)

        processor = CacheTestProcessor()

        # Access multiple times to ensure caching works
        input_type_1 = processor.input_type
        input_type_2 = processor.input_type
        output_type_1 = processor.output_type
        output_type_2 = processor.output_type

        # Should be the same object due to caching
        assert input_type_1 is input_type_2
        assert output_type_1 is output_type_2


class TestConcreteProcessorImplementations:
    """Test concrete implementations of the Processor class."""

    async def test_simple_string_processor(self):
        """Test a simple string transformation processor."""

        class UpperCaseProcessor(Processor[str, str]):

            async def process(self, item: str) -> str:
                return item.upper()

        processor = UpperCaseProcessor()
        result = await processor.process("hello world")
        assert result == "HELLO WORLD"

    async def test_type_conversion_processor(self):
        """Test a processor that converts between different types."""

        class StringLengthProcessor(Processor[str, int]):

            async def process(self, item: str) -> int:
                return len(item)

        processor = StringLengthProcessor()
        result = await processor.process("test string")
        assert result == 11

    async def test_list_processing_processor(self):
        """Test a processor that works with list types."""

        class ListSumProcessor(Processor[list[int], int]):

            async def process(self, item: list[int]) -> int:
                return sum(item)

        processor = ListSumProcessor()
        result = await processor.process([1, 2, 3, 4, 5])
        assert result == 15

    async def test_dict_processing_processor(self):
        """Test a processor that works with dictionary types."""

        class DictKeyCountProcessor(Processor[dict[str, Any], int]):

            async def process(self, item: dict[str, Any]) -> int:
                return len(item)

        processor = DictKeyCountProcessor()
        result = await processor.process({"a": 1, "b": 2, "c": 3})
        assert result == 3

    async def test_processor_with_async_operations(self):
        """Test a processor that performs async operations."""

        class AsyncDelayProcessor(Processor[str, str]):

            async def process(self, item: str) -> str:
                # Simulate some async work
                import asyncio
                await asyncio.sleep(0.001)  # Very short delay for testing
                return f"processed: {item}"

        processor = AsyncDelayProcessor()
        result = await processor.process("test")
        assert result == "processed: test"

    async def test_docstring_example_processor(self):
        """Test the processor example from the docstring to ensure it works as documented."""

        # Mock Span and OtelSpan classes for the docstring example
        class Span:

            def __init__(self, name: str):
                self.name = name

        class OtelSpan:

            def __init__(self, name: str):
                self.name = name

        def convert_span_to_otel(span: Span) -> OtelSpan:
            return OtelSpan(span.name)

        class SpanToOtelProcessor(Processor[Span, OtelSpan]):

            async def process(self, item: Span) -> OtelSpan:
                return convert_span_to_otel(item)

        processor = SpanToOtelProcessor()
        assert processor.input_type == Span
        assert processor.output_type == OtelSpan

        span = Span("test-span")
        result = await processor.process(span)
        assert isinstance(result, OtelSpan)
        assert result.name == "test-span"


class TestProcessorErrorHandling:
    """Test error handling in processor implementations."""

    async def test_processor_with_exception(self):
        """Test that exceptions in process method are properly raised."""

        class FailingProcessor(Processor[str, str]):

            async def process(self, item: str) -> str:
                raise ValueError("Processing failed")

        processor = FailingProcessor()
        with pytest.raises(ValueError, match="Processing failed"):
            await processor.process("test")

    async def test_processor_with_type_error(self):
        """Test processor behavior with incorrect input types."""

        class StrictProcessor(Processor[str, int]):

            async def process(self, item: str) -> int:
                if not isinstance(item, str):
                    raise TypeError("Expected string input")
                return len(item)

        processor = StrictProcessor()

        # This should work
        result = await processor.process("test")
        assert result == 4

        # This should raise an error (though type checking would catch this)
        with pytest.raises(TypeError, match="Expected string input"):
            await processor.process(123)  # type: ignore


class TestProcessorInheritance:
    """Test inheritance patterns with Processor."""

    def test_multi_level_inheritance(self):
        """Test that processors can be inherited from other processors."""

        class BaseStringProcessor(Processor[str, str]):

            async def process(self, item: str) -> str:
                return item.strip()

        class ExtendedStringProcessor(BaseStringProcessor):

            async def process(self, item: str) -> str:
                # Call parent's process method and extend it
                stripped = await super().process(item)
                return stripped.upper()

        processor = ExtendedStringProcessor()
        # Type introspection should still work
        assert processor.input_type is str
        assert processor.output_type is str

    async def test_inherited_processor_functionality(self):
        """Test that inherited processors work correctly."""

        class BaseProcessor(Processor[str, str]):

            async def process(self, item: str) -> str:
                return item.strip()

        class ChildProcessor(BaseProcessor):

            async def process(self, item: str) -> str:
                stripped = await super().process(item)
                return stripped.title()

        processor = ChildProcessor()
        result = await processor.process("  hello world  ")
        assert result == "Hello World"

    def test_diamond_inheritance_pattern(self):
        """Test processors with diamond inheritance pattern."""

        class ProcessorMixin:

            def get_timestamp(self) -> str:
                return "2025-01-01T00:00:00Z"

        class BaseProcessor(Processor[str, str]):

            async def process(self, item: str) -> str:
                return item.upper()

        class TimestampProcessor(BaseProcessor, ProcessorMixin):

            async def process(self, item: str) -> str:
                processed = await super().process(item)
                timestamp = self.get_timestamp()
                return f"{processed} - {timestamp}"

        processor = TimestampProcessor()
        assert processor.input_type is str
        assert processor.output_type is str


class TestProcessorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_processor_with_none_types(self):
        """Test processor that can handle None types."""

        class OptionalProcessor(Processor[str | None, str]):

            async def process(self, item: str | None) -> str:
                return item if item is not None else "None"

        processor = OptionalProcessor()
        assert processor.input_type == str | None
        assert processor.output_type is str

    async def test_processor_with_same_input_output_type(self):
        """Test processor where input and output types are the same."""

        class IdentityProcessor(Processor[str, str]):

            async def process(self, item: str) -> str:
                return item

        processor = IdentityProcessor()
        assert processor.input_type is str
        assert processor.output_type is str

        result = await processor.process("test")
        assert result == "test"

    def test_processor_with_custom_classes(self):
        """Test processor with custom class types."""

        class CustomInput:

            def __init__(self, value: str):
                self.value = value

        class CustomOutput:

            def __init__(self, processed_value: str):
                self.processed_value = processed_value

        class CustomProcessor(Processor[CustomInput, CustomOutput]):

            async def process(self, item: CustomInput) -> CustomOutput:
                return CustomOutput(f"processed: {item.value}")

        processor = CustomProcessor()
        assert processor.input_type == CustomInput
        assert processor.output_type == CustomOutput
        assert processor.input_class is CustomInput
        assert processor.output_class == CustomOutput

    def test_processor_with_union_types(self):
        """Test processor with Union types."""
        from typing import get_origin

        class UnionProcessor(Processor[str | int, str]):

            async def process(self, item: str | int) -> str:
                return str(item)

        processor = UnionProcessor()
        assert processor.input_type == str | int
        assert processor.output_type is str
        # Union types have Union as their origin, not the full str | int
        assert processor.input_class is get_origin(str | int)  # This is just Union
        assert processor.output_class is str

    async def test_processor_with_empty_string(self):
        """Test processor edge case with empty input."""

        class EmptyStringProcessor(Processor[str, int]):

            async def process(self, item: str) -> int:
                return len(item)

        processor = EmptyStringProcessor()
        result = await processor.process("")
        assert result == 0

    def test_processor_class_name_in_error_messages(self):
        """Test that processor class names appear correctly in error messages."""
        from nat.observability.mixin.type_introspection_mixin import TypeIntrospectionMixin

        class ProcessorWithoutGenerics(TypeIntrospectionMixin):
            pass

        processor = ProcessorWithoutGenerics()

        with pytest.raises(ValueError, match="Could not find input type for ProcessorWithoutGenerics"):
            _ = processor.input_type

        with pytest.raises(ValueError, match="Could not find output type for ProcessorWithoutGenerics"):
            _ = processor.output_type
