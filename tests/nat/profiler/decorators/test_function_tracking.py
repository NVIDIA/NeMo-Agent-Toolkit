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
import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Generator
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import TraceMetadata
from nat.profiler.decorators.function_tracking import track_function
from nat.profiler.decorators.function_tracking import track_unregistered_function
from nat.utils.reactive.subject import Subject


async def test_sync_function_no_metadata(reactive_stream: Subject):
    """Test a simple synchronous function with no metadata."""
    published_events = []
    reactive_stream.subscribe(published_events.append)

    @track_function
    def add(a, b):
        return a + b

    out = add(2, 3)
    assert out == 5

    # We expect exactly 2 events for a normal (non-generator) function: SPAN_START and SPAN_END
    assert len(published_events) == 2

    # Check SPAN_START
    start_event: IntermediateStepPayload = published_events[0].payload
    assert start_event.event_type == IntermediateStepType.SPAN_START
    assert start_event.metadata.span_inputs[0] == [2, 3]
    assert start_event.metadata.span_inputs[1] == {}

    # Check SPAN_END
    end_event: IntermediateStepPayload = published_events[1].payload
    assert end_event.event_type == IntermediateStepType.SPAN_END
    assert end_event.metadata.span_outputs == 5


async def test_sync_function_with_metadata(reactive_stream: Subject):
    """Test a synchronous function with metadata."""
    published_events = []
    reactive_stream.subscribe(published_events.append)

    @track_function(metadata={"purpose": "test_sync"})
    def multiply(x, y):
        return x * y

    result = multiply(4, 5)
    assert result == 20

    assert len(published_events) == 2
    start_event: IntermediateStepPayload = published_events[0].payload
    end_event: IntermediateStepPayload = published_events[1].payload

    assert start_event.event_type == IntermediateStepType.SPAN_START
    assert end_event.event_type == IntermediateStepType.SPAN_END

    assert end_event.metadata.span_outputs == 20
    assert start_event.metadata.provided_metadata == {"purpose": "test_sync"}


async def test_sync_generator(reactive_stream: Subject):
    """Test a synchronous generator with three yields."""
    published_events = []
    reactive_stream.subscribe(published_events.append)

    @track_function
    def number_generator(n):
        for i in range(n):
            yield i

    nums = list(number_generator(3))
    assert nums == [0, 1, 2]

    # For a generator: SPAN_START, SPAN_CHUNK (for each yield), SPAN_END
    # We yield 3 items => 1 start, 3 chunk, 1 end => total 5 events
    assert len(published_events) == 5

    assert published_events[0].payload.event_type == IntermediateStepType.SPAN_START
    for i in range(1, 4):
        assert published_events[i].payload.event_type == IntermediateStepType.SPAN_CHUNK
        assert published_events[i].payload.metadata.span_outputs == i - 1  # i-th event has output i-1
    assert published_events[4].payload.event_type == IntermediateStepType.SPAN_END


async def test_class_method(reactive_stream: Subject):
    """Test decorating a class method."""
    published_events = []
    reactive_stream.subscribe(published_events.append)

    class Calculator:

        @track_function(metadata={"class_method": True})
        def subtract(self, x, y):
            return x - y

    calc = Calculator()
    result = calc.subtract(10, 4)
    assert result == 6

    assert len(published_events) == 2
    start_event: IntermediateStepPayload = published_events[0].payload
    end_event: IntermediateStepPayload = published_events[1].payload

    assert start_event.event_type == IntermediateStepType.SPAN_START
    assert start_event.metadata.span_inputs[0][1:] == [10, 4]
    assert end_event.metadata.span_outputs == 6


async def test_async_function(reactive_stream: Subject):
    """Test an async function decorated with track_function."""
    published_events = []
    reactive_stream.subscribe(published_events.append)

    @track_function
    async def async_add(a, b):
        await asyncio.sleep(0.1)
        return a + b

    result = await async_add(7, 3)
    assert result == 10

    # For an async, non-generator function => SPAN_START and SPAN_END
    assert len(published_events) == 2
    assert published_events[0].payload.event_type == IntermediateStepType.SPAN_START
    assert published_events[0].payload.metadata.span_inputs[0] == [7, 3]
    assert published_events[1].payload.event_type == IntermediateStepType.SPAN_END
    assert published_events[1].payload.metadata.span_outputs == 10


async def test_async_generator(reactive_stream: Subject):
    """Test an async generator function with multiple yields."""
    published_events = []
    reactive_stream.subscribe(published_events.append)

    @track_function(metadata={"test": "async_gen"})
    async def countdown(n):
        while n > 0:
            yield n
            n -= 1

    collected = []
    async for val in countdown(3):
        collected.append(val)

    assert collected == [3, 2, 1]

    # For an async generator with 3 yields => 1 SPAN_START, 3 SPAN_CHUNK, 1 SPAN_END => total 5
    assert len(published_events) == 5
    assert published_events[0].payload.event_type == IntermediateStepType.SPAN_START
    assert published_events[0].payload.metadata.span_inputs[0] == [3]
    for i in range(1, 4):
        assert published_events[i].payload.event_type == IntermediateStepType.SPAN_CHUNK
        # The output is 3, 2, 1 respectively
        assert published_events[i].payload.metadata.span_outputs == 4 - i
    assert published_events[4].payload.event_type == IntermediateStepType.SPAN_END


class MyModel(BaseModel):
    """Simple Pydantic model for testing serialization."""
    name: str
    value: int


async def test_sync_function_pydantic(reactive_stream: Subject):
    """
    Test that a synchronous function with a Pydantic model input
    properly serializes the model via model_dump().
    """
    published_events = []
    reactive_stream.subscribe(published_events.append)

    @track_function
    def process_model(m: MyModel):
        return f"Model is {m.name} with value {m.value}"

    my_obj = MyModel(name="test", value=42)
    output = process_model(my_obj)

    assert output == "Model is test with value 42"
    assert len(published_events) == 2

    start_event: IntermediateStepPayload = published_events[0].payload
    end_event: IntermediateStepPayload = published_events[1].payload

    # Check SPAN_START has the model fully serialized
    assert start_event.event_type == IntermediateStepType.SPAN_START
    # Should see something like [{"name": "test", "value": 42}] for the args
    assert start_event.metadata.span_inputs[0] == [{"name": "test", "value": 42}]
    assert start_event.metadata.span_inputs[1] == {}

    # Check SPAN_END output
    assert end_event.event_type == IntermediateStepType.SPAN_END
    assert end_event.metadata.span_outputs == "Model is test with value 42"


class TestTrackUnregisteredFunction:
    """Tests for the track_unregistered_function decorator."""

    @pytest.fixture
    def mock_context(self):
        """Mock Context and its push_active_function method."""
        with patch('nat.profiler.decorators.function_tracking.Context') as mock_context_class:
            mock_context_instance = Mock()
            mock_manager = Mock()
            mock_context_instance.push_active_function.return_value.__enter__ = Mock(return_value=mock_manager)
            mock_context_instance.push_active_function.return_value.__exit__ = Mock(return_value=None)
            mock_context_class.get.return_value = mock_context_instance

            yield mock_context_instance, mock_manager

    def test_basic_decoration_sync_function(self, mock_context):
        """Test basic decoration of sync function without parameters."""
        context_instance, manager = mock_context

        @track_unregistered_function
        def test_func(x: int, y: int) -> int:
            return x + y

        result = test_func(3, 5)

        assert result == 8
        context_instance.push_active_function.assert_called_once()
        call_args = context_instance.push_active_function.call_args
        assert call_args[0][0] == "test_func"  # function name
        assert call_args[1]['input_data'] == (3, 5, {})  # args + kwargs
        manager.set_output.assert_called_once_with(8)

    def test_decoration_with_custom_name(self, mock_context):
        """Test decoration with custom name parameter."""
        context_instance, _ = mock_context

        @track_unregistered_function(name="custom_calculation")
        def add_numbers(a: int, b: int) -> int:
            return a + b

        result = add_numbers(10, 20)

        assert result == 30
        call_args = context_instance.push_active_function.call_args
        assert call_args[0][0] == "custom_calculation"  # custom name used

    def test_decoration_with_metadata(self, mock_context):
        """Test decoration with metadata parameter."""
        context_instance, _ = mock_context
        test_metadata = {"version": "1.0", "category": "math"}

        @track_unregistered_function(metadata=test_metadata)
        def multiply(x: int, y: int) -> int:
            return x * y

        result = multiply(4, 7)

        assert result == 28
        call_args = context_instance.push_active_function.call_args
        trace_metadata = call_args[1]['metadata']
        assert isinstance(trace_metadata, TraceMetadata)
        assert trace_metadata.provided_metadata == test_metadata

    def test_decoration_with_name_and_metadata(self, mock_context):
        """Test decoration with both custom name and metadata."""
        context_instance, _ = mock_context
        test_metadata = {"operation": "division"}

        @track_unregistered_function(name="divide_operation", metadata=test_metadata)
        def divide(numerator: float, denominator: float) -> float:
            return numerator / denominator

        result = divide(15.0, 3.0)

        assert result == 5.0
        call_args = context_instance.push_active_function.call_args
        assert call_args[0][0] == "divide_operation"
        trace_metadata = call_args[1]['metadata']
        assert trace_metadata.provided_metadata == test_metadata

    def test_invalid_metadata_type(self):
        """Test that non-dict metadata raises TypeError."""
        with pytest.raises(TypeError, match="metadata must be a dict"):

            @track_unregistered_function(metadata="invalid")  # type: ignore
            def some_func():
                pass

    def test_invalid_metadata_keys(self):
        """Test that non-string metadata keys raise TypeError."""
        with pytest.raises(TypeError, match="All metadata keys must be strings"):

            @track_unregistered_function(metadata={123: "value"})  # type: ignore
            def some_func():
                pass

    async def test_async_function_decoration(self, mock_context):
        """Test decoration of async functions."""
        context_instance, manager = mock_context

        @track_unregistered_function
        async def async_add(x: int, y: int) -> int:
            await asyncio.sleep(0.01)  # Simulate async work
            return x + y

        result = await async_add(5, 10)

        assert result == 15
        context_instance.push_active_function.assert_called_once()
        call_args = context_instance.push_active_function.call_args
        assert call_args[0][0] == "async_add"
        assert call_args[1]['input_data'] == (5, 10, {})
        manager.set_output.assert_called_once_with(15)

    def test_sync_generator_decoration(self, mock_context):
        """Test decoration of sync generator functions."""
        context_instance, manager = mock_context

        @track_unregistered_function
        def count_up_to(n: int) -> Generator[int, None, None]:
            for i in range(n):
                yield i

        results = list(count_up_to(3))

        assert results == [0, 1, 2]
        context_instance.push_active_function.assert_called_once()
        call_args = context_instance.push_active_function.call_args
        assert call_args[0][0] == "count_up_to"
        assert call_args[1]['input_data'] == (3, {})
        manager.set_output.assert_called_once_with([0, 1, 2])

    async def test_async_generator_decoration(self, mock_context):
        """Test decoration of async generator functions."""
        context_instance, manager = mock_context

        @track_unregistered_function
        async def async_count_up_to(n: int) -> AsyncGenerator[int, None]:
            for i in range(n):
                await asyncio.sleep(0.001)  # Simulate async work
                yield i

        results = []
        async for value in async_count_up_to(3):
            results.append(value)

        assert results == [0, 1, 2]
        context_instance.push_active_function.assert_called_once()
        call_args = context_instance.push_active_function.call_args
        assert call_args[0][0] == "async_count_up_to"
        assert call_args[1]['input_data'] == (3, {})
        manager.set_output.assert_called_once_with([0, 1, 2])

    def test_function_with_kwargs(self, mock_context):
        """Test function decoration with keyword arguments."""
        context_instance, manager = mock_context

        @track_unregistered_function
        def calculate(base: int, multiplier: int = 2, offset: int = 0) -> int:
            return base * multiplier + offset

        result = calculate(5, multiplier=3, offset=10)

        assert result == 25
        call_args = context_instance.push_active_function.call_args
        expected_input = (5, {'multiplier': 3, 'offset': 10})
        assert call_args[1]['input_data'] == expected_input
        manager.set_output.assert_called_once_with(25)

    def test_decorator_preserves_function_attributes(self):
        """Test that the decorator preserves original function attributes."""

        @track_unregistered_function
        def original_function(x: int) -> int:
            """This is a test function."""
            return x * 2

        assert original_function.__name__ == "original_function"
        assert original_function.__doc__ is not None
        assert "This is a test function." in original_function.__doc__

    def test_no_parentheses_vs_with_parentheses(self, mock_context):
        """Test that both @decorator and @decorator() syntax work."""
        context_instance, _ = mock_context

        # Without parentheses
        @track_unregistered_function
        def func1(x: int) -> int:
            return x

        # With parentheses
        @track_unregistered_function()
        def func2(x: int) -> int:
            return x

        result1 = func1(42)
        result2 = func2(42)

        assert result1 == 42
        assert result2 == 42
        assert context_instance.push_active_function.call_count == 2

    def test_manual_decorator_application(self, mock_context):
        """Test manual application of decorator without @ syntax."""
        context_instance, _ = mock_context

        def original_func(data: str) -> str:
            return data.upper()

        # Apply decorator manually
        decorated_func = track_unregistered_function(original_func, name="manual_handler")

        result = decorated_func("hello")

        assert result == "HELLO"
        call_args = context_instance.push_active_function.call_args
        assert call_args[0][0] == "manual_handler"
