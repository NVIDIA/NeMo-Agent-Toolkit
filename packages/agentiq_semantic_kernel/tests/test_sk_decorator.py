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

from pydantic import BaseModel

# Import the semantic_kernel_tool_wrapper from tool_wrapper.py
from aiq.plugins.semantic_kernel.tool_wrapper import semantic_kernel_tool_wrapper

# ----------------------------
# Dummy Models for Testing
# ----------------------------


class DummyInput(BaseModel):
    value: int


class DummyOutput(BaseModel):
    result: int


# Models for nested type testing
class InnerModel(BaseModel):
    x: int


class OuterModel(BaseModel):
    inner: InnerModel
    y: str


class NestedOutput(BaseModel):
    result: int


# ----------------------------
# Dummy Function Implementations
# ----------------------------


class DummyFunction:
    """Dummy function with simple input/output."""

    def __init__(self):
        self.description = "Dummy description"
        # Create a simple config object with attribute 'type'
        self.config = type('Config', (), {'type': 'dummy_func'})
        self.has_single_output = True
        self.has_streaming_output = False
        self.input_schema = DummyInput
        self.single_output_schema = DummyOutput
        self.streaming_output_schema = None

    async def acall_invoke(self, *args, **kwargs):
        # For testing, simply multiply the input value by 2
        input_obj = args[0]
        return DummyOutput(result=input_obj.value * 2)


class DummyNestedFunction:
    """Dummy function using a nested BaseModel for input."""

    def __init__(self):
        self.description = "Nested function"
        self.config = type('Config', (), {'type': 'nested_func'})
        self.has_single_output = True
        self.has_streaming_output = False
        self.input_schema = OuterModel
        self.single_output_schema = NestedOutput
        self.streaming_output_schema = None

    async def acall_invoke(self, *args, **kwargs):
        # For testing, sum inner.x and the length of y
        outer = args[0]
        return NestedOutput(result=outer.inner.x + len(outer.y))


class DummyStreamingFunction:
    """Dummy function that simulates a streaming output."""

    def __init__(self):
        self.description = "Streaming function"
        self.config = type('Config', (), {'type': 'streaming_func'})
        self.has_single_output = False
        self.has_streaming_output = True
        self.input_schema = DummyInput
        self.streaming_output_schema = DummyOutput
        self.single_output_schema = None

    async def acall_stream(self, *args, **kwargs):
        # For simplicity, return the first value from the streaming generator
        async for item in self._astream(args[0]):
            yield item

    async def _astream(self, value):
        for i in range(3):
            yield DummyOutput(result=value.value + i)


# ----------------------------
# Pytest Unit Tests
# ----------------------------


async def test_semantic_kernel_tool_wrapper_simple_arguments():
    """Test the tool wrapper with a function that has simple arguments."""
    dummy_fn = DummyFunction()
    # Invoke the semantic kernel tool wrapper
    wrapper = semantic_kernel_tool_wrapper('dummy_func', dummy_fn, builder=None)

    # Ensure the wrapper returns a dictionary with our function name as key
    assert 'dummy_func' in wrapper
    decorated_func = wrapper['dummy_func']

    # Check that kernel function attributes are set
    assert hasattr(decorated_func, '__kernel_function__')
    assert decorated_func.__kernel_function__ is True
    assert decorated_func.__kernel_function_name__ == dummy_fn.config.type
    assert decorated_func.__kernel_function_description__ == dummy_fn.description

    # Check that __kernel_function_parameters__ contains the expected parameter
    params = getattr(decorated_func, '__kernel_function_parameters__')
    # DummyInput has one field 'value'
    assert isinstance(params, list)
    assert any(param['name'] == 'value' for param in params)

    # Check the __kernel_function_streaming__ attribute (should be False for single output)
    assert getattr(decorated_func, '__kernel_function_streaming__') is False

    # Call the decorated function with a simple DummyInput
    dummy_input = DummyInput(value=5)
    result = await decorated_func(dummy_input)
    # Expect the output to be value * 2
    assert result.result == 10

    # Also check return type info (for DummyOutput, field 'result' is int)
    return_type = getattr(decorated_func, '__kernel_function_return_type__')
    assert return_type == 'int'


async def test_semantic_kernel_tool_wrapper_nested_base_model():
    """Test the tool wrapper with a function that uses nested BaseModel types in its input."""
    dummy_fn = DummyNestedFunction()
    wrapper = semantic_kernel_tool_wrapper('nested_func', dummy_fn, builder=None)

    assert 'nested_func' in wrapper
    decorated_func = wrapper['nested_func']

    # Extract kernel function parameters
    params = getattr(decorated_func, '__kernel_function_parameters__')
    # OuterModel has two fields: 'inner' (a nested BaseModel) and 'y' (a simple type)
    inner_param = next(param for param in params if param['name'] == 'inner')
    y_param = next(param for param in params if param['name'] == 'y')

    # For nested BaseModel fields, include_in_function_choices should be False
    assert inner_param['include_in_function_choices'] is False
    # For simple types (like str), it should remain True
    assert y_param['include_in_function_choices'] is True

    # Check the __kernel_function_streaming__ attribute (should be False for single output)
    assert getattr(decorated_func, '__kernel_function_streaming__') is False

    # Test function invocation
    dummy_input = OuterModel(inner=InnerModel(x=3), y='test')
    result = await decorated_func(dummy_input)
    # Expected: inner.x (3) + length of 'test' (4) = 7
    assert result.result == 7

    # Check return type info
    return_type = getattr(decorated_func, '__kernel_function_return_type__')
    assert return_type == 'int'


async def test_semantic_kernel_tool_wrapper_streaming():
    """Test the tool wrapper with a function that has streaming output."""
    dummy_fn = DummyStreamingFunction()
    wrapper = semantic_kernel_tool_wrapper('streaming_func', dummy_fn, builder=None)

    assert 'streaming_func' in wrapper
    decorated_func = wrapper['streaming_func']

    # For streaming functions, __kernel_function_streaming__ should be True
    assert getattr(decorated_func, '__kernel_function_streaming__') is True

    dummy_input = DummyInput(value=10)
    results = []
    async for item in decorated_func(dummy_input):
        results.append(item)
    # Verify that we get the complete streaming output from the generator
    # For DummyStreamingFunction, _astream yields three items with result values: value + 0, value + 1, and value + 2
    assert len(results) == 3
    assert results[0].result == 10
    assert results[1].result == 11
    assert results[2].result == 12
