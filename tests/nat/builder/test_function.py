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
from types import NoneType

import pytest
from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.function import Function
from nat.builder.function import LambdaFunction
from nat.builder.function_info import FunctionInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class DummyConfig(FunctionBaseConfig, name="dummy"):
    pass


class LambdaFnConfig(FunctionBaseConfig, name="test_lambda"):
    pass


class LambdaStreamFnConfig(FunctionBaseConfig, name="test_lambda_stream"):
    pass


@pytest.fixture(scope="module", autouse=True)
async def _register_lambda_fn():

    @register_function(config_type=LambdaFnConfig)
    async def register(config: LambdaFnConfig, b: Builder):

        async def _inner(some_input: str) -> str:
            return some_input + "!"

        def _convert(int_input: int) -> str:
            return str(int_input)

        yield FunctionInfo.from_fn(_inner, converters=[_convert])


@pytest.fixture(scope="module", autouse=True)
async def _register_lambda_stream_fn():

    @register_function(config_type=LambdaStreamFnConfig)
    async def register(config: LambdaStreamFnConfig, b: Builder):

        async def _inner_stream(some_input: str) -> AsyncGenerator[str]:
            yield some_input + "!"

        def _convert(int_input: int) -> str:
            return str(int_input)

        yield FunctionInfo.from_fn(_inner_stream, converters=[_convert])


async def test_direct_create_with_lambda():

    async with WorkflowBuilder() as builder:

        fn_obj = await builder.add_function(name="test_function", config=LambdaFnConfig())

        assert isinstance(fn_obj, LambdaFunction)

        assert await fn_obj.ainvoke("test", to_type=str) == "test!"


async def test_direct_create_with_class():

    class ClassFnConfig(FunctionBaseConfig, name="test_class"):
        pass

    class TestFunction(Function[str, str, None]):

        def __init__(self, config: ClassFnConfig):
            super().__init__(config=config, description="Test function")

        def some_method(self, val):
            return "some_method" + val

        async def _ainvoke(self, value: str) -> str:
            return value + "!"

        async def _astream(self, value: str):
            yield value + "!"

    @register_function(config_type=ClassFnConfig)
    async def _register(config: ClassFnConfig, b: Builder):

        yield TestFunction(config)

    async with WorkflowBuilder() as builder:

        fn_obj = await builder.add_function(name="test_function", config=ClassFnConfig())

        assert isinstance(fn_obj, TestFunction)

        assert fn_obj.some_method("test") == "some_methodtest"

        assert await fn_obj.ainvoke("test", to_type=str) == "test!"


async def test_functions_call_functions():

    class ChainedFnConfig(FunctionBaseConfig, name="test_chained"):
        function_name: str

    @register_function(config_type=ChainedFnConfig)
    async def _register(config: ChainedFnConfig, b: Builder):

        other_fn = b.get_function(config.function_name)

        async def _inner(some_input: str) -> str:
            return await other_fn.ainvoke(some_input, to_type=str) + "!"

        yield _inner

    async with WorkflowBuilder() as builder:

        await builder.add_function(name="test_function", config=LambdaFnConfig())

        fn_obj = await builder.add_function(name="second_function",
                                            config=ChainedFnConfig(function_name="test_function"))

        assert isinstance(fn_obj, LambdaFunction)

        assert await fn_obj.ainvoke("test", to_type=str) == "test!!"


async def test_functions_single_pod_input_pod_output():

    async with WorkflowBuilder() as builder:

        fn_obj = await builder.add_function(name="test_function", config=LambdaFnConfig())

        assert fn_obj.input_type is str
        assert fn_obj.single_output_type is str
        assert fn_obj.streaming_output_type is str

        # Invoke with actual input
        assert await fn_obj.ainvoke("test", to_type=str) == "test!"

        # Invoke with input schema as dict
        assert await fn_obj.ainvoke({"some_input": "test2"}, to_type=str) == "test2!"

        # Invoke with input schema as pydantic model
        assert await fn_obj.ainvoke(fn_obj.input_schema.model_validate({"some_input": "test3"}),
                                    to_type=str) == "test3!"

        # Invoke with input as int using converter
        assert await fn_obj.ainvoke(4, to_type=str) == "4!"

        # Invoke with input which is not convertible
        with pytest.raises(TypeError):
            await fn_obj.ainvoke([4.5], to_type=str)


async def test_functions_single_dict_input_pod_output():

    @register_function(config_type=DummyConfig)
    async def _register(config: DummyConfig, b: Builder):

        async def _inner(dict_input: dict[int, typing.Any]) -> str:
            return dict_input[0] + "!"

        yield _inner

    async with WorkflowBuilder() as builder:

        fn_obj = await builder.add_function(name="test_function", config=DummyConfig())

        assert fn_obj.input_type == dict[int, typing.Any]
        assert fn_obj.input_class == dict
        assert fn_obj.single_output_type == str
        assert fn_obj.streaming_output_type == str

        assert await fn_obj.ainvoke({0: "test"}, to_type=str) == "test!"

        assert await fn_obj.ainvoke(fn_obj.input_schema.model_validate({"dict_input": {
            0: "test3"
        }}), to_type=str) == "test3!"


async def test_functions_multi_pod_input_pod_output():

    @register_function(config_type=DummyConfig)
    async def _register(config: DummyConfig, b: Builder):

        async def _inner(first_input: str, second_input: int, third_input: list[str]) -> str:
            return first_input + str(second_input) + str(third_input) + "!"

        yield _inner

    async with WorkflowBuilder() as builder:

        fn_obj = await builder.add_function(name="test_function", config=DummyConfig())

        assert fn_obj.input_type == fn_obj.input_schema
        assert issubclass(fn_obj.input_type, BaseModel)

        assert await fn_obj.ainvoke({
            "first_input": "test", "second_input": 4, "third_input": ["a", "b", "c"]
        },
                                    to_type=str) == "test4['a', 'b', 'c']!"

        assert await fn_obj.ainvoke(fn_obj.input_schema.model_validate({
            "first_input": "test", "second_input": 2, "third_input": ["a", "b", "c"]
        }),
                                    to_type=str) == "test2['a', 'b', 'c']!"


async def test_stream_functions_single_pod_input_pod_output():

    async with WorkflowBuilder() as builder:

        fn_obj = await builder.add_function(name="test_function", config=LambdaStreamFnConfig())

        assert fn_obj.input_type is str
        assert fn_obj.single_output_type == NoneType
        assert fn_obj.streaming_output_type is str

        # Stream output with actual input
        result: None | str = None
        async for output in fn_obj.astream("test", to_type=str):
            result = output
        assert result == "test!"

        # Stream output with actual input and to_type set to None
        result: None | str = None
        async for output in fn_obj.astream("test", to_type=None):
            result = output
        assert result == "test!"

        # Stream output with input schema as dict
        result: None | dict = None
        async for output in fn_obj.astream({"some_input": "test2"}, to_type=str):
            result = output
        assert result == "test2!"

        # Stream output with input schema as dict to_type set to None
        result: None | dict = None
        async for output in fn_obj.astream({"some_input": "test2"}, to_type=None):
            result = output
        assert result == "test2!"

        # Stream output with input schema as pydantic model
        result: None | BaseModel = None
        async for output in fn_obj.astream(fn_obj.input_schema.model_validate({"some_input": "test3"}), to_type=str):
            result = output
        assert result == "test3!"

        # Stream output with input schema as pydantic model to_type set to None
        result: None | BaseModel = None
        async for output in fn_obj.astream(fn_obj.input_schema.model_validate({"some_input": "test3"}), to_type=None):
            result = output
        assert result == "test3!"

        # Stream output with input as int using converter
        result: None | BaseModel = None
        async for output in fn_obj.astream(4, to_type=str):
            result = output
        assert result == "4!"

        # Stream output with input as int using converter and to_type set to None
        result: None | BaseModel = None
        async for output in fn_obj.astream(4, to_type=None):
            result = output
        assert result == "4!"

        # Stream output with input which is not convertible
        result: None | BaseModel = None
        with pytest.raises(TypeError):
            async for output in fn_obj.astream([4.5], to_type=str):
                result = output

        # Stream output with input which is not convertible and to_type set to None
        result: None | BaseModel = None
        with pytest.raises(TypeError):
            async for output in fn_obj.astream([4.5], to_type=None):
                result = output


async def test_stream_functions_single_dict_input_pod_output():

    @register_function(config_type=DummyConfig)
    async def _register(config: DummyConfig, b: Builder):

        async def _stream_inner(dict_input: dict[int, typing.Any]) -> AsyncGenerator[str]:
            yield dict_input[0] + "!"

        yield _stream_inner

    async with WorkflowBuilder() as builder:

        fn_obj = await builder.add_function(name="test_function", config=DummyConfig())

        assert fn_obj.input_type == dict[int, typing.Any]
        assert fn_obj.input_class == dict
        assert fn_obj.single_output_type == NoneType
        assert fn_obj.streaming_output_type == str

        # Stream output with input which is not convertible
        result: None | str = None
        async for output in fn_obj.astream({0: "test"}, to_type=str):
            result = output
        assert result == "test!"

        # Stream output with input which is not convertible and to_type set to None
        result: None | str = None
        async for output in fn_obj.astream({0: "test"}, to_type=None):
            result = output
        assert result == "test!"

        # Stream output with input which is not convertible
        result: None | str = None
        async for output in fn_obj.astream(fn_obj.input_schema.model_validate({"dict_input": {
                0: "test3"
        }}),
                                           to_type=str):
            result = output
        assert result == "test3!"

        # Stream output with input which is not convertible and to_type set to None
        result: None | str = None
        async for output in fn_obj.astream(fn_obj.input_schema.model_validate({"dict_input": {
                0: "test3"
        }}),
                                           to_type=str):
            result = output
        assert result == "test3!"


async def test_stream_functions_multi_pod_input_pod_output():

    @register_function(config_type=DummyConfig)
    async def _register(config: DummyConfig, b: Builder):

        async def _stream_inner(first_input: str, second_input: int, third_input: list[str]) -> AsyncGenerator[str]:
            yield first_input + str(second_input) + str(third_input) + "!"

        yield _stream_inner

    async with WorkflowBuilder() as builder:

        fn_obj = await builder.add_function(name="test_function", config=DummyConfig())

        assert fn_obj.input_type == fn_obj.input_schema
        assert issubclass(fn_obj.input_type, BaseModel)

        # Stream output with input which is not convertible
        result: None | str
        async for output in fn_obj.astream({
                "first_input": "test", "second_input": 4, "third_input": ["a", "b", "c"]
        },
                                           to_type=str):
            result = output
        assert result == "test4['a', 'b', 'c']!"

        # Stream output with input which is not convertible and to_type set to None
        result: None | str
        async for output in fn_obj.astream({
                "first_input": "test", "second_input": 4, "third_input": ["a", "b", "c"]
        },
                                           to_type=None):
            result = output
        assert result == "test4['a', 'b', 'c']!"

        # Stream output with input which is not convertible
        result: None | str
        async for output in fn_obj.astream(fn_obj.input_schema.model_validate({
                "first_input": "test", "second_input": 2, "third_input": ["a", "b", "c"]
        }),
                                           to_type=str):
            result = output
        assert result == "test2['a', 'b', 'c']!"

        # Stream output with input which is not convertible and to_type set to None
        result: None | str
        async for output in fn_obj.astream(fn_obj.input_schema.model_validate({
                "first_input": "test", "second_input": 2, "third_input": ["a", "b", "c"]
        }),
                                           to_type=None):
            result = output
        assert result == "test2['a', 'b', 'c']!"


async def test_auto_streaming_conversion():

    class AutoStreamOutput(BaseModel):
        output: str

    @register_function(config_type=DummyConfig)
    async def _register(config: DummyConfig, b: Builder):

        async def _inner(message: str) -> AutoStreamOutput:
            return AutoStreamOutput(output=message + "!")

        yield _inner

    async with WorkflowBuilder() as builder:

        fn_obj = await builder.add_function(name="test_function", config=DummyConfig())

        assert fn_obj.has_single_output

        # We expect that a streaming conversion is added for single only functions
        assert fn_obj.has_streaming_output

        # Single output and the streaming output should be the same
        assert fn_obj.single_output_type == fn_obj.streaming_output_type
        assert fn_obj.single_output_class == fn_obj.single_output_class
        assert fn_obj.single_output_schema == fn_obj.streaming_output_schema

        assert (await fn_obj.ainvoke("test", to_type=AutoStreamOutput)).output == "test!"

        stream_results = []

        async for result in fn_obj.astream("test", to_type=AutoStreamOutput):
            stream_results.append(result.output)

        assert stream_results == ["test!"]


async def test_auto_streaming_conversion_multi_pod_input_pod_output():

    class AutoStreamOutput(BaseModel):
        output: str

    @register_function(config_type=DummyConfig)
    async def _register(config: DummyConfig, b: Builder):

        async def _inner(first_input: str, second_input: int, third_input: list[str]) -> AutoStreamOutput:
            return AutoStreamOutput(output=first_input + str(second_input) + str(third_input) + "!")

        yield _inner

    async with WorkflowBuilder() as builder:

        fn_obj = await builder.add_function(name="test_function", config=DummyConfig())

        assert fn_obj.has_single_output

        # We expect that a streaming conversion is added for single only functions
        assert fn_obj.has_streaming_output

        # Single output and the streaming output should be the same
        assert fn_obj.single_output_type == fn_obj.streaming_output_type
        assert fn_obj.single_output_class == fn_obj.single_output_class
        assert fn_obj.single_output_schema == fn_obj.streaming_output_schema

        assert (await fn_obj.ainvoke({
            "first_input": "test", "second_input": 4, "third_input": ["a", "b", "c"]
        },
                                     to_type=AutoStreamOutput)).output == "test4['a', 'b', 'c']!"

        stream_results = []

        async for result in fn_obj.astream({
                "first_input": "test", "second_input": 4, "third_input": ["a", "b", "c"]
        },
                                           to_type=AutoStreamOutput):
            stream_results.append(result.output)

        assert stream_results == ["test4['a', 'b', 'c']!"]


async def test_manual_single_to_stream_conversion():

    class TestOutput(BaseModel):
        output: str

    class TestOutputChunk(BaseModel):
        output_char: str

    @register_function(config_type=DummyConfig)
    async def _register(config: DummyConfig, b: Builder):

        async def _inner(message: str) -> TestOutput:
            return TestOutput(output=message + "!")

        async def _convert_to_stream(message: TestOutput) -> AsyncGenerator[TestOutputChunk]:
            for char in message.output:
                yield TestOutputChunk(output_char=char)

        yield FunctionInfo.create(single_fn=_inner, single_to_stream_fn=_convert_to_stream)

    async with WorkflowBuilder() as builder:

        fn_obj = await builder.add_function(name="test_function", config=DummyConfig())

        # Check the output types
        assert fn_obj.has_single_output
        assert fn_obj.has_streaming_output
        assert fn_obj.single_output_type == TestOutput
        assert fn_obj.streaming_output_type == TestOutputChunk

        # Sanity check
        assert (await fn_obj.ainvoke("test", to_type=TestOutput)).output == "test!"

        stream_results = []

        async for result in fn_obj.astream("test", to_type=TestOutputChunk):
            stream_results.append(result.output_char)

        assert "".join(stream_results) == "test!"


async def test_manual_stream_to_single_conversion():

    class TestOutputChunk(BaseModel):
        output_char: str

    class TestOutput(BaseModel):
        output: str

    @register_function(config_type=DummyConfig)
    async def _register(config: DummyConfig, b: Builder):

        async def _inner(message: str) -> AsyncGenerator[TestOutputChunk]:
            for char in (message + "!"):
                yield TestOutputChunk(output_char=char)

        async def _convert_to_single(message: AsyncGenerator[TestOutputChunk]) -> TestOutput:
            output = ""

            async for chunk in message:
                output += chunk.output_char

            return TestOutput(output=output)

        yield FunctionInfo.create(stream_fn=_inner, stream_to_single_fn=_convert_to_single)

    async with WorkflowBuilder() as builder:

        fn_obj = await builder.add_function(name="test_function", config=DummyConfig())

        # Check the output types
        assert fn_obj.has_single_output
        assert fn_obj.has_streaming_output
        assert fn_obj.single_output_type == TestOutput
        assert fn_obj.streaming_output_type == TestOutputChunk

        # Sanity check
        stream_results = []

        async for result in fn_obj.astream("test", to_type=TestOutputChunk):
            stream_results.append(result.output_char)

        assert "".join(stream_results) == "test!"

        assert (await fn_obj.ainvoke("test", to_type=TestOutput)).output == "test!"


async def test_ainvoke_output_type_conversion_failure():
    """Test that ainvoke raises an exception when output cannot be converted to the specified to_type."""

    class UnconvertibleOutput(BaseModel):
        value: str

    class IncompatibleType(BaseModel):
        different_field: int

    @register_function(config_type=DummyConfig)
    async def _register(config: DummyConfig, b: Builder):

        async def _inner(message: str) -> UnconvertibleOutput:
            return UnconvertibleOutput(value=message + "!")

        yield _inner

    async with WorkflowBuilder() as builder:

        fn_obj = await builder.add_function(name="test_function", config=DummyConfig())

        # Verify normal operation works
        result = await fn_obj.ainvoke("test", to_type=UnconvertibleOutput)
        assert result.value == "test!"

        # Test that conversion to incompatible type raises ValueError
        with pytest.raises(ValueError, match="Cannot convert type .* to .* No match found"):
            await fn_obj.ainvoke("test", to_type=IncompatibleType)


async def test_astream_output_type_conversion_failure():
    """Test that astream raises an exception when output cannot be converted to the specified to_type."""

    class UnconvertibleOutput(BaseModel):
        value: str

    class IncompatibleType(BaseModel):
        different_field: int

    @register_function(config_type=DummyConfig)
    async def _register(config: DummyConfig, b: Builder):

        async def _stream_inner(message: str) -> AsyncGenerator[UnconvertibleOutput]:
            yield UnconvertibleOutput(value=message + "!")

        yield _stream_inner

    async with WorkflowBuilder() as builder:

        fn_obj = await builder.add_function(name="test_function", config=DummyConfig())

        # Verify normal operation works
        result = None
        async for output in fn_obj.astream("test", to_type=UnconvertibleOutput):
            result = output
        assert result.value == "test!"

        # Test that conversion to incompatible type raises ValueError during streaming
        with pytest.raises(ValueError, match="Cannot convert type .* to .* No match found"):
            async for output in fn_obj.astream("test", to_type=IncompatibleType):
                pass  # The exception should be raised during the first iteration


async def test_ainvoke_primitive_type_conversion_failure():
    """Test that ainvoke raises an exception when a primitive output cannot be converted to an incompatible type."""

    @register_function(config_type=DummyConfig)
    async def _register(config: DummyConfig, b: Builder):

        async def _inner(message: str) -> str:
            return message + "!"

        yield _inner

    async with WorkflowBuilder() as builder:

        fn_obj = await builder.add_function(name="test_function", config=DummyConfig())

        # Verify normal operation works
        result = await fn_obj.ainvoke("test", to_type=str)
        assert result == "test!"

        # Test that conversion to incompatible type raises ValueError
        # Try to convert string output to a complex type that has no converter
        with pytest.raises(ValueError, match="Cannot convert type .* to .* No match found"):
            await fn_obj.ainvoke("test", to_type=dict)


async def test_astream_primitive_type_conversion_failure():
    """Test that astream raises an exception when a primitive output cannot be converted to an incompatible type."""

    @register_function(config_type=DummyConfig)
    async def _register(config: DummyConfig, b: Builder):

        async def _stream_inner(message: str) -> AsyncGenerator[str]:
            yield message + "!"

        yield _stream_inner

    async with WorkflowBuilder() as builder:

        fn_obj = await builder.add_function(name="test_function", config=DummyConfig())

        # Verify normal operation works
        result = None
        async for output in fn_obj.astream("test", to_type=str):
            result = output
        assert result == "test!"

        # Test that conversion to incompatible type raises ValueError during streaming
        # Try to convert string output to a complex type that has no converter
        with pytest.raises(ValueError, match="Cannot convert type .* to .* No match found"):
            async for output in fn_obj.astream("test", to_type=dict):
                pass  # The exception should be raised during the first iteration
