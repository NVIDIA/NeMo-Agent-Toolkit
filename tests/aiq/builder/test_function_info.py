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
import typing
from collections.abc import AsyncGenerator
from collections.abc import Callable
from types import NoneType

import pytest
from pydantic import BaseModel
from pydantic import Field

from aiq.builder.function_info import FunctionDescriptor
from aiq.builder.function_info import FunctionInfo


def _compare_dicts_partial(test_dict: dict, valid_dict: dict):

    for key, value in valid_dict.items():

        if key not in test_dict:
            return False

        if isinstance(value, dict):
            if not _compare_dicts_partial(test_dict[key], value):
                return False

        elif value != test_dict[key]:
            return False

    return True


class SingleInputModel(BaseModel):
    data: str


class MultipleInputModel(BaseModel):
    data: str
    data2: float
    data3: int


# Int to String functions
async def fn_int_to_str(param: int) -> str:
    return str(param)


async def fn_int_annotated_to_str(param: typing.Annotated[int, ...]) -> str:
    return str(param)


async def fn_int_to_str_annotated(param: int) -> typing.Annotated[str, ...]:
    return str(param)


async def fn_int_annotated_to_str_annotated(param: typing.Annotated[int, ...]) -> typing.Annotated[str, ...]:
    return str(param)


# Int to String streaming functions
async def fn_int_to_str_stream(param: int) -> AsyncGenerator[str]:
    yield str(param)


async def fn_int_annotated_to_str_stream(param: typing.Annotated[int, ...]) -> AsyncGenerator[str]:
    yield str(param)


async def fn_int_to_str_annotated_stream(param: int) -> typing.Annotated[AsyncGenerator[str], ...]:
    yield str(param)


async def fn_int_annotated_to_str_annotated_stream(
        param: typing.Annotated[int, ...]) -> typing.Annotated[AsyncGenerator[str], ...]:
    yield str(param)


# Multiple arguments to string functions
async def fn_multiple_args_to_str(param1: int, param2: MultipleInputModel) -> str:
    return str(param1) + str(param2)


async def fn_multiple_args_annotated_to_str(param1: typing.Annotated[int, ...],
                                            param2: typing.Annotated[MultipleInputModel, ...]) -> str:
    return str(param1) + str(param2)


async def fn_multiple_args_to_str_annotated(param1: int, param2: MultipleInputModel) -> typing.Annotated[str, ...]:
    return str(param1) + str(param2)


async def fn_multiple_args_annotated_to_str_annotated(
        param1: typing.Annotated[int, ...], param2: typing.Annotated[MultipleInputModel,
                                                                     ...]) -> typing.Annotated[str, ...]:
    return str(param1) + str(param2)


# Multiple arguments to string streaming functions
async def fn_multiple_args_to_str_stream(param1: int, param2: MultipleInputModel) -> AsyncGenerator[str]:
    yield str(param1) + str(param2)


async def fn_multiple_args_annotated_to_str_stream(
        param1: typing.Annotated[int, ...], param2: typing.Annotated[MultipleInputModel, ...]) -> AsyncGenerator[str]:
    yield str(param1) + str(param2)


async def fn_multiple_args_to_str_annotated_stream(
        param1: int, param2: MultipleInputModel) -> typing.Annotated[AsyncGenerator[str], ...]:
    yield str(param1) + str(param2)


async def fn_multiple_args_annotated_to_str_annotated_stream(
        param1: typing.Annotated[int, ...],
        param2: typing.Annotated[MultipleInputModel, ...]) -> typing.Annotated[AsyncGenerator[str], ...]:
    yield str(param1) + str(param2)


# Union arguments single functions
async def fn_union_to_str(param: int | float) -> str:
    return str(param)


async def fn_int_to_union(param: int) -> str | float:
    if (param > 5):
        return str(param)

    return param


async def fn_union_to_union(param: int | float) -> str | float:
    if (param > 5):
        return str(param)

    return param


async def fn_union_annotated_to_str(param: typing.Annotated[int | float, ...]) -> str:
    return str(param)


async def fn_int_to_union_annotated(param: int) -> typing.Annotated[str | float, ...]:
    if (param > 5):
        return str(param)

    return param


# Union arguments streaming functions
async def fn_union_to_str_stream(param: int | float) -> AsyncGenerator[str]:
    yield str(param)


async def fn_int_to_union_stream(param: int) -> AsyncGenerator[str | float]:
    if (param > 5):
        yield str(param)
    else:
        yield param


async def fn_union_to_union_stream(param: int | float) -> AsyncGenerator[str | float]:
    if (param > 5):
        yield str(param)
    else:
        yield param


async def fn_union_annotated_to_str_stream(param: typing.Annotated[int | float, ...]) -> AsyncGenerator[str]:
    yield str(param)


async def fn_int_to_union_annotated_stream(param: int) -> typing.Annotated[AsyncGenerator[str | float], ...]:
    if (param > 5):
        yield str(param)
    else:
        yield param


# Base model arguments single
async def fn_base_model_to_str(param: SingleInputModel) -> str:
    return str(param)


async def fn_int_to_base_model(param: int) -> SingleInputModel:
    return SingleInputModel(data=str(param))


async def fn_base_model_to_base_model(param: SingleInputModel) -> SingleInputModel:
    return param


# Base model arguments streaming
async def fn_base_model_to_str_stream(param: SingleInputModel) -> AsyncGenerator[str]:
    yield str(param)


async def fn_int_to_base_model_stream(param: int) -> AsyncGenerator[SingleInputModel]:
    yield SingleInputModel(data=str(param))


async def fn_base_model_to_base_model_stream(param: SingleInputModel) -> AsyncGenerator[SingleInputModel]:
    yield param


schema_input_int = {
    "properties": {
        "param": {
            "type": "integer"
        }
    },
    "required": ["param"],
}

schema_input_multi = {
    "$defs": {
        "MultipleInputModel": {
            "properties": {
                "data": {
                    "type": "string"
                }, "data2": {
                    "type": "number"
                }, "data3": {
                    "type": "integer"
                }
            },
            "required": ["data", "data2", "data3"],
            "type": "object"
        }
    },
    "properties": {
        "param1": {
            "type": "integer"
        }, "param2": {
            "$ref": "#/$defs/MultipleInputModel"
        }
    },
    "required": ["param1", "param2"],
}

schema_input_union = {
    "properties": {
        "param": {
            "anyOf": [{
                "type": "integer"
            }, {
                "type": "number"
            }]
        }
    },
    "required": ["param"],
}

schema_input_base_model = {
    "$defs": {
        "SingleInputModel": {
            "properties": {
                "data": {
                    "type": "string"
                }
            }, "required": ["data"], "type": "object"
        }
    },
    "properties": {
        "param": {
            "$ref": "#/$defs/SingleInputModel"
        }
    },
    "required": ["param"],
}

schema_output_str = {
    "properties": {
        "value": {
            "type": "string"
        }
    },
    "required": ["value"],
}

schema_output_union = {
    "properties": {
        "value": {
            "anyOf": [{
                "type": "string"
            }, {
                "type": "number"
            }]
        }
    },
    "required": ["value"],
}

schema_output_base_model = {
    "properties": {
        "data": {
            "type": "string"
        }
    },
    "required": ["data"],
    "title": "SingleInputModel",
}


def _build_schema_params(functions: list[tuple[list[Callable], dict, dict]]) -> list[tuple[Callable, dict, dict]]:

    final_params: list[tuple[Callable, dict, dict]] = []

    for function_list, in_schema, out_schema in functions:

        final_params.extend([(fn, in_schema, out_schema) for fn in function_list])

    return final_params


@pytest.mark.parametrize(
    "function, input_schema, output_schema",
    _build_schema_params([
        ([
            fn_int_to_str,
            fn_int_annotated_to_str,
            fn_int_to_str_annotated,
            fn_int_annotated_to_str_annotated,
            fn_int_to_str_stream,
            fn_int_annotated_to_str_stream,
            fn_int_to_str_annotated_stream,
            fn_int_annotated_to_str_annotated_stream,
        ],
         schema_input_int,
         schema_output_str),
        ([
            fn_int_to_union,
            fn_int_to_union_stream,
            fn_int_to_union_annotated,
            fn_int_to_union_annotated_stream,
        ],
         schema_input_int,
         schema_output_union),
        ([
            fn_multiple_args_to_str,
            fn_multiple_args_annotated_to_str,
            fn_multiple_args_to_str_annotated,
            fn_multiple_args_annotated_to_str_annotated,
            fn_multiple_args_to_str_stream,
            fn_multiple_args_annotated_to_str_stream,
            fn_multiple_args_to_str_annotated_stream,
            fn_multiple_args_annotated_to_str_annotated_stream
        ],
         schema_input_multi,
         schema_output_str),
        ([
            fn_union_to_str,
            fn_union_to_str_stream,
            fn_union_annotated_to_str,
            fn_union_annotated_to_str_stream,
        ],
         schema_input_union,
         schema_output_str),
        ([
            fn_union_to_union,
            fn_union_to_union_stream,
        ], schema_input_union, schema_output_union),
        ([
            fn_int_to_base_model,
            fn_int_to_base_model_stream,
        ], schema_input_int, schema_output_base_model),
        ([
            fn_base_model_to_str,
            fn_base_model_to_str_stream,
        ], schema_input_base_model, schema_output_str),
        ([
            fn_base_model_to_base_model,
            fn_base_model_to_base_model_stream,
        ],
         schema_input_base_model,
         schema_output_base_model),
    ]))
def test_schema_from_function(function: Callable, input_schema: dict, output_schema):

    test_desc = FunctionDescriptor.from_function(function)

    in_schema = test_desc.input_schema
    out_schema = test_desc.output_schema

    assert in_schema is not None and in_schema != type[None]
    assert out_schema is not None and out_schema != type[None]

    assert _compare_dicts_partial(in_schema.model_json_schema(), input_schema)
    assert _compare_dicts_partial(out_schema.model_json_schema(), output_schema)


def test_constructor():

    test_desc = FunctionDescriptor.from_function(fn_int_to_str)

    schema_in = test_desc.input_schema
    schema_out = test_desc.output_schema

    assert schema_in is not None and schema_in != NoneType
    assert schema_out is not None and schema_out != NoneType

    # Test no functions provided
    with pytest.raises(ValueError):
        info = FunctionInfo(input_schema=NoneType, single_output_schema=NoneType, stream_output_schema=NoneType)

    # Test no input schema provided
    with pytest.raises(ValueError):
        info = FunctionInfo(single_fn=fn_int_to_str,
                            input_schema=NoneType,
                            single_output_schema=schema_in,
                            stream_output_schema=schema_in)

    # Test no single output schema provided
    with pytest.raises(ValueError):
        info = FunctionInfo(single_fn=fn_int_to_str,
                            input_schema=schema_in,
                            single_output_schema=NoneType,
                            stream_output_schema=NoneType)

    # Test no stream output schema provided
    with pytest.raises(ValueError):
        info = FunctionInfo(stream_fn=fn_int_to_str_stream,
                            input_schema=schema_in,
                            single_output_schema=NoneType,
                            stream_output_schema=NoneType)

    # Test extra stream schema provided
    with pytest.raises(ValueError):
        info = FunctionInfo(single_fn=fn_int_to_str,
                            input_schema=schema_in,
                            single_output_schema=schema_in,
                            stream_output_schema=schema_in)

    # Test extra single schema provided
    with pytest.raises(ValueError):
        info = FunctionInfo(stream_fn=fn_int_to_str_stream,
                            input_schema=schema_in,
                            single_output_schema=schema_in,
                            stream_output_schema=schema_in)

    # Test differing single and stream input types
    with pytest.raises(ValueError):
        info = FunctionInfo(single_fn=fn_multiple_args_to_str,
                            stream_fn=fn_int_to_str_stream,
                            input_schema=schema_in,
                            single_output_schema=schema_in,
                            stream_output_schema=schema_in)

    # Negative test, multiple arguments to single function
    with pytest.raises(ValueError):
        info = FunctionInfo(single_fn=fn_multiple_args_to_str,
                            input_schema=schema_in,
                            single_output_schema=schema_out,
                            stream_output_schema=NoneType)

    # Negative test, multiple arguments to stream function
    with pytest.raises(ValueError):
        info = FunctionInfo(stream_fn=fn_multiple_args_to_str_stream,
                            input_schema=schema_in,
                            single_output_schema=NoneType,
                            stream_output_schema=schema_out)

    # Positing single only test
    info = FunctionInfo(single_fn=fn_int_to_str,
                        input_schema=schema_in,
                        single_output_schema=schema_out,
                        stream_output_schema=NoneType)

    assert info.single_fn is fn_int_to_str
    assert info.stream_fn is None
    assert info.input_schema == schema_in
    assert info.single_output_schema == schema_out
    assert info.stream_output_schema is NoneType
    assert info.input_type == int
    assert info.single_output_type == str
    assert info.stream_output_type is NoneType

    # Positive stream only test
    info = FunctionInfo(stream_fn=fn_int_to_str_stream,
                        input_schema=schema_in,
                        single_output_schema=NoneType,
                        stream_output_schema=schema_out)

    assert info.single_fn is None
    assert info.stream_fn is fn_int_to_str_stream
    assert info.input_schema == schema_in
    assert info.single_output_schema is NoneType
    assert info.stream_output_schema == schema_out
    assert info.input_type == int
    assert info.single_output_type is NoneType
    assert info.stream_output_type == str

    # Positive single and stream test
    info = FunctionInfo(single_fn=fn_int_to_str,
                        stream_fn=fn_int_to_str_stream,
                        input_schema=schema_in,
                        single_output_schema=schema_out,
                        stream_output_schema=schema_out)

    assert info.single_fn is fn_int_to_str
    assert info.stream_fn is fn_int_to_str_stream
    assert info.input_schema == schema_in
    assert info.single_output_schema == schema_out
    assert info.stream_output_schema == schema_out
    assert info.input_type == int
    assert info.single_output_type == str
    assert info.stream_output_type == str


@pytest.mark.parametrize("function, input_type, output_type",
                         [
                             (fn_int_to_str, int, str),
                             (fn_int_annotated_to_str, int, str),
                             (fn_int_to_str_annotated, int, str),
                             (fn_int_annotated_to_str_annotated, int, str),
                             (fn_int_to_union, int, str | float),
                             (fn_int_to_union_annotated, int, str | float),
                             (fn_union_to_str, int | float, str),
                             (fn_union_annotated_to_str, int | float, str),
                             (fn_union_to_union, int | float, str | float),
                             (fn_int_to_base_model, int, SingleInputModel),
                             (fn_base_model_to_str, SingleInputModel, str),
                             (fn_base_model_to_base_model, SingleInputModel, SingleInputModel),
                         ])
def test_constructor_single_input_types(function: Callable, input_type: type, output_type: type):

    test_desc = FunctionDescriptor.from_function(function)

    assert test_desc.input_schema is not None
    assert test_desc.output_schema is not None

    info = FunctionInfo(single_fn=function,
                        input_schema=test_desc.input_schema,
                        single_output_schema=test_desc.output_schema,
                        stream_output_schema=NoneType)

    assert info.input_type == input_type
    assert info.single_output_type == output_type


@pytest.mark.parametrize("function, input_type, output_type",
                         [
                             (fn_int_to_str_stream, int, str),
                             (fn_int_annotated_to_str_stream, int, str),
                             (fn_int_to_str_annotated_stream, int, str),
                             (fn_int_annotated_to_str_annotated_stream, int, str),
                             (fn_union_to_str_stream, int | float, str),
                             (fn_union_annotated_to_str_stream, int | float, str),
                             (fn_union_to_union_stream, int | float, str | float),
                             (fn_int_to_base_model_stream, int, SingleInputModel),
                             (fn_base_model_to_str_stream, SingleInputModel, str),
                             (fn_base_model_to_base_model_stream, SingleInputModel, SingleInputModel),
                         ])
def test_constructor_stream_input_types(function: Callable, input_type: type, output_type: type):

    test_desc = FunctionDescriptor.from_function(function)

    assert test_desc.input_schema is not None
    assert test_desc.output_schema is not None

    info = FunctionInfo(stream_fn=function,
                        input_schema=test_desc.input_schema,
                        single_output_schema=NoneType,
                        stream_output_schema=test_desc.output_schema)

    assert info.input_type == input_type
    assert info.stream_output_type == output_type


def test_single_fn_bad_signatures():

    test_desc = FunctionDescriptor.from_function(fn_int_to_str)

    assert test_desc.input_schema is not None
    assert test_desc.output_schema is not None

    schema_int_in = test_desc.input_schema
    schema_int_out = test_desc.output_schema

    async def no_arg_annotation(arg) -> str:
        return "test"

    async def no_return_annotation(arg: int):
        return "test"

    async def multiple_args(arg1: int, arg2: int) -> str:
        return "test"

    with pytest.raises(ValueError):
        FunctionInfo(single_fn=no_arg_annotation,
                     input_schema=schema_int_in,
                     single_output_schema=schema_int_out,
                     stream_output_schema=NoneType)

    with pytest.raises(ValueError):
        FunctionInfo(single_fn=no_return_annotation,
                     input_schema=schema_int_in,
                     single_output_schema=schema_int_out,
                     stream_output_schema=NoneType)

    with pytest.raises(ValueError):
        FunctionInfo(single_fn=multiple_args,
                     input_schema=schema_int_in,
                     single_output_schema=schema_int_out,
                     stream_output_schema=NoneType)


@pytest.mark.parametrize(
    "function, is_streaming, input_type, output_type, input_schema, output_schema",
    [
        (fn_int_to_str, False, int, str, None, None),
        (fn_int_annotated_to_str, False, int, str, None, None),
        (fn_int_to_str_annotated, False, int, str, None, None),
        (fn_int_annotated_to_str_annotated, False, int, str, None, None),
        (fn_int_to_str_stream, True, int, str, None, None),
        (fn_int_annotated_to_str_stream, True, int, str, None, None),
        (fn_int_to_str_annotated_stream, True, int, str, None, None),
        (fn_int_annotated_to_str_annotated_stream, True, int, str, None, None),
        (fn_multiple_args_to_str, False, None, str, None, None),
        (fn_multiple_args_annotated_to_str, False, None, str, None, None),
        (fn_multiple_args_to_str_annotated, False, None, str, None, None),
        (fn_multiple_args_annotated_to_str_annotated, False, None, str, None, None),
        (fn_multiple_args_to_str_stream, True, None, str, None, None),
        (fn_multiple_args_annotated_to_str_stream, True, None, str, None, None),
        (fn_multiple_args_to_str_annotated_stream, True, None, str, None, None),
        (fn_multiple_args_annotated_to_str_annotated_stream, True, None, str, None, None),
        (fn_union_to_str, False, int | float, str, None, None),
        (fn_int_to_union, False, int, str | float, None, None),
        (fn_union_to_union, False, int | float, str | float, None, None),
        (fn_union_to_str_stream, True, int | float, str, None, None),
        (fn_int_to_union_stream, True, int, str | float, None, None),
        (fn_union_to_union_stream, True, int | float, str | float, None, None),
        (fn_union_annotated_to_str, False, int | float, str, None, None),
        (fn_int_to_union_annotated, False, int, str | float, None, None),
        (fn_union_annotated_to_str_stream, True, int | float, str, None, None),
        (fn_int_to_union_annotated_stream, True, int, str | float, None, None),
        (fn_base_model_to_str, False, SingleInputModel, str, SingleInputModel, None),
        (fn_int_to_base_model, False, int, SingleInputModel, None, SingleInputModel),
        (fn_base_model_to_base_model, False, SingleInputModel, SingleInputModel, SingleInputModel, SingleInputModel),
        (fn_base_model_to_str_stream, True, SingleInputModel, str, SingleInputModel, None),
        (fn_int_to_base_model_stream, True, int, SingleInputModel, None, SingleInputModel),
        (fn_base_model_to_base_model_stream,
         True,
         SingleInputModel,
         SingleInputModel,
         SingleInputModel,
         SingleInputModel),
    ])
def test_create_and_from_fn(function: Callable,
                            is_streaming: bool,
                            input_type: type | None,
                            output_type: type | None,
                            input_schema: type[BaseModel] | None,
                            output_schema: type[BaseModel] | None):

    info_from_fn = FunctionInfo.from_fn(function)
    info_create = FunctionInfo.create(single_fn=function if not is_streaming else None,
                                      stream_fn=function if is_streaming else None)

    for info in [info_from_fn, info_create]:
        # If we dont have an input type, we much change something about the function. Skip the assertion
        if (input_type is not None):
            assert info.input_type == input_type

        if (input_schema is not None):
            assert info.input_schema == input_schema

        if is_streaming:

            if (input_type is not None):
                assert info.stream_fn is function

            if (output_type is not None):
                assert info.stream_output_type == output_type

            if (output_schema is not None):
                assert info.stream_output_schema == output_schema

            # When creating a streaming only function, there will be no single
            assert info.single_fn is None
            assert info.single_output_type == NoneType
            assert info.single_output_schema == NoneType

        else:

            # When creating a single only function, we automatically create a streaming one
            if (input_type is not None):
                assert info.single_fn is function
                assert info.stream_fn is not None

            if (output_type is not None):
                assert info.single_output_type == output_type
                assert info.stream_output_type == output_type

            if (output_schema is not None):
                assert info.single_output_schema == output_schema
                assert info.stream_output_schema == output_schema


@pytest.mark.parametrize("function, is_streaming, input_val, output_val",
                         [
                             (fn_int_to_str, False, 10, "10"),
                             (fn_int_annotated_to_str, False, 10, "10"),
                             (fn_int_to_str_annotated, False, 10, "10"),
                             (fn_int_annotated_to_str_annotated, False, 10, "10"),
                             (fn_int_to_str_stream, True, 10, "10"),
                             (fn_int_annotated_to_str_stream, True, 10, "10"),
                             (fn_int_to_str_annotated_stream, True, 10, "10"),
                             (fn_int_annotated_to_str_annotated_stream, True, 10, "10"),
                             (fn_multiple_args_to_str,
                              False, {
                                  "param1": 10, "param2": {
                                      "data": "test", "data2": 10.0, "data3": 7
                                  }
                              },
                              "10{'data': 'test', 'data2': 10.0, 'data3': 7}"),
                             (fn_multiple_args_annotated_to_str,
                              False, {
                                  "param1": 10, "param2": {
                                      "data": "test", "data2": 10.0, "data3": 7
                                  }
                              },
                              "10{'data': 'test', 'data2': 10.0, 'data3': 7}"),
                             (fn_multiple_args_to_str_annotated,
                              False, {
                                  "param1": 10, "param2": {
                                      "data": "test", "data2": 10.0, "data3": 7
                                  }
                              },
                              "10{'data': 'test', 'data2': 10.0, 'data3': 7}"),
                             (fn_multiple_args_annotated_to_str_annotated,
                              False, {
                                  "param1": 10, "param2": {
                                      "data": "test", "data2": 10.0, "data3": 7
                                  }
                              },
                              "10{'data': 'test', 'data2': 10.0, 'data3': 7}"),
                             (fn_multiple_args_to_str_stream,
                              True, {
                                  "param1": 10, "param2": {
                                      "data": "test", "data2": 10.0, "data3": 7
                                  }
                              },
                              "10{'data': 'test', 'data2': 10.0, 'data3': 7}"),
                             (fn_multiple_args_annotated_to_str_stream,
                              True, {
                                  "param1": 10, "param2": {
                                      "data": "test", "data2": 10.0, "data3": 7
                                  }
                              },
                              "10{'data': 'test', 'data2': 10.0, 'data3': 7}"),
                             (fn_multiple_args_to_str_annotated_stream,
                              True, {
                                  "param1": 10, "param2": {
                                      "data": "test", "data2": 10.0, "data3": 7
                                  }
                              },
                              "10{'data': 'test', 'data2': 10.0, 'data3': 7}"),
                             (fn_multiple_args_annotated_to_str_annotated_stream,
                              True, {
                                  "param1": 10, "param2": {
                                      "data": "test", "data2": 10.0, "data3": 7
                                  }
                              },
                              "10{'data': 'test', 'data2': 10.0, 'data3': 7}"),
                             (fn_union_to_str, False, 10, "10"),
                             (fn_int_to_union, False, 10, "10"),
                             (fn_int_to_union, False, 2, 2),
                             (fn_union_to_union, False, 10, "10"),
                             (fn_union_to_union, False, 2, 2),
                             (fn_union_to_str_stream, True, 10, "10"),
                             (fn_int_to_union_stream, True, 10, "10"),
                             (fn_int_to_union_stream, True, 2, 2),
                             (fn_union_to_union_stream, True, 10, "10"),
                             (fn_union_to_union_stream, True, 2, 2),
                             (fn_union_annotated_to_str, False, 10, "10"),
                             (fn_int_to_union_annotated, False, 10, "10"),
                             (fn_int_to_union_annotated, False, 2, 2),
                             (fn_union_annotated_to_str_stream, True, 10, "10"),
                             (fn_int_to_union_annotated_stream, True, 10, "10"),
                             (fn_int_to_union_annotated_stream, True, 2, 2),
                         ])
async def test_create_and_from_fn_call(function: Callable, is_streaming: bool, input_val, output_val):

    info_from_fn = FunctionInfo.from_fn(function)
    info_create = FunctionInfo.create(single_fn=function if not is_streaming else None,
                                      stream_fn=function if is_streaming else None)

    for info in [info_from_fn, info_create]:

        final_input_val = input_val

        if (inspect.isclass(info.input_type) and issubclass(info.input_type, BaseModel)):
            final_input_val = info.input_type(**final_input_val)

        if is_streaming:

            assert info.stream_fn is not None

            values = []

            async for value in info.stream_fn(final_input_val):
                values.append(value)

            assert values == [output_val]

        else:

            assert info.single_fn is not None

            assert await info.single_fn(final_input_val) == output_val


async def test_create_and_from_fn_description():

    info_from_fn = FunctionInfo.from_fn(fn_int_to_str, description="Test Description")
    info_create = FunctionInfo.create(single_fn=fn_int_to_str, description="Test Description")

    assert info_from_fn.description == "Test Description"
    assert info_create.description == "Test Description"


async def test_create_and_from_fn_input_schema():

    class TestSchema(BaseModel):
        param: str = Field(description="Param Description")

    info_from_fn = FunctionInfo.from_fn(fn_int_to_str, input_schema=TestSchema)
    info_create = FunctionInfo.create(single_fn=fn_int_to_str, input_schema=TestSchema)

    assert info_from_fn.input_schema == TestSchema
    assert info_create.input_schema == TestSchema


async def test_create_and_from_fn_converters():

    def convert_fn1(param: int) -> str:
        return str(param)

    def convert_fn2(param: str) -> int:
        return int(param)

    converters = [convert_fn1, convert_fn2]

    info_from_fn = FunctionInfo.from_fn(fn_int_to_str, converters=converters)
    info_create = FunctionInfo.create(single_fn=fn_int_to_str, converters=converters)

    assert info_from_fn.converters == converters
    assert info_create.converters == converters


async def test_create_output_schema():

    class TestSchema(BaseModel):
        value: str = Field(description="Param Description")

    info = FunctionInfo.create(single_fn=fn_int_to_str,
                               single_output_schema=TestSchema,
                               stream_fn=fn_int_to_str_stream,
                               stream_output_schema=TestSchema)

    assert info.single_output_schema == TestSchema
    assert info.stream_output_schema == TestSchema


async def test_create_single_to_stream_conversion():

    async def convert_to_stream(param: str) -> AsyncGenerator[int]:
        yield int(param)

    info = FunctionInfo.create(single_fn=fn_int_to_str, single_to_stream_fn=convert_to_stream)

    assert info.stream_fn is not None
    assert info.stream_output_type == int
    assert info.stream_output_schema is not None

    async for value in info.stream_fn(10):
        assert value == 10

    # ===== Negative tests =====

    # Test no single but single to stream function provided
    with pytest.raises(ValueError):
        FunctionInfo.create(stream_fn=fn_int_to_str_stream, single_to_stream_fn=convert_to_stream)

    # Test multiple arguments in convert function
    async def multiple_args(param1: int, param2: int) -> AsyncGenerator[int]:
        yield int(param1) + int(param2)

    with pytest.raises(ValueError):
        FunctionInfo.create(single_fn=fn_int_to_str, single_to_stream_fn=multiple_args)

    # Test mismatch between single and stream input types
    async def mismatch_type(param: dict) -> AsyncGenerator[int]:
        yield int(param["param"])

    with pytest.raises(ValueError):
        FunctionInfo.create(single_fn=fn_int_to_str, single_to_stream_fn=mismatch_type)

    # Missing output annotation
    with pytest.raises(ValueError):
        FunctionInfo.create(single_fn=fn_int_to_str, single_to_stream_fn=lambda x: x)

    # Not a streaming function
    async def not_streaming(param: str) -> int:
        return int(param)

    with pytest.raises(ValueError):
        FunctionInfo.create(single_fn=fn_int_to_str, single_to_stream_fn=not_streaming)


async def test_create_stream_to_single_conversion():

    async def convert_to_single(param: AsyncGenerator[str]) -> int:

        return int("".join([x async for x in param]))

    info = FunctionInfo.create(stream_fn=fn_int_to_str_stream, stream_to_single_fn=convert_to_single)

    assert info.single_fn is not None
    assert info.single_output_type == int
    assert info.single_output_schema is not None

    assert await info.single_fn(10) == 10

    # ===== Negative tests =====

    # Test no stream but stream to single function provided
    with pytest.raises(ValueError):
        FunctionInfo.create(single_fn=fn_int_to_str, stream_to_single_fn=convert_to_single)

    # Test multiple arguments in convert function
    async def multiple_args(param1: int, param2: int) -> int:
        return param1 + param2

    with pytest.raises(ValueError):
        FunctionInfo.create(stream_fn=fn_int_to_str_stream, stream_to_single_fn=multiple_args)

    # Test mismatch between single and stream input types
    async def mismatch_type(param: dict) -> int:
        return int(param["param"])

    with pytest.raises(ValueError):
        FunctionInfo.create(stream_fn=fn_int_to_str_stream, stream_to_single_fn=mismatch_type)

    # Missing output annotation
    with pytest.raises(ValueError):
        FunctionInfo.create(stream_fn=fn_int_to_str_stream, stream_to_single_fn=lambda x: x)

    # Not a streaming function
    def not_single(param: str) -> int:
        return int(param)

    with pytest.raises(ValueError):
        FunctionInfo.create(stream_fn=fn_int_to_str_stream, stream_to_single_fn=not_single)
