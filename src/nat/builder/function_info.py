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

import dataclasses
import inspect
import logging
import typing
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Coroutine
from types import NoneType

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import create_model
from pydantic_core import PydanticUndefined

from nat.data_models.streaming import Streaming
from nat.utils.type_utils import DecomposedType

logger = logging.getLogger(__name__)

P = typing.ParamSpec("P")
SingleCallableT = Callable[P, Coroutine[None, None, typing.Any]]
StreamCallableT = Callable[P, AsyncGenerator[typing.Any]]


def _get_annotated_type(annotated_type: type) -> type:
    origin = typing.get_origin(annotated_type)
    args = typing.get_args(annotated_type)

    # If its annotated, the first arg is the type
    if (origin == typing.Annotated):
        return args[0]

    return annotated_type


def _validate_single_fn(single_fn: SingleCallableT | None) -> tuple[type, type]:

    if single_fn is None:
        return NoneType, NoneType

    sig = inspect.signature(single_fn)

    if len(sig.parameters) != 1:
        raise ValueError("single_fn must have exactly one parameter")

    if (sig.parameters[list(sig.parameters.keys())[0]].annotation == sig.empty):
        raise ValueError("single_fn must have an input annotation")

    if sig.return_annotation == sig.empty:
        raise ValueError("single_fn must have a return annotation")

    if not inspect.iscoroutinefunction(single_fn):
        raise ValueError("single_fn must be a coroutine")

    type_hints = typing.get_type_hints(single_fn)

    output_type = type_hints.pop("return")

    assert len(type_hints) == 1

    input_type = next(iter(type_hints.values()))

    return input_type, output_type


def _validate_stream_fn(stream_fn: StreamCallableT | None) -> tuple[type, type]:

    if stream_fn is None:
        return NoneType, NoneType

    sig = inspect.signature(stream_fn)

    if len(sig.parameters) != 1:
        raise ValueError("stream_fn must have exactly one parameter")

    if sig.return_annotation == sig.empty:
        raise ValueError("stream_fn must have a return annotation")

    if not inspect.isasyncgenfunction(stream_fn):
        raise ValueError("stream_fn must be an async generator")

    type_hints = typing.get_type_hints(stream_fn)

    # AsyncGenerator[OutputType, None]
    async_gen_type = DecomposedType(type_hints.pop("return"))

    if (not async_gen_type.is_async_generator):
        raise ValueError("stream_fn return value must be annotated as an async generator")

    # If the output type is annotated, get the actual type
    output_type = async_gen_type.get_async_generator_type().type

    assert len(type_hints) == 1

    input_type = next(iter(type_hints.values()))

    return input_type, output_type


@dataclasses.dataclass
class FunctionDescriptor:

    func: Callable

    arg_count: int

    is_coroutine: bool
    """
    Whether the function is a coroutine or not.
    """

    is_async_gen: bool
    """
    Whether the function is an async generator or not.
    """

    input_type: type | type[None] | None
    """
    The direct annotated input type to the function. If the function has multiple arguments, this will be a tuple of
    the annotated types. If the function has no annotations, this will be None. If the function has no arguments, this
    will be NoneType.
    """
    input_schema: type[BaseModel] | type[None] | None
    """
    The Pydantic schema for the input to the function. This will always be a Pydantic model with the arguments as fields
    ( even if the function only has one BaseModel input argument). If the function has no input, this will be NoneType.
    If the function has no annotations, this will be None.
    """
    input_type_is_base_model: bool
    """
    True if the input type is a subclass of BaseModel, False otherwise
    """

    output_type: type | type[None] | None
    """
    The direct annotated output type to the function. If the function has no annotations, this will be None. If the
    function has no return type, this will be NoneType.
    """
    output_schema: type[BaseModel] | type[None] | None
    """
    The Pydantic schema for the output of the function. If the return type is already a BaseModel, the schema will be
    the same as the `output_type`. If the function has no return type, this will be NoneType. If the function has no
    annotations, this will be None.
    """
    output_type_is_base_model: bool
    """
    True if the output type is a subclass of BaseModel, False otherwise
    """

    is_input_typed: bool
    """
    True if all of the functions input arguments have type annotations, False otherwise
    """

    is_output_typed: bool
    """
    True if the function has a return type annotation, False otherwise
    """

    converters: list[Callable]
    """
    A list of converters for converting to/from the function's input/output types. Converters are created when
    determining the output schema of a function.
    """

    def get_base_model_function_input(self) -> type[BaseModel] | type[None] | None:
        """
        Returns a BaseModel type which can be used as the function input. If the InputType is a BaseModel, it will be
        returned, otherwise the InputSchema will be returned. If the function has no input, NoneType will be returned.
        """

        if self.input_type_is_base_model:
            return self.input_type

        return self.input_schema

    def get_base_model_function_output(self,
                                       converters: list[Callable] | None = None) -> type[BaseModel] | type[None] | None:
        """
        Returns a BaseModel type which can be used as the function output. If the OutputType is a BaseModel, it will be
        returned, otherwise the OutputSchema will be returned. If the function has no output, NoneType will be returned.
        """

        if (converters is not None):
            converters.extend(self.converters)

        if self.output_type_is_base_model:
            return self.output_type

        return self.output_schema

    @staticmethod
    def from_function(func: Callable) -> 'FunctionDescriptor':

        is_coroutine = inspect.iscoroutinefunction(func)
        is_async_gen = inspect.isasyncgenfunction(func)

        converters = []

        sig = inspect.signature(func)

        arg_count = len(sig.parameters)

        if (arg_count == 0):
            input_type = NoneType
            is_input_typed = False
            input_schema = NoneType
        elif (arg_count == 1):
            first_annotation = sig.parameters[list(sig.parameters.keys())[0]].annotation

            is_input_typed = first_annotation != sig.empty

            input_type = first_annotation if is_input_typed else None
        else:
            annotations = [param.annotation for param in sig.parameters.values()]

            is_input_typed = all([a != sig.empty for a in annotations])  # pylint: disable=use-a-generator

            input_type = tuple[*annotations] if is_input_typed else None  # noqa: syntax-error

        # Get the base type here removing all annotations and async generators
        output_annotation_decomp = DecomposedType(sig.return_annotation).get_base_type()

        is_output_typed = not output_annotation_decomp.is_empty

        output_type = output_annotation_decomp.type if is_output_typed else None

        output_schema = output_annotation_decomp.get_pydantic_schema(converters) if is_output_typed else None

        if (input_type is not None):

            args_schema: dict[str, tuple[type, typing.Any]] = {}

            for param in sig.parameters.values():

                default_val = PydanticUndefined

                if (param.default != sig.empty):
                    default_val = param.default

                args_schema[param.name] = (param.annotation, Field(default=default_val))

            input_schema = create_model("InputArgsSchema",
                                        __config__=ConfigDict(arbitrary_types_allowed=True),
                                        **args_schema)
        else:
            input_schema = None

        input_type_is_base_model = False
        output_type_is_base_model = False

        if (input_type is not None):
            input_type_is_base_model = DecomposedType(input_type).is_subtype(BaseModel)

        if (output_type is not None):
            output_type_is_base_model = DecomposedType(output_type).is_subtype(BaseModel)

        return FunctionDescriptor(func=func,
                                  arg_count=arg_count,
                                  is_coroutine=is_coroutine,
                                  is_async_gen=is_async_gen,
                                  is_input_typed=is_input_typed,
                                  is_output_typed=is_output_typed,
                                  input_type=input_type,
                                  output_type=output_type,
                                  input_schema=input_schema,
                                  output_schema=output_schema,
                                  input_type_is_base_model=input_type_is_base_model,
                                  output_type_is_base_model=output_type_is_base_model,
                                  converters=converters)


class FunctionInfo:

    def __init__(self,
                 *,
                 single_fn: SingleCallableT | None = None,
                 stream_fn: StreamCallableT | None = None,
                 input_schema: type[BaseModel] | type[None],
                 single_output_schema: type[BaseModel] | type[None],
                 stream_output_schema: type[BaseModel] | type[None],
                 description: str | None = None,
                 converters: list[Callable] | None = None):
        self.single_fn = single_fn
        self.stream_fn = stream_fn
        self.input_schema = input_schema
        self.single_output_schema = single_output_schema
        self.stream_output_schema = stream_output_schema
        self.description = description
        self.converters = converters

        # At this point, we only are validating the passed in information. We are not converting anything. That will
        # be done in the `create()`` and `from_fn()` static methods.
        single_input_type, single_output_type = _validate_single_fn(single_fn)
        stream_input_type, stream_output_type = _validate_stream_fn(stream_fn)

        if ((NoneType not in (single_input_type, stream_input_type)) and (single_input_type != stream_input_type)):
            raise ValueError("single_fn and stream_fn must have the same input type")

        if (single_input_type is not NoneType):
            self.input_type = single_input_type
        elif (stream_input_type is not None):
            self.input_type = stream_input_type
        else:
            raise ValueError("At least one of single_fn or stream_fn must be provided")

        self.single_output_type: type = single_output_type
        self.stream_output_type: type = stream_output_type

        if (self.single_fn is None and self.stream_fn is None):
            raise ValueError("At least one of single_fn or stream_fn must be provided")

        # All of the schemas must be provided. NoneType indicates there is no type. None indicates not set
        if (self.input_schema is None):
            raise ValueError("input_schema must be provided")

        if (self.single_output_schema is None):
            raise ValueError("single_output_schema must be provided. Use NoneType if there is single output")

        if (self.stream_output_schema is None):
            raise ValueError("stream_output_schema must be provided. Use NoneType if there is stream output")

        if (self.single_fn and self.single_output_schema == NoneType):
            raise ValueError("single_output_schema must be provided if single_fn is provided")
        if (not self.single_fn and self.single_output_schema != NoneType):
            raise ValueError("single_output_schema must be NoneType if single_fn is not provided")

        if (self.stream_fn and self.stream_output_schema is NoneType):
            raise ValueError("stream_output_schema must be provided if stream_fn is provided")
        if (not self.stream_fn and self.stream_output_schema != NoneType):
            raise ValueError("stream_output_schema must be NoneType if stream_fn is not provided")

    @staticmethod
    def create(*,
               single_fn: SingleCallableT | None = None,
               stream_fn: StreamCallableT | None = None,
               input_schema: type[BaseModel] | type[None] | None = None,
               single_output_schema: type[BaseModel] | type[None] | None = None,
               stream_output_schema: type[BaseModel] | type[None] | None = None,
               single_to_stream_fn: Callable[[typing.Any], AsyncGenerator[typing.Any]]
               | None = None,
               stream_to_single_fn: Callable[[AsyncGenerator[typing.Any]], Awaitable[typing.Any]]
               | None = None,
               description: str | None = None,
               converters: list[Callable] | None = None) -> 'FunctionInfo':

        converters = converters or []

        final_single_fn: SingleCallableT | None = None
        final_stream_fn: StreamCallableT | None = None

        # Check the correct combination of functions
        if (single_fn is not None):
            final_single_fn = single_fn

            if (stream_to_single_fn is not None):
                raise ValueError("Cannot provide both single_fn and stream_to_single_fn")
        else:
            if (stream_to_single_fn is not None and stream_fn is None):
                raise ValueError("stream_fn must be provided if stream_to_single_fn is provided")

        if (stream_fn is not None):
            final_stream_fn = stream_fn

            if (single_to_stream_fn is not None):
                raise ValueError("Cannot provide both stream_fn and single_to_stream_fn")
        else:
            if (single_to_stream_fn is not None and single_fn is None):
                raise ValueError("single_fn must be provided if single_to_stream_fn is provided")

        if (single_fn is None and stream_fn is None):
            raise ValueError("At least one of single_fn or stream_fn must be provided")

        # Now we know that we have the correct combination of functions. See if we can make conversions
        if (single_to_stream_fn is not None):

            if (single_fn is None):
                raise ValueError("single_fn must be provided if single_to_stream_fn is provided")

            single_to_stream_fn_desc = FunctionDescriptor.from_function(single_to_stream_fn)

            if single_to_stream_fn_desc.arg_count != 1:
                raise ValueError("single_to_stream_fn must have exactly one argument")

            if not single_to_stream_fn_desc.is_output_typed:
                raise ValueError("single_to_stream_fn must have a return annotation")

            if not single_to_stream_fn_desc.is_async_gen:
                raise ValueError("single_to_stream_fn must be an async generator")

            single_fn_desc = FunctionDescriptor.from_function(single_fn)

            if (single_fn_desc.output_type != single_to_stream_fn_desc.input_type):
                raise ValueError("single_to_stream_fn must have the same input type as the output from single_fn")

            async def _converted_stream_fn(
                    message: single_fn_desc.input_type) -> AsyncGenerator[single_to_stream_fn_desc.output_type]:
                value = await single_fn(message)

                async for m in single_to_stream_fn(value):
                    yield m

            final_stream_fn = _converted_stream_fn

        if (stream_to_single_fn is not None):

            if (stream_fn is None):
                raise ValueError("stream_fn must be provided if stream_to_single_fn is provided")

            stream_to_single_fn_desc = FunctionDescriptor.from_function(stream_to_single_fn)

            if stream_to_single_fn_desc.arg_count != 1:
                raise ValueError("stream_to_single_fn must have exactly one parameter")

            if not stream_to_single_fn_desc.is_output_typed:
                raise ValueError("stream_to_single_fn must have a return annotation")

            if not stream_to_single_fn_desc.is_coroutine:
                raise ValueError("stream_to_single_fn must be a coroutine")

            stream_fn_desc = FunctionDescriptor.from_function(stream_fn)

            if (AsyncGenerator[stream_fn_desc.output_type] != stream_to_single_fn_desc.input_type):
                raise ValueError("stream_to_single_fn must take an async generator with "
                                 "the same input type as the output from stream_fn")

            async def _converted_single_fn(message: stream_fn_desc.input_type) -> stream_to_single_fn_desc.output_type:

                return await stream_to_single_fn(stream_fn(message))

            final_single_fn = _converted_single_fn

        # Check the input/output of the functions to make sure they are all BaseModels
        if (final_single_fn is not None):

            final_single_fn_desc = FunctionDescriptor.from_function(final_single_fn)

            if (final_single_fn_desc.arg_count > 1):
                if (input_schema is not None):
                    logger.warning("Using provided input_schema for multi-argument function")
                else:
                    input_schema = final_single_fn_desc.get_base_model_function_input()

                saved_final_single_fn = final_single_fn

                async def _convert_input_pydantic(value: input_schema) -> final_single_fn_desc.output_type:

                    # Unpack the pydantic model into the arguments
                    return await saved_final_single_fn(**value.model_dump())

                final_single_fn = _convert_input_pydantic

                # Reset the descriptor
                final_single_fn_desc = FunctionDescriptor.from_function(final_single_fn)

            input_schema = input_schema or final_single_fn_desc.get_base_model_function_input()

            single_output_schema = single_output_schema or final_single_fn_desc.get_base_model_function_output(
                converters)

            # Check if the final_stream_fn is None. We can use the final_single_fn to create a streaming version
            # automatically
            if (final_stream_fn is None):

                async def _stream_from_single_fn(
                        message: final_single_fn_desc.input_type) -> AsyncGenerator[final_single_fn_desc.output_type]:
                    value = await final_single_fn(message)

                    yield value

                final_stream_fn = _stream_from_single_fn

        else:
            single_output_schema = NoneType

        if (final_stream_fn is not None):

            final_stream_fn_desc = FunctionDescriptor.from_function(final_stream_fn)

            if (final_stream_fn_desc.arg_count > 1):
                if (input_schema is not None):
                    logger.warning("Using provided input_schema for multi-argument function")
                else:
                    input_schema = final_stream_fn_desc.get_base_model_function_input()

                saved_final_stream_fn = final_stream_fn

                async def _convert_input_pydantic_stream(
                        value: input_schema) -> AsyncGenerator[final_stream_fn_desc.output_type]:

                    # Unpack the pydantic model into the arguments
                    async for m in saved_final_stream_fn(**value.model_dump()):
                        yield m

                final_stream_fn = _convert_input_pydantic_stream

                # Reset the descriptor
                final_stream_fn_desc = FunctionDescriptor.from_function(final_stream_fn)

            input_schema = input_schema or final_stream_fn_desc.get_base_model_function_input()

            stream_output_schema = stream_output_schema or final_stream_fn_desc.get_base_model_function_output(
                converters)
        else:
            stream_output_schema = NoneType

        # Do the final check for the input schema from the final functions
        if (input_schema is None):

            if (final_single_fn):

                final_single_fn_desc = FunctionDescriptor.from_function(final_single_fn)

                if (final_single_fn_desc.input_type != NoneType):
                    input_schema = final_single_fn_desc.get_base_model_function_output(converters)

            elif (final_stream_fn):

                final_stream_fn_desc = FunctionDescriptor.from_function(final_stream_fn)

                if (final_stream_fn_desc.input_type != NoneType):
                    input_schema = final_stream_fn_desc.get_base_model_function_output(converters)

            else:
                # Cant be None
                input_schema = NoneType

        return FunctionInfo(single_fn=final_single_fn,
                            stream_fn=final_stream_fn,
                            input_schema=input_schema,
                            single_output_schema=single_output_schema,
                            stream_output_schema=stream_output_schema,
                            description=description,
                            converters=converters)

    @staticmethod
    def from_fn(fn: SingleCallableT | StreamCallableT,
                *,
                input_schema: type[BaseModel] | None = None,
                description: str | None = None,
                converters: list[Callable] | None = None) -> 'FunctionInfo':
        """
        Creates a FunctionInfo object from either a single or stream function. Automatically determines the type of
        function and creates the appropriate FunctionInfo object. Supports type annotations for conversion functions.

        Parameters
        ----------
        fn : SingleCallableT | StreamCallableT
            The function to create the FunctionInfo object from
        input_schema : type[BaseModel] | None, optional
            A schema object which defines the input to the function, by default None
        description : str | None, optional
            A description to set to the function, by default None
        converters : list[Callable] | None, optional
            A list of converters for converting to/from the function's input/output types, by default None

        Returns
        -------
        FunctionInfo
            The created FunctionInfo object which can be used to create a Generic NAT function.

        """

        stream_fn: StreamCallableT | None = None
        single_fn: SingleCallableT | None = None

        if (inspect.isasyncgenfunction(fn)):
            stream_fn = fn

            sig = inspect.signature(fn)

            output_origin = typing.get_origin(sig.return_annotation)
            output_args = typing.get_args(sig.return_annotation)

            if (output_origin == typing.Annotated):
                # typing.Annotated[AsyncGenerator[OutputType, None], ...]
                annotated_args = output_args[1:]

                stream_arg = None

                for arg in annotated_args:
                    if (isinstance(arg, Streaming)):
                        stream_arg = arg
                        break

                if (stream_arg):
                    single_input_type = sig.parameters[list(sig.parameters.keys())[0]].annotation
                    single_output_type = stream_arg.single_output_type

                    async def _stream_to_single_output(message: single_input_type) -> single_output_type:
                        values = []

                        async for m in stream_fn(message):
                            values.append(m)

                        return stream_arg.convert(values)

                    single_fn = _stream_to_single_output

        elif (inspect.iscoroutinefunction(fn)):
            single_fn = fn

        else:
            raise ValueError("Invalid workflow function. Must be an async generator or coroutine")

        return FunctionInfo.create(single_fn=single_fn,
                                   stream_fn=stream_fn,
                                   input_schema=input_schema,
                                   description=description,
                                   converters=converters or [])
