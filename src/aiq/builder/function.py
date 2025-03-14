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

import logging
import typing
from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable

from pydantic import BaseModel

from aiq.builder.context import AIQContext
from aiq.builder.function_base import FunctionBase
from aiq.builder.function_base import InputT
from aiq.builder.function_base import SingleOutputT
from aiq.builder.function_base import StreamingOutputT
from aiq.builder.function_info import FunctionInfo
from aiq.data_models.function import FunctionBaseConfig

_InvokeFnT = Callable[[InputT], Awaitable[SingleOutputT]]
_StreamFnT = Callable[[InputT], AsyncGenerator[StreamingOutputT]]

_T = typing.TypeVar("_T")

logger = logging.getLogger(__name__)


class Function(FunctionBase[InputT, StreamingOutputT, SingleOutputT], ABC):

    def __init__(self,
                 *,
                 config: FunctionBaseConfig,
                 description: str | None,
                 input_schema: type[BaseModel] | None = None,
                 streaming_output_schema: type[BaseModel] | type[None] | None = None,
                 single_output_schema: type[BaseModel] | type[None] | None = None,
                 converters: list[Callable[[typing.Any], typing.Any]] | None = None):

        super().__init__(input_schema=input_schema,
                         streaming_output_schema=streaming_output_schema,
                         single_output_schema=single_output_schema,
                         converters=converters)

        self.config = config
        self.description = description
        self._context = AIQContext.get()

    def convert(self, value: typing.Any, to_type: type[_T]) -> _T:
        """
        Converts the given value to the specified type using the function's converter.

        Parameters
        ----------
        value : typing.Any
            The value to convert.
        to_type : type
            The type to convert the value to.

        Returns
        -------
        _T
            The converted value.
        """

        return self._converter.convert(value, to_type=to_type)

    @abstractmethod
    async def _ainvoke(self, value: InputT) -> SingleOutputT:
        pass

    @typing.overload
    async def ainvoke(self, value: InputT | typing.Any) -> SingleOutputT:
        ...

    @typing.overload
    async def ainvoke(self, value: InputT | typing.Any, to_type: type[_T]) -> _T:
        ...

    @typing.final
    async def ainvoke(self, value: InputT | typing.Any, to_type: type | None = None):
        """
        Runs the function with the given input and returns a single output from the function. This is the
        main entry point for running a function.

        Parameters
        ----------
        value : InputT | typing.Any
            The input to the function.
        to_type : type | None, optional
            The type to convert the output to using the function's converter. When not specified, the
            output will match `single_output_type`.

        Returns
        -------
        typing.Any
            The output of the function optionally converted to the specified type.
        """

        with self._context.push_active_function(self.config.type,
                                                input_data=value) as manager:  # Set the current invocation context
            try:
                converted_input: InputT = self._convert_input(value)  # type: ignore

                result = await self._ainvoke(converted_input)

                if to_type is not None and not isinstance(result, to_type):
                    result = self._converter.convert(result, to_type=to_type)

                manager.set_output(result)

                return result
            except Exception as e:
                logger.error("Error with ainvoke in function with input: %s.", value, exc_info=True)
                raise e

    @typing.final
    async def acall_invoke(self, *args, **kwargs):
        """
        A wrapper around `ainvoke` that allows for calling the function with arbitrary arguments and keyword arguments.
        This is useful in scenarios where the function might be called by an LLM or other system which gives varying
        inputs to the function. The function will attempt to convert the args and kwargs to the input schema of the
        function.

        Returns
        -------
        SingleOutputT
            The output of the function.
        """

        if (len(args) == 1 and not kwargs):
            # If only one argument is passed, assume it is the input just like ainvoke
            return await self.ainvoke(value=args[0])

        if (not args and kwargs):
            # If only kwargs are passed, assume we are calling a function with named arguments in a dict
            # This will rely on the processing in ainvoke to convert from dict to the correct input type
            return await self.ainvoke(value=kwargs)

        # Possibly have both args and kwargs, final attempt is to use the input schema object constructor.
        try:
            input_obj = self.input_schema(*args, **kwargs)

            return await self.ainvoke(value=input_obj)
        except Exception as e:
            logger.error(
                "Error in acall_invoke() converting input to function schema. Both args and kwargs were "
                "supplied which could not be converted to the input schema. args: %s\nkwargs: %s\nschema: %s",
                args,
                kwargs,
                self.input_schema)
            raise e

    @abstractmethod
    async def _astream(self, value: InputT) -> AsyncGenerator[StreamingOutputT]:
        yield  # type: ignore

    @typing.overload
    async def astream(self, value: InputT | typing.Any) -> AsyncGenerator[SingleOutputT]:
        ...

    @typing.overload
    async def astream(self, value: InputT | typing.Any, to_type: type[_T]) -> AsyncGenerator[_T]:
        ...

    @typing.final
    async def astream(self, value: InputT | typing.Any, to_type: type | None = None):
        """
        Runs the function with the given input and returns a stream of outputs from the function. This is the main entry
        point for running a function with streaming output.

        Parameters
        ----------
        value : InputT | typing.Any
            The input to the function.
        to_type : type | None, optional
            The type to convert the output to using the function's converter. When not specified, the
            output will match `streaming_output_type`.

        Yields
        ------
        typing.Any
            The output of the function optionally converted to the specified type.
        """

        with self._context.push_active_function(self.config.type, input_data=value):
            try:
                converted_input: InputT = self._convert_input(value)  # type: ignore

                async for data in self._astream(converted_input):

                    if to_type is not None and not isinstance(data, to_type):
                        yield self._converter.convert(data, to_type=to_type)
                    else:
                        yield data
            except Exception as e:
                logger.error("Error with astream in function with input: %s.", value, exc_info=True)
                raise e

    @typing.final
    async def acall_stream(self, *args, **kwargs):
        """
        A wrapper around `astream` that allows for calling the function with arbitrary arguments and keyword arguments.
        This is useful in scenarios where the function might be called by an LLM or other system which gives varying
        inputs to the function. The function will attempt to convert the args and kwargs to the input schema of the
        function.

        Yields
        ------
        StreamingOutputT
            The output of the function.
        """

        if (len(args) == 1 and not kwargs):
            # If only one argument is passed, assume it is the input just like ainvoke
            async for x in self.astream(value=args[0]):
                yield x

        elif (not args and kwargs):
            # If only kwargs are passed, assume we are calling a function with named arguments in a dict
            # This will rely on the processing in ainvoke to convert from dict to the correct input type
            async for x in self.astream(value=kwargs):
                yield x

        # Possibly have both args and kwargs, final attempt is to use the input schema object constructor.
        else:
            try:
                input_obj = self.input_schema(*args, **kwargs)

                async for x in self.astream(value=input_obj):
                    yield x
            except Exception as e:
                logger.error(
                    "Error in acall_stream() converting input to function schema. Both args and kwargs were "
                    "supplied which could not be converted to the input schema. args: %s\nkwargs: %s\nschema: %s",
                    args,
                    kwargs,
                    self.input_schema)
                raise e


class LambdaFunction(Function[InputT, StreamingOutputT, SingleOutputT]):

    def __init__(self, *, config: FunctionBaseConfig, info: FunctionInfo):

        super().__init__(config=config,
                         description=info.description,
                         input_schema=info.input_schema,
                         streaming_output_schema=info.stream_output_schema,
                         single_output_schema=info.single_output_schema,
                         converters=info.converters)

        self._info = info

        self._ainvoke_fn: _InvokeFnT = info.single_fn
        self._astream_fn: _StreamFnT = info.stream_fn

    @property
    def has_streaming_output(self) -> bool:
        return self._astream_fn is not None

    @property
    def has_single_output(self) -> bool:
        return self._ainvoke_fn is not None

    async def _ainvoke(self, value: InputT) -> SingleOutputT:
        return await self._ainvoke_fn(value)

    async def _astream(self, value: InputT) -> AsyncGenerator[StreamingOutputT]:
        async for x in self._astream_fn(value):
            yield x

    @staticmethod
    def from_info(*, config: FunctionBaseConfig,
                  info: FunctionInfo) -> 'LambdaFunction[InputT, StreamingOutputT, SingleOutputT]':

        input_type: type = info.input_type
        streaming_output_type = info.stream_output_type
        single_output_type = info.single_output_type

        class FunctionImpl(LambdaFunction[input_type, streaming_output_type, single_output_type]):
            pass

        return FunctionImpl(config=config, info=info)
