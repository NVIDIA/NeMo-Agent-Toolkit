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
import re
import typing
from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence

from pydantic import BaseModel

from nat.builder.context import Context
from nat.builder.function_base import FunctionBase
from nat.builder.function_base import InputT
from nat.builder.function_base import SingleOutputT
from nat.builder.function_base import StreamingOutputT
from nat.builder.function_info import FunctionInfo
from nat.data_models.function import EmptyFunctionConfig
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.middleware.function_middleware import FunctionMiddlewareChain
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.middleware import Middleware

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
                 converters: list[Callable[[typing.Any], typing.Any]] | None = None,
                 instance_name: str | None = None):

        super().__init__(input_schema=input_schema,
                         streaming_output_schema=streaming_output_schema,
                         single_output_schema=single_output_schema,
                         converters=converters)

        self.config = config
        self.description = description
        self.instance_name = instance_name or config.type
        self._context = Context.get()
        self._configured_middleware: tuple[Middleware, ...] = tuple()
        self._middlewared_single: _InvokeFnT | None = None
        self._middlewared_stream: _StreamFnT | None = None

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

        Raises
        ------
        ValueError
            If the value cannot be converted to the specified type (when `to_type` is specified).
        """

        return self._converter.convert(value, to_type=to_type)

    def try_convert(self, value: typing.Any, to_type: type[_T]) -> _T | typing.Any:
        """
        Converts the given value to the specified type using graceful error handling.
        If conversion fails, returns the original value and continues processing.

        Parameters
        ----------
        value : typing.Any
            The value to convert.
        to_type : type
            The type to convert the value to.

        Returns
        -------
        _T | typing.Any
            The converted value, or original value if conversion fails.
        """
        return self._converter.try_convert(value, to_type=to_type)

    @property
    def middleware(self) -> tuple[Middleware, ...]:
        """Return the currently configured middleware chain."""

        return self._configured_middleware

    def configure_middleware(self, middleware: Sequence[Middleware] | None = None) -> None:
        """Attach an ordered list of middleware to this function instance."""

        middleware_tuple: tuple[Middleware, ...] = tuple(middleware or ())

        self._configured_middleware = middleware_tuple

        if not middleware_tuple:
            self._middlewared_single = None
            self._middlewared_stream = None
            return

        logger.info(f"Building middleware for function '{self.instance_name}' in order of: {middleware_tuple}")

        context = FunctionMiddlewareContext(name=self.instance_name,
                                            config=self.config,
                                            description=self.description,
                                            input_schema=self.input_schema,
                                            single_output_schema=self.single_output_schema,
                                            stream_output_schema=self.streaming_output_schema)

        chain = FunctionMiddlewareChain(middleware=middleware_tuple, context=context)

        self._middlewared_single = chain.build_single(self._ainvoke) if self.has_single_output else None
        self._middlewared_stream = chain.build_stream(self._astream) if self.has_streaming_output else None

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

        Raises
        ------
        ValueError
            If the output of the function cannot be converted to the specified type.
        """

        with self._context.push_active_function(self.instance_name,
                                                input_data=value) as manager:  # Set the current invocation context
            try:
                converted_input: InputT = self._convert_input(value)

                invoke_callable = self._middlewared_single or self._ainvoke

                result = await invoke_callable(converted_input)

                if to_type is not None and not isinstance(result, to_type):
                    result = self.convert(result, to_type)

                manager.set_output(result)

                return result
            except Exception as e:
                logger.error("Error with ainvoke in function with input: %s. Error: %s", value, e)
                raise

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
        except Exception:
            logger.error(
                "Error in acall_invoke() converting input to function schema. Both args and kwargs were "
                "supplied which could not be converted to the input schema. args: %s\nkwargs: %s\nschema: %s",
                args,
                kwargs,
                self.input_schema)
            raise

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

        Raises
        ------
        ValueError
            If the output of the function cannot be converted to the specified type (when `to_type` is specified).
        """

        with self._context.push_active_function(self.instance_name, input_data=value) as manager:
            try:
                converted_input: InputT = self._convert_input(value)

                # Collect streaming outputs to capture the final result
                final_output: list[typing.Any] = []

                stream_callable = self._middlewared_stream or self._astream

                async for data in stream_callable(converted_input):
                    if to_type is not None and not isinstance(data, to_type):
                        converted_data = self.convert(data, to_type=to_type)
                        final_output.append(converted_data)
                        yield converted_data
                    else:
                        final_output.append(data)
                        yield data

                # Set the final output for intermediate step tracking
                manager.set_output(final_output)

            except Exception as e:
                logger.error("Error with astream in function with input: %s. Error: %s", value, e)
                raise

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
            except Exception:
                logger.error(
                    "Error in acall_stream() converting input to function schema. Both args and kwargs were "
                    "supplied which could not be converted to the input schema. args: %s\nkwargs: %s\nschema: %s",
                    args,
                    kwargs,
                    self.input_schema)
                raise


class LambdaFunction(Function[InputT, StreamingOutputT, SingleOutputT]):

    def __init__(self, *, config: FunctionBaseConfig, info: FunctionInfo, instance_name: str | None = None):

        super().__init__(config=config,
                         description=info.description,
                         input_schema=info.input_schema,
                         streaming_output_schema=info.stream_output_schema,
                         single_output_schema=info.single_output_schema,
                         converters=info.converters,
                         instance_name=instance_name)

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
    def from_info(*,
                  config: FunctionBaseConfig,
                  info: FunctionInfo,
                  instance_name: str | None = None) -> 'LambdaFunction[InputT, StreamingOutputT, SingleOutputT]':

        input_type: type = info.input_type
        streaming_output_type = info.stream_output_type
        single_output_type = info.single_output_type

        class FunctionImpl(LambdaFunction[input_type, streaming_output_type, single_output_type]):
            pass

        return FunctionImpl(config=config, info=info, instance_name=instance_name)


class FunctionGroup:
    """
    A group of functions that can be used together, sharing the same configuration, context, and resources.
    """

    def __init__(self,
                 *,
                 config: FunctionGroupBaseConfig,
                 instance_name: str | None = None,
                 filter_fn: Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None = None,
                 middleware: Sequence[Middleware] | None = None):
        """
        Creates a new function group.

        Parameters
        ----------
        config : FunctionGroupBaseConfig
            The configuration for the function group.
        instance_name : str | None, optional
            The name of the function group. If not provided, the type of the function group will be used.
        filter_fn : Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None, optional
            A callback function to additionally filter the functions in the function group dynamically when
            the functions are accessed via any accessor method.
        middleware : Sequence[Middleware] | None, optional
            The middleware instances to apply to all functions in this group.
        """
        self._config = config
        self._instance_name = instance_name or config.type
        self._functions: dict[str, Function] = dict()
        self._filter_fn = filter_fn
        self._per_function_filter_fn: dict[str, Callable[[str], Awaitable[bool]]] = dict()
        self._middleware: tuple[Middleware, ...] = tuple(middleware or ())

    def add_function(self,
                     name: str,
                     fn: Callable,
                     *,
                     input_schema: type[BaseModel] | None = None,
                     description: str | None = None,
                     converters: list[Callable] | None = None,
                     filter_fn: Callable[[str], Awaitable[bool]] | None = None):
        """
        Adds a function to the function group.

        Parameters
        ----------
        name : str
            The name of the function.
        fn : Callable
            The function to add to the function group.
        input_schema : type[BaseModel] | None, optional
            The input schema for the function.
        description : str | None, optional
            The description of the function.
        converters : list[Callable] | None, optional
            The converters to use for the function.
        filter_fn : Callable[[str], Awaitable[bool]] | None, optional
            A callback to determine if the function should be included in the function group. The
            callback will be called with the function name. The callback is invoked dynamically when
            the functions are accessed via any accessor method such as `get_accessible_functions`,
            `get_included_functions`, `get_excluded_functions`, `get_all_functions`.

        Raises
        ------
        ValueError
            When the function name is empty or blank.
            When the function name contains invalid characters.
            When the function already exists in the function group.
        """
        if not name.strip():
            raise ValueError("Function name cannot be empty or blank")
        if not re.match(r"^[a-zA-Z0-9_.-]+$", name):
            raise ValueError(
                f"Function name can only contain letters, numbers, underscores, periods, and hyphens: {name}")
        if name in self._functions:
            raise ValueError(f"Function {name} already exists in function group {self._instance_name}")

        info = FunctionInfo.from_fn(fn, input_schema=input_schema, description=description, converters=converters)
        full_name = self._get_fn_name(name)
        lambda_fn = LambdaFunction.from_info(config=EmptyFunctionConfig(), info=info, instance_name=full_name)
        # Configure middleware from the function group if any
        if self._middleware:
            lambda_fn.configure_middleware(self._middleware)
        self._functions[name] = lambda_fn
        if filter_fn:
            self._per_function_filter_fn[name] = filter_fn

    def get_config(self) -> FunctionGroupBaseConfig:
        """
        Returns the configuration for the function group.

        Returns
        -------
        FunctionGroupBaseConfig
            The configuration for the function group.
        """
        return self._config

    def _get_fn_name(self, name: str) -> str:
        return f"{self._instance_name}.{name}"

    async def _fn_should_be_included(self, name: str) -> bool:
        if name not in self._per_function_filter_fn:
            return True
        return await self._per_function_filter_fn[name](name)

    async def _get_all_but_excluded_functions(
        self,
        filter_fn: Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None = None,
    ) -> dict[str, Function]:
        """
        Returns a dictionary of all functions in the function group except the excluded functions.
        """
        missing = set(self._config.exclude) - set(self._functions.keys())
        if missing:
            raise ValueError(f"Unknown excluded functions: {sorted(missing)}")

        if filter_fn is None:
            if self._filter_fn is None:

                async def identity_filter(x: Sequence[str]) -> Sequence[str]:
                    return x

                filter_fn = identity_filter
            else:
                filter_fn = self._filter_fn

        excluded = set(self._config.exclude)
        included = set(await filter_fn(list(self._functions.keys())))

        result = {}
        for name in self._functions:
            if name in excluded:
                continue
            if not await self._fn_should_be_included(name):
                continue
            if name not in included:
                continue
            result[self._get_fn_name(name)] = self._functions[name]

        return result

    async def get_accessible_functions(
        self,
        filter_fn: Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None = None,
    ) -> dict[str, Function]:
        """
        Returns a dictionary of all accessible functions in the function group.

        First, the functions are filtered by the function group's configuration.
        If the function group is configured to:
        - include some functions, this will return only the included functions.
        - not include or exclude any function, this will return all functions in the group.
        - exclude some functions, this will return all functions in the group except the excluded functions.

        Then, the functions are filtered by filter function and per-function filter functions.

        Parameters
        ----------
        filter_fn : Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None, optional
            A callback function to additionally filter the functions in the function group dynamically. If not provided
            then fall back to the function group's filter function. If no filter function is set for the function group
            all functions will be returned.

        Returns
        -------
        dict[str, Function]
            A dictionary of all accessible functions in the function group.

        Raises
        ------
        ValueError
            When the function group is configured to include functions that are not found in the group.
        """
        if self._config.include:
            return await self.get_included_functions(filter_fn=filter_fn)
        if self._config.exclude:
            return await self._get_all_but_excluded_functions(filter_fn=filter_fn)
        return await self.get_all_functions(filter_fn=filter_fn)

    async def get_excluded_functions(
        self,
        filter_fn: Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None = None,
    ) -> dict[str, Function]:
        """
        Returns a dictionary of all functions in the function group which are configured to be excluded or filtered
        out by a filter function or per-function filter function.

        Parameters
        ----------
        filter_fn : Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None, optional
            A callback function to additionally filter the functions in the function group dynamically. If not provided
            then fall back to the function group's filter function. If no filter function is set for the function group
            then no functions will be added to the returned dictionary.

        Returns
        -------
        dict[str, Function]
            A dictionary of all excluded functions in the function group.

        Raises
        ------
        ValueError
            When the function group is configured to exclude functions that are not found in the group.
        """
        missing = set(self._config.exclude) - set(self._functions.keys())
        if missing:
            raise ValueError(f"Unknown excluded functions: {sorted(missing)}")

        if filter_fn is None:
            if self._filter_fn is None:

                async def identity_filter(x: Sequence[str]) -> Sequence[str]:
                    return x

                filter_fn = identity_filter
            else:
                filter_fn = self._filter_fn

        excluded = set(self._config.exclude)
        included = set(await filter_fn(list(self._functions.keys())))

        result = {}
        for name in self._functions:
            is_excluded = False
            if name in excluded:
                is_excluded = True
            elif not await self._fn_should_be_included(name):
                is_excluded = True
            elif name not in included:
                is_excluded = True

            if is_excluded:
                result[self._get_fn_name(name)] = self._functions[name]

        return result

    async def get_included_functions(
        self,
        filter_fn: Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None = None,
    ) -> dict[str, Function]:
        """
        Returns a dictionary of all functions in the function group which are:
        - configured to be included and added to the global function registry
        - not configured to be excluded.
        - not filtered out by a filter function.

        Parameters
        ----------
        filter_fn : Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None, optional
            A callback function to additionally filter the functions in the function group dynamically. If not provided
            then fall back to the function group's filter function. If no filter function is set for the function group
            all functions will be returned.

        Returns
        -------
        dict[str, Function]
            A dictionary of all included functions in the function group.

        Raises
        ------
        ValueError
            When the function group is configured to include functions that are not found in the group.
        """
        missing = set(self._config.include) - set(self._functions.keys())
        if missing:
            raise ValueError(f"Unknown included functions: {sorted(missing)}")

        if filter_fn is None:
            if self._filter_fn is None:

                async def identity_filter(x: Sequence[str]) -> Sequence[str]:
                    return x

                filter_fn = identity_filter
            else:
                filter_fn = self._filter_fn

        included = set(await filter_fn(list(self._config.include)))
        result = {}
        for name in included:
            if await self._fn_should_be_included(name):
                result[self._get_fn_name(name)] = self._functions[name]
        return result

    async def get_all_functions(
        self,
        filter_fn: Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None = None,
    ) -> dict[str, Function]:
        """
        Returns a dictionary of all functions in the function group, regardless if they are included or excluded.

        If a filter function has been set, the returned functions will additionally be filtered by the callback.

        Parameters
        ----------
        filter_fn : Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None, optional
            A callback function to additionally filter the functions in the function group dynamically. If not provided
            then fall back to the function group's filter function. If no filter function is set for the function group
            all functions will be returned.

        Returns
        -------
        dict[str, Function]
            A dictionary of all functions in the function group.
        """
        if filter_fn is None:
            if self._filter_fn is None:

                async def identity_filter(x: Sequence[str]) -> Sequence[str]:
                    return x

                filter_fn = identity_filter
            else:
                filter_fn = self._filter_fn

        included = set(await filter_fn(list(self._functions.keys())))
        result = {}
        for name in included:
            if await self._fn_should_be_included(name):
                result[self._get_fn_name(name)] = self._functions[name]
        return result

    def set_filter_fn(self, filter_fn: Callable[[Sequence[str]], Awaitable[Sequence[str]]]):
        """
        Sets the filter function for the function group.

        Parameters
        ----------
        filter_fn : Callable[[Sequence[str]], Awaitable[Sequence[str]]]
            The filter function to set for the function group.
        """
        self._filter_fn = filter_fn

    def set_per_function_filter_fn(self, name: str, filter_fn: Callable[[str], Awaitable[bool]]):
        """
        Sets the a per-function filter function for the a function within the function group.

        Parameters
        ----------
        name : str
            The name of the function.
        filter_fn : Callable[[str], Awaitable[bool]]
            The per-function filter function to set for the function group.

        Raises
        ------
        ValueError
            When the function is not found in the function group.
        """
        if name not in self._functions:
            raise ValueError(f"Function {name} not found in function group {self._instance_name}")
        self._per_function_filter_fn[name] = filter_fn

    def set_instance_name(self, instance_name: str):
        """
        Sets the instance name for the function group.

        Parameters
        ----------
        instance_name : str
            The instance name to set for the function group.
        """
        self._instance_name = instance_name

    @property
    def instance_name(self) -> str:
        """
        Returns the instance name for the function group.
        """
        return self._instance_name

    @property
    def middleware(self) -> tuple[Middleware, ...]:
        """
        Returns the middleware configured for this function group.
        """
        return self._middleware

    def configure_middleware(self, middleware: Sequence[Middleware] | None = None) -> None:
        """
        Configure the middleware for this function group.
        These middleware will be applied to all functions added to the group.

        Parameters
        ----------
        middleware : Sequence[Middleware] | None
            The middleware to configure for the function group.
        """
        self._middleware = tuple(middleware or ())
        # Update existing functions with the new middleware
        for func in self._functions.values():
            func.configure_middleware(self._middleware)
