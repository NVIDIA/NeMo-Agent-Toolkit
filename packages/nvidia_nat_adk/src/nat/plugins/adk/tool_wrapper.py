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
"""Tool Wrapper file"""
import logging
import types
from collections.abc import AsyncIterator
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import Any
from typing import get_args
from typing import get_origin

from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.cli.register_workflow import register_tool_wrapper

logger = logging.getLogger(__name__)


def resolve_type(t: Any) -> Any:
    """Return the non-None member of a Union/PEP 604 union;
    otherwise return the type unchanged.

    Args:
        t (Any): The type to resolve.

    Returns:
        Any: The resolved type.
    """
    origin = get_origin(t)
    if origin is types.UnionType:
        for arg in get_args(t):
            if arg is not type(None):
                return arg
        return t
    return t


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.ADK)
def google_adk_tool_wrapper(
    name: str,
    fn: Function,
    _builder: Builder  # pylint: disable=W0613
) -> Any:  # Changed from Callable[..., Any] to Any to allow FunctionTool return
    """Wrap a NAT `Function` as a Google ADK `FunctionTool`.

    Args:
        name (str): The name of the tool.
        fn (Function): The NAT `Function` to wrap.
        _builder (Builder): The NAT `Builder` (not used).
    Returns:
        A Google ADK `FunctionTool` wrapping the NAT `Function`.
    """
    import inspect

    async def callable_ainvoke(*args: Any, **kwargs: Any) -> Any:
        """Async function to invoke the NAT function.

        Args:
            *args: Positional arguments to pass to the NAT function.
            **kwargs: Keyword arguments to pass to the NAT function.
        Returns:
            Any: The result of invoking the NAT function.
        """
        return await fn.acall_invoke(*args, **kwargs)

    async def callable_astream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Async generator to stream results from the NAT function.

        Args:
            *args: Positional arguments to pass to the NAT function.
            **kwargs: Keyword arguments to pass to the NAT function.
        Yields:
            Any: Streamed items from the NAT function.
        """
        async for item in fn.acall_stream(*args, **kwargs):
            yield item

    def nat_function(
        func: Callable[..., Any] | None = None,
        *,
        name: str = name,
        description: str | None = fn.description,
        input_schema: Any = fn.input_schema,
    ) -> Callable[..., Any]:
        """
        Decorator to wrap a function as a NAT function.

        Args:
            func (Callable): The function to wrap.
            name (str): The name of the function.
            description (str): The description of the function.
            input_schema (BaseModel): The Pydantic model defining the input schema.

        Returns:
            Callable[..., Any]: The wrapped function.
        """
        if func is None:
            raise ValueError("'func' must be provided.")

        # If input_schema is a dataclass, convert it to a Pydantic model
        if input_schema is not None and is_dataclass(input_schema):
            input_schema = BaseModel.model_validate(input_schema)

        def decorator(func_to_wrap: Callable[..., Any]) -> Callable[..., Any]:
            """
            Decorator to set metadata on the function.
            """
            # Set the function's metadata
            if name is not None:
                func_to_wrap.__name__ = name
            if description is not None:
                func_to_wrap.__doc__ = description

            # Set signature only if input_schema is provided
            params: list[inspect.Parameter] = []
            if input_schema is not None:
                annotations = getattr(input_schema, "__annotations__", {}) or {}
                for param_name, param_annotation in annotations.items():
                    params.append(
                        inspect.Parameter(
                            param_name,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=resolve_type(param_annotation),
                        ))
            setattr(func_to_wrap, "__signature__", inspect.Signature(parameters=params))

            return func_to_wrap

        # If func is None, return the decorator itself to be applied later
        if func is None:
            return decorator
        # Otherwise, apply the decorator to the provided function
        return decorator(func)

    from google.adk.tools.function_tool import FunctionTool

    if fn.has_streaming_output and not fn.has_single_output:
        logger.debug("Creating streaming FunctionTool for: %s", name)
        callable_tool = nat_function(func=callable_astream)
    else:
        logger.debug("Creating non-streaming FunctionTool for: %s", name)
        callable_tool = nat_function(func=callable_ainvoke)
    return FunctionTool(callable_tool)
