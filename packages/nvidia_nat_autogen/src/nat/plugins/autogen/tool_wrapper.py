# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tool wrapper for AutoGen integration with NAT."""

import logging
from collections.abc import AsyncIterator
from collections.abc import Callable
from dataclasses import is_dataclass

# PythonType not available in AutoGen 0.7.4, using Any instead
from typing import Any

from autogen_core.tools import FunctionTool
from pydantic import BaseModel
from pydantic.dataclasses import dataclass as pydantic_dataclass

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.cli.register_workflow import register_tool_wrapper
from nat.utils.type_utils import DecomposedType

logger = logging.getLogger(__name__)


def resolve_type(t: Any) -> Any:
    """Return the non-None member of a Union/PEP 604 union;
    otherwise return the type unchanged.

    Args:
        t (Any): The type to resolve.

    Returns:
        Any: The resolved type.
    """
    resolved = DecomposedType(t)
    if resolved.is_optional:
        return resolved.get_optional_type().type
    return resolved.type


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.AUTOGEN)
def autogen_tool_wrapper(
    name: str,
    fn: Function,
    _builder: Builder  # pylint: disable=W0613
) -> Any:  # Changed from Callable[..., Any] to Any to allow FunctionTool return
    """Wrap a NAT `Function` as an AutoGen `FunctionTool`.

    Args:
        name (str): The name of the tool.
        fn (Function): The NAT function to wrap.
        _builder (Builder): The NAT workflow builder to access registered components.

    Returns:
        Any: The AutoGen FunctionTool wrapping the NAT function.
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
            *args (Any): Positional arguments to pass to the NAT function.
            **kwargs (Any): Keyword arguments to pass to the NAT function.
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
            input_schema = pydantic_dataclass(input_schema)

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
            annotations: dict[str, Any] = {}

            if input_schema is not None:
                annotations = {}
                params = []
                model_fields = getattr(input_schema, "model_fields", {})
                for param_name, model_field in model_fields.items():
                    resolved_type = resolve_type(model_field.annotation)

                    # Warn about nested Pydantic models or dataclasses that may not serialize properly
                    # Note: If autogen is updated to support nested models, this warning can be removed - or
                    # if autogen adds a mechanism to remove the tool from the function choices we can add that later.
                    if isinstance(resolved_type, type) and (issubclass(resolved_type, BaseModel)
                                                            or is_dataclass(resolved_type)):
                        logger.warning(
                            "Nested model detected in input schema for parameter '%s' in tool '%s'. "
                            "AutoGen may not properly serialize complex nested types for function calling. "
                            "Consider flattening the schema or using primitive types.",
                            param_name,
                            name,
                        )

                    default = inspect.Parameter.empty if model_field.is_required() else model_field.default
                    params.append(
                        inspect.Parameter(param_name,
                                          inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                          annotation=resolved_type,
                                          default=default))
                    annotations[param_name] = resolved_type
                func_to_wrap.__signature__ = inspect.Signature(parameters=params)
                func_to_wrap.__annotations__ = annotations

            return func_to_wrap

        # Apply the decorator to the provided function
        return decorator(func)

    if fn.has_streaming_output and not fn.has_single_output:
        logger.debug("Creating streaming FunctionTool for: %s", name)
        callable_tool = nat_function(func=callable_astream)
    else:
        logger.debug("Creating non-streaming FunctionTool for: %s", name)
        callable_tool = nat_function(func=callable_ainvoke)
    return FunctionTool(
        func=callable_tool,
        name=name,
        description=fn.description or "No description provided.",
    )
