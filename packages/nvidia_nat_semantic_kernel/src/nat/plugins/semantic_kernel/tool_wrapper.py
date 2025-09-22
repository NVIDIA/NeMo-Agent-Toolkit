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
import types
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import Any
from typing import Union
from typing import get_args
from typing import get_origin

from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.cli.register_workflow import register_tool_wrapper

logger = logging.getLogger(__name__)


def get_type_info(field_type):
    origin = get_origin(field_type)
    if origin is None:
        # It’s a simple type
        return getattr(field_type, "__name__", str(field_type))

    # Handle Union types specially
    if origin in (Union, types.UnionType):
        # Pick the first type that isn’t NoneType
        non_none = [arg for arg in get_args(field_type) if arg is not type(None)]
        if non_none:
            return getattr(non_none[0], "__name__", str(non_none[0]))

        return 'str'  # fallback if union is only str (unlikely)

    # For other generics, capture both the origin and its parameters
    return getattr(origin, "__name__", str(origin))


def resolve_type(t):
    origin = get_origin(t)
    if origin in (Union, types.UnionType):
        # Pick the first type that isn’t NoneType
        for arg in get_args(t):
            if arg is not None:
                return arg

        return t  # fallback if union is only NoneType (unlikely)
    return t


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)
def semantic_kernel_tool_wrapper(name: str, fn: Function, builder: Builder):

    async def callable_ainvoke(*args, **kwargs):
        return await fn.acall_invoke(*args, **kwargs)

    async def callable_astream(*args, **kwargs):
        async for item in fn.acall_stream(*args, **kwargs):
            yield item

    def nat_kernel_function(
        func: Callable[..., object] | None = None,
        nat_function: Function | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[..., Any]:
        """
        Modified version of Semantic Kernel's kernel_function decorator.

        Uses `nat` Function properties instead of doing type inference on the function's inner
        """

        def decorator(func: Callable[..., object]) -> Callable[..., object]:
            """The actual decorator function."""
            setattr(func, "__kernel_function__", True)
            setattr(func, "__kernel_function_description__", description or nat_function.description)
            setattr(func, "__kernel_function_name__", name or nat_function.config.type)

            # Always defer to single output schema, if present, for now
            # No need to check streaming output is present given one of the two is always present
            has_single = nat_function.has_single_output
            has_streaming = nat_function.has_streaming_output
            output_schema = nat_function.single_output_schema if has_single else nat_function.streaming_output_schema
            setattr(func, "__kernel_function_streaming__", not nat_function.has_single_output if has_single else True)

            if has_single and has_streaming:
                logger.warning("Function has both single and streaming output schemas. "
                               "Defaulting to single output schema.")

            input_annotations = []
            for arg_name, annotation in nat_function.input_schema.model_fields.items():
                type_obj = resolve_type(annotation.annotation)
                include_in_choices = True
                if isinstance(type_obj, type) and (issubclass(type_obj, BaseModel) or is_dataclass(type_obj)):
                    logger.warning(
                        "Nested non-native model detected in input schema for parameter: %s. "
                        "Setting include_in_function_choices to False.",
                        arg_name)
                    # Don't error out here
                    # Just instead avoid showing the tool to the model
                    include_in_choices = False
                input_annotations.append({
                    "is_required": annotation.is_required(),
                    "name": arg_name,
                    "type_": get_type_info(annotation.annotation),
                    "type_object": type_obj,
                    "include_in_function_choices": include_in_choices
                })

            setattr(func, "__kernel_function_parameters__", input_annotations)

            return_annotations = []
            for arg_name, annotation in output_schema.model_fields.items():
                type_obj = resolve_type(annotation.annotation)
                include_in_choices = True
                if isinstance(type_obj, type) and (issubclass(type_obj, BaseModel) or is_dataclass(type_obj)):
                    logger.warning(
                        "Nested non-native model detected in output schema for parameter: %s. "
                        "Setting include_in_function_choices to False.",
                        arg_name)
                    include_in_choices = False
                return_annotations.append({
                    "is_required": annotation.is_required(),
                    "name": arg_name,
                    "type_": get_type_info(annotation.annotation),
                    "type_object": type_obj,
                    "include_in_function_choices": include_in_choices
                })
            return_annotation = return_annotations[0]

            setattr(func, "__kernel_function_return_type__", return_annotation.get("type_", "None"))
            setattr(func, "__kernel_function_return_type_object__", return_annotation.get("type_object", None))
            setattr(func, "__kernel_function_return_description__", return_annotation.get("description", ""))
            setattr(func, "__kernel_function_return_required__", return_annotation.get("is_required", False))
            return func

        if func:
            return decorator(func)
        return decorator

    if fn.has_streaming_output and not fn.has_single_output:
        kernel_func = nat_kernel_function(func=callable_astream, nat_function=fn, name=name, description=fn.description)
    else:
        kernel_func = nat_kernel_function(func=callable_ainvoke, nat_function=fn, name=name, description=fn.description)

    return {name: kernel_func}
