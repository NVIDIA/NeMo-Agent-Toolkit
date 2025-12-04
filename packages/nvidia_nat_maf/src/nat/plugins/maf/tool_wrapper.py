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
    """Maps Python types to common string types used in function calling schemas."""
    origin = get_origin(field_type)
    if origin is None:
        # It’s a simple type
        name = getattr(field_type, "__name__", str(field_type)).lower()
        if name in ('str', 'int', 'float', 'bool'):
            return name
        return 'object' # Default for complex types like dataclasses/Pydantic models

    # Handle Union types specially
    if origin in (Union, types.UnionType):
        # Pick the first type that isn’t NoneType
        non_none = [arg for arg in get_args(field_type) if arg is not type(None)]
        if non_none:
            return get_type_info(non_none[0])

        return 'string' # fallback

    # For other generics (e.g., List, Dict), use the origin's name or 'array'/'object'
    name = getattr(origin, "__name__", str(origin)).lower()
    if name in ('list', 'array'):
        return 'array'
    return 'object'


def resolve_type(t):
    """Resolves Unions to their non-None type or returns the type itself."""
    origin = get_origin(t)
    if origin in (Union, types.UnionType):
        # Pick the first type that isn’t NoneType
        for arg in get_args(t):
            if arg is not type(None):
                return arg

        return t  # fallback
    return t


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.MICROSOFT_AGENT_FRAMEWORK) # 1. Changed framework type
def microsoft_agent_framework_tool_wrapper(name: str, fn: Function, builder: Builder):
    """
    Wraps a nat Function into an OpenAI-compatible function schema
    for use with the Microsoft Agent Framework.
    """

    # MAF/OpenAI tools expect a callable and a schema (dict).
    # We define the callable here and the schema below.
    async def callable_ainvoke(*args, **kwargs):
        """Standard asynchronous tool invocation."""
        return await fn.acall_invoke(*args, **kwargs)

    # MAF tools typically don't directly stream; they are usually defined
    # as single-call functions in the schema. We'll prioritize the single
    # invoke path for simplicity and MAF convention.

    def generate_openai_function_schema(nat_function: Function, function_name: str) -> dict[str, Any]:
        """
        Generates an OpenAI-compatible function schema dictionary.
        This is the format MAF tools often expect.
        """
        
        # 2. Extract properties for the function schema
        input_schema = nat_function.input_schema.model_fields
        required_params = []
        properties = {}

        for arg_name, annotation in input_schema.items():
            type_obj = resolve_type(annotation.annotation)
            
            # MAF/OpenAI schema doesn't support complex nested types well; 
            # we check but still generate the schema for the root function.
            if isinstance(type_obj, type) and (issubclass(type_obj, BaseModel) or is_dataclass(type_obj)):
                # Complex type warning: MAF might struggle with these parameters
                logger.warning(
                    "Nested non-native model detected in input schema for parameter: %s. "
                    "This may not be fully supported by the MAF model.",
                    arg_name)

            param_type = get_type_info(annotation.annotation)
            
            # Map Python/Pydantic type info to OpenAI schema properties
            param_def = {
                "type": param_type,
                # Description often comes from the Pydantic field's description/docstring
                "description": annotation.description or f"Parameter {arg_name} of type {param_type}.",
            }
            properties[arg_name] = param_def

            if annotation.is_required():
                required_params.append(arg_name)

        schema = {
            "type": "function",
            "function": {
                "name": function_name,
                "description": nat_function.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params,
                }
            }
        }
        return schema


    # 3. Generate the tool schema and return the callable alongside it.
    tool_schema = generate_openai_function_schema(nat_function=fn, function_name=name)

    # The MAF/OpenAI format returns a list of tools, where each tool is a dict
    # containing the callable and the schema.
    # The return format should be a dict of {tool_name: (callable, tool_schema_dict)}
    # or just {tool_name: callable} depending on the exact MAF wrapper expectation.
    #