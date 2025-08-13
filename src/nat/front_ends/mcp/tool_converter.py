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

import json
import logging
from inspect import Parameter
from inspect import Signature

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from nat.builder.function import Function
from nat.builder.function_base import FunctionBase
from nat.builder.workflow import Workflow

logger = logging.getLogger(__name__)


def create_function_wrapper(
    function_name: str,
    function: FunctionBase,
    schema: type[BaseModel],
    is_workflow: bool = False,
):
    """Create a wrapper function that exposes the actual parameters of a NAT Function as an MCP tool.

    Args:
        function_name: The name of the function/tool
        function: The NAT Function object
        schema: The input schema of the function
        is_workflow: Whether the function is a Workflow

    Returns:
        A wrapper function suitable for registration with MCP
    """
    # Check if we're dealing with ChatRequest - special case
    is_chat_request = False

    # Check if the schema name is ChatRequest
    if schema.__name__ == "ChatRequest" or (hasattr(schema, "__qualname__") and "ChatRequest" in schema.__qualname__):
        is_chat_request = True
        logger.info("Function %s uses ChatRequest - creating simplified interface", function_name)

        # For ChatRequest, we'll create a simple wrapper with just a query parameter
        parameters = [Parameter(
            name="query",
            kind=Parameter.KEYWORD_ONLY,
            default=Parameter.empty,
            annotation=str,
        )]
    else:
        # Regular case - extract parameter information from the input schema
        # Extract parameter information from the input schema
        param_fields = schema.model_fields

        parameters = []
        for name, field in param_fields.items():
            # Get the field type and convert to appropriate Python type
            field_type = field.annotation

            # Add the parameter to our list
            parameters.append(
                Parameter(
                    name=name,
                    kind=Parameter.KEYWORD_ONLY,
                    default=Parameter.empty if field.is_required else None,
                    annotation=field_type,
                ))

    # Create the function signature WITHOUT the ctx parameter
    # We'll handle this in the wrapper function internally
    sig = Signature(parameters=parameters, return_annotation=str)

    # Define the actual wrapper function that accepts ctx but doesn't expose it
    def create_wrapper():

        async def wrapper_with_ctx(**kwargs):
            """Internal wrapper that will be called by MCP."""
            # MCP will add a ctx parameter, extract it
            ctx = kwargs.get("ctx")

            # Remove ctx if present
            if "ctx" in kwargs:
                del kwargs["ctx"]

            # Process the function call
            if ctx:
                ctx.info("Calling function %s with args: %s", function_name, json.dumps(kwargs, default=str))
                await ctx.report_progress(0, 100)

            try:
                # Special handling for ChatRequest
                if is_chat_request:
                    from nat.data_models.api_server import ChatRequest

                    # Create a chat request from the query string
                    query = kwargs.get("query", "")
                    chat_request = ChatRequest.from_string(query)

                    # Special handling for Workflow objects
                    if is_workflow:
                        # Workflows have a run method that is an async context manager
                        # that returns a Runner
                        async with function.run(chat_request) as runner:
                            # Get the result from the runner
                            result = await runner.result(to_type=str)
                    else:
                        # Regular functions use ainvoke
                        result = await function.ainvoke(chat_request, to_type=str)
                else:
                    # Regular handling
                    # Handle complex input schema - if we extracted fields from a nested schema,
                    # we need to reconstruct the input
                    if len(schema.model_fields) == 1 and len(parameters) > 1:
                        # Get the field name from the original schema
                        field_name = next(iter(schema.model_fields.keys()))
                        field_type = schema.model_fields[field_name].annotation

                        # If it's a pydantic model, we need to create an instance
                        if hasattr(field_type, "model_validate"):
                            # Create the nested object
                            nested_obj = field_type.model_validate(kwargs)
                            # Call with the nested object
                            kwargs = {field_name: nested_obj}

                    # Call the NAT function with the parameters - special handling for Workflow
                    if is_workflow:
                        # For workflow with regular input, we'll assume the first parameter is the input
                        input_value = list(kwargs.values())[0] if kwargs else ""

                        # Workflows have a run method that is an async context manager
                        # that returns a Runner
                        async with function.run(input_value) as runner:
                            # Get the result from the runner
                            result = await runner.result(to_type=str)
                    else:
                        # Regular function call
                        result = await function.acall_invoke(**kwargs)

                # Report completion
                if ctx:
                    await ctx.report_progress(100, 100)

                # Handle different result types for proper formatting
                if isinstance(result, str):
                    return result
                if isinstance(result, (dict, list)):
                    return json.dumps(result, default=str)
                return str(result)
            except Exception as e:
                if ctx:
                    ctx.error("Error calling function %s: %s", function_name, str(e))
                raise

        return wrapper_with_ctx

    # Create the wrapper function
    wrapper = create_wrapper()

    # Set the signature on the wrapper function (WITHOUT ctx)
    wrapper.__signature__ = sig
    wrapper.__name__ = function_name

    # Return the wrapper with proper signature
    return wrapper


def get_function_description(function: FunctionBase) -> str:
    """
    Retrieve a human-readable description for a NAT function or workflow.

    The description is determined using the following precedence:
       1. If the function is a Workflow and has a 'description' attribute, use it.
       2. If the Workflow's config has a 'topic', use it.
       3. If the Workflow's config has a 'description', use it.
       4. If the function is a regular Function, use its 'description' attribute.

    Args:
        function: The NAT FunctionBase instance (Function or Workflow).

    Returns:
        The best available description string for the function.
    """
    function_description = ""

    if isinstance(function, Workflow):
        config = function.config

        # Workflow doesn't have a description, but probably should
        if hasattr(function, "description") and function.description:
            function_description = function.description
        # Try to get description from config
        elif hasattr(config, "description") and config.description:
            function_description = config.description
        # Try to get anything that might be a description
        elif hasattr(config, "topic") and config.topic:
            function_description = config.topic

    elif isinstance(function, Function):
        function_description = function.description

    return function_description


def register_function_with_mcp(mcp: FastMCP, function_name: str, function: FunctionBase) -> None:
    """Register a NAT Function as an MCP tool.

    Args:
        mcp: The FastMCP instance
        function_name: The name to register the function under
        function: The NAT Function to register
    """
    logger.info("Registering function %s with MCP", function_name)

    # Get the input schema from the function
    input_schema = function.input_schema
    logger.info("Function %s has input schema: %s", function_name, input_schema)

    # Check if we're dealing with a Workflow
    is_workflow = isinstance(function, Workflow)
    if is_workflow:
        logger.info("Function %s is a Workflow", function_name)

    # Get function description
    function_description = get_function_description(function)

    # Create and register the wrapper function with MCP
    wrapper_func = create_function_wrapper(function_name, function, input_schema, is_workflow)
    mcp.tool(name=function_name, description=function_description)(wrapper_func)
