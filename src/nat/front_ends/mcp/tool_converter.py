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
from typing import TYPE_CHECKING
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from nat.builder.context import ContextState
from nat.builder.function import Function
from nat.builder.function_base import FunctionBase

if TYPE_CHECKING:
    from nat.builder.workflow import Workflow
    from nat.front_ends.mcp.memory_profiler import MemoryProfiler

logger = logging.getLogger(__name__)

# Sentinel: marks "optional; let Pydantic supply default/factory"
_USE_PYDANTIC_DEFAULT = object()


def is_field_optional(field: FieldInfo) -> tuple[bool, Any]:
    """Determine if a Pydantic field is optional and extract its default value for MCP signatures.

    For MCP tool signatures, we need to distinguish:
    - Required fields: marked with Parameter.empty
    - Optional with concrete default: use that default
    - Optional with factory: use sentinel so Pydantic can apply the factory later

    Args:
        field: The Pydantic FieldInfo to check

    Returns:
        A tuple of (is_optional, default_value):
        - (False, Parameter.empty) for required fields
        - (True, actual_default) for optional fields with explicit defaults
        - (True, _USE_PYDANTIC_DEFAULT) for optional fields with default_factory
    """
    if field.is_required():
        return False, Parameter.empty

    # Field is optional - has either default or factory
    if field.default is not PydanticUndefined:
        return True, field.default

    # Factory case: mark optional in signature but don't fabricate a value
    if field.default_factory is not None:
        return True, _USE_PYDANTIC_DEFAULT

    # Rare corner case: non-required yet no default surfaced
    return True, _USE_PYDANTIC_DEFAULT


def create_function_wrapper(
    function_name: str,
    function: FunctionBase,
    schema: type[BaseModel],
    is_workflow: bool = False,
    workflow: 'Workflow | None' = None,
    memory_profiler: 'MemoryProfiler | None' = None,
):
    """Create a wrapper function that exposes the actual parameters of a NAT Function as an MCP tool.

    Args:
        function_name (str): The name of the function/tool
        function (FunctionBase): The NAT Function object
        schema (type[BaseModel]): The input schema of the function
        is_workflow (bool): Whether the function is a Workflow
        workflow (Workflow | None): The parent workflow for observability context
        memory_profiler: Optional memory profiler to track requests

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

            # Check if field is optional and get its default value
            _is_optional, param_default = is_field_optional(field)

            # Add the parameter to our list
            parameters.append(
                Parameter(
                    name=name,
                    kind=Parameter.KEYWORD_ONLY,
                    default=param_default,
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
                # Helper function to wrap function calls with observability
                async def call_with_observability(func_call):
                    # Use workflow's observability context (workflow should always be available)
                    if not workflow:
                        logger.error("Missing workflow context for function %s - observability will not be available",
                                     function_name)
                        raise RuntimeError("Workflow context is required for observability")

                    logger.debug("Starting observability context for function %s", function_name)
                    context_state = ContextState.get()
                    async with workflow.exporter_manager.start(context_state=context_state):
                        return await func_call()

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
                        result = await call_with_observability(lambda: function.ainvoke(chat_request, to_type=str))
                else:
                    # Regular handling
                    # Strip sentinel values so Pydantic can apply defaults/factories
                    cleaned_kwargs = {k: v for k, v in kwargs.items() if v is not _USE_PYDANTIC_DEFAULT}

                    # Always validate with the declared schema
                    # This handles defaults, factories, nested models, validators, etc.
                    model_input = schema.model_validate(cleaned_kwargs)

                    # Call the NAT function with the parameters - special handling for Workflow
                    if is_workflow:
                        # Workflows expect the model instance directly
                        async with function.run(model_input) as runner:
                            # Get the result from the runner
                            result = await runner.result(to_type=str)
                    else:
                        # Regular function call - unpack the validated model
                        result = await call_with_observability(lambda: function.acall_invoke(**model_input.model_dump())
                                                               )

                # Report completion
                if ctx:
                    await ctx.report_progress(100, 100)

                # Track request completion for memory profiling
                if memory_profiler:
                    memory_profiler.on_request_complete()

                # Handle different result types for proper formatting
                if isinstance(result, str):
                    return result
                if isinstance(result, dict | list):
                    return json.dumps(result, default=str)
                return str(result)
            except Exception as e:
                if ctx:
                    ctx.error("Error calling function %s: %s", function_name, str(e))

                # Track request completion even on error
                if memory_profiler:
                    memory_profiler.on_request_complete()

                raise

        return wrapper_with_ctx

    # Create the wrapper function
    wrapper = create_wrapper()

    # Set the signature on the wrapper function (WITHOUT ctx)
    wrapper.__signature__ = sig  # type: ignore
    wrapper.__name__ = function_name

    # Return the wrapper with proper signature
    return wrapper


def get_function_description(function: FunctionBase) -> str:
    """
    Retrieve a human-readable description for a NAT function or workflow.

    The description is determined using the following precedence:
       1. If the function is a Workflow and has a 'description' attribute, use it.
       2. If the Workflow's config has a 'description', use it.
       3. If the Workflow's config has a 'topic', use it.
       4. If the function is a regular Function, use its 'description' attribute.

    Args:
        function: The NAT FunctionBase instance (Function or Workflow).

    Returns:
        The best available description string for the function.
    """
    function_description = ""

    # Import here to avoid circular imports
    from nat.builder.workflow import Workflow

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
        # Try to get description from the workflow config
        elif hasattr(config, "workflow") and hasattr(config.workflow, "description") and config.workflow.description:
            function_description = config.workflow.description

    elif isinstance(function, Function):
        function_description = function.description

    return function_description


def register_function_with_mcp(mcp: FastMCP,
                               function_name: str,
                               function: FunctionBase,
                               workflow: 'Workflow | None' = None,
                               memory_profiler: 'MemoryProfiler | None' = None) -> None:
    """Register a NAT Function as an MCP tool.

    Args:
        mcp: The FastMCP instance
        function_name: The name to register the function under
        function: The NAT Function to register
        workflow: The parent workflow for observability context (if available)
        memory_profiler: Optional memory profiler to track requests
    """
    logger.info("Registering function %s with MCP", function_name)

    # Get the input schema from the function
    input_schema = function.input_schema
    logger.info("Function %s has input schema: %s", function_name, input_schema)

    # Check if we're dealing with a Workflow
    from nat.builder.workflow import Workflow
    is_workflow = isinstance(function, Workflow)
    if is_workflow:
        logger.info("Function %s is a Workflow", function_name)

    # Get function description
    function_description = get_function_description(function)

    # Create and register the wrapper function with MCP
    wrapper_func = create_function_wrapper(function_name,
                                           function,
                                           input_schema,
                                           is_workflow,
                                           workflow,
                                           memory_profiler)
    mcp.tool(name=function_name, description=function_description)(wrapper_func)
