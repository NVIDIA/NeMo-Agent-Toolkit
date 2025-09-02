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

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from nat.builder.context import ContextState
from nat.builder.function import Function
from nat.builder.function_base import FunctionBase

if TYPE_CHECKING:
    from nat.builder.workflow import Workflow

logger = logging.getLogger(__name__)


async def send_mcp_log(mcp_server, level: str, logger_name: str, message: str, data: dict | None = None, config=None):
    """Send a log message via MCP protocol.

    Args:
        mcp_server: The FastMCP server instance
        level: Log level (debug, info, notice, warning, error, critical, alert, emergency)
        logger_name: Name of the logger
        message: Log message
        data: Additional structured data
        config: MCP frontend config for logging settings
    """
    if not mcp_server:
        return

    # Check if MCP logging is enabled
    if config and not config.enable_mcp_logging:
        return

    # Check log level filtering
    if config and config.mcp_log_level:
        log_levels = ["debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"]
        current_level_index = log_levels.index(level)
        min_level_index = log_levels.index(config.mcp_log_level)
        if current_level_index < min_level_index:
            return

    try:
        log_data = {"message": message}
        if data:
            log_data.update(data)

        await mcp_server.send_notification("notifications/message", {
            "level": level,
            "logger": logger_name,
            "data": log_data
        })
    except Exception as e:
        # Don't let logging errors break the main functionality
        logger.debug("Failed to send MCP log message: %s", e)


def create_function_wrapper(
    function_name: str,
    function: FunctionBase,
    schema: type[BaseModel],
    is_workflow: bool = False,
    workflow: 'Workflow | None' = None,
    mcp_server=None,
    config=None,
):
    """Create a wrapper function that exposes the actual parameters of a NAT Function as an MCP tool.

    Args:
        function_name (str): The name of the function/tool
        function (FunctionBase): The NAT Function object
        schema (type[BaseModel]): The input schema of the function
        is_workflow (bool): Whether the function is a Workflow
        workflow (Workflow | None): The parent workflow for observability context
        mcp_server: The FastMCP server instance for logging
        config: MCP frontend config for logging settings

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

            # Send MCP log for function start
            await send_mcp_log(mcp_server, "info", function_name, f"Starting function {function_name}", {
                "args": kwargs,
                "is_workflow": is_workflow
            }, config)

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
                    # Handle complex input schema - if we extracted fields from a nested schema,
                    # we need to reconstruct the input
                    if len(schema.model_fields) == 1 and len(parameters) > 1:
                        # Get the field name from the original schema
                        field_name = next(iter(schema.model_fields.keys()))
                        field_type = schema.model_fields[field_name].annotation

                        # If it's a pydantic model, we need to create an instance
                        if field_type and hasattr(field_type, "model_validate"):
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
                        result = await call_with_observability(lambda: function.acall_invoke(**kwargs))

                # Report completion
                if ctx:
                    await ctx.report_progress(100, 100)

                # Send MCP log for successful completion
                await send_mcp_log(mcp_server, "info", function_name, f"Function {function_name} completed successfully", {
                    "result_type": type(result).__name__,
                    "result_length": len(str(result)) if result else 0
                }, config)

                # Handle different result types for proper formatting
                if isinstance(result, str):
                    return result
                if isinstance(result, (dict, list)):
                    return json.dumps(result, default=str)
                return str(result)
            except Exception as e:
                # Send MCP log for errors
                await send_mcp_log(mcp_server, "error", function_name, f"Function {function_name} failed", {
                    "error": str(e),
                    "error_type": type(e).__name__
                }, config)

                if ctx:
                    ctx.error("Error calling function %s: %s", function_name, str(e))
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

    elif isinstance(function, Function):
        function_description = function.description

    return function_description or ""


def register_function_with_mcp(mcp: FastMCP,
                               function_name: str,
                               function: FunctionBase,
                               workflow: 'Workflow | None' = None,
                               mcp_server=None,
                               config=None) -> None:
    """Register a NAT Function as an MCP tool.

    Args:
        mcp: The FastMCP instance
        function_name: The name to register the function under
        function: The NAT Function to register
        workflow: The parent workflow for observability context (if available)
        mcp_server: The FastMCP server instance for logging (if None, uses mcp parameter)
        config: MCP frontend config for logging settings
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

    # Use mcp_server if provided, otherwise fall back to mcp parameter
    server_for_logging = mcp_server if mcp_server is not None else mcp

    # Create and register the wrapper function with MCP
    wrapper_func = create_function_wrapper(function_name, function, input_schema, is_workflow, workflow, server_for_logging, config)
    mcp.tool(name=function_name, description=function_description)(wrapper_func)
