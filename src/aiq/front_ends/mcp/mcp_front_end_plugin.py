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

from aiq.builder.front_end import FrontEndBase
from aiq.builder.function import Function
from aiq.builder.workflow import Workflow
from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.front_ends.mcp.mcp_front_end_config import MCPFrontEndConfig
from aiq.runtime.session import AIQSessionManager
from pydantic import BaseModel
from typing import Dict
import logging
import json
import inspect

logger = logging.getLogger(__name__)


class MCPFrontEndPlugin(FrontEndBase[MCPFrontEndConfig]):
    """MCP front end plugin implementation."""

    async def run(self) -> None:
        """Run the MCP server."""
        # Import FastMCP
        from mcp.server.fastmcp import FastMCP, Context

        # Create an MCP server with the configured parameters
        mcp = FastMCP(
            self.front_end_config.name,
            host=self.front_end_config.host,
            port=self.front_end_config.port,
            debug=self.front_end_config.debug,
            log_level=self.front_end_config.log_level,
        )

        # Build the workflow and register all functions with MCP
        async with WorkflowBuilder.from_config(config=self.full_config) as builder:
            # Build the workflow
            workflow = builder.build()

            # Create a session manager
            session_manager = AIQSessionManager(workflow)

            # Get all functions from the workflow
            functions = self._get_all_functions(workflow)

            # Filter functions based on tool_names if provided
            if self.front_end_config.tool_names:
                logger.info(f"Filtering functions based on tool_names: {self.front_end_config.tool_names}")
                filtered_functions: Dict[str, Function] = {}
                for function_name, function in functions.items():
                    if function_name in self.front_end_config.tool_names:
                        filtered_functions[function_name] = function
                    else:
                        logger.debug(f"Skipping function {function_name} as it's not in tool_names")
                functions = filtered_functions

            # Register each function with MCP
            for function_name, function in functions.items():
                logger.info(f"Registering function {function_name} with MCP")

                # Get the input schema from the function
                input_schema = function.input_schema
                logger.info(f"Function {function_name} has input schema: {input_schema}")

                # Check if we're dealing with a Workflow
                is_workflow = False
                from aiq.builder.workflow import Workflow

                if isinstance(function, Workflow):
                    is_workflow = True
                    logger.info(f"Function {function_name} is a Workflow")
                elif (
                    hasattr(function, "__class__")
                    and hasattr(function.__class__, "__name__")
                    and function.__class__.__name__ == "Workflow"
                ):
                    is_workflow = True
                    logger.info(f"Function {function_name} is a Workflow (detected by class name)")

                # Get function description following priority:
                # 1. topic from config
                # 2. description from config
                # 3. description from function
                # 4. docstring first paragraph
                function_description = ""
                function_config = function.config

                # Try to get topic from config (highest priority)
                if hasattr(function_config, "topic") and function_config.topic:
                    function_description = function_config.topic
                # Try to get description from config
                elif hasattr(function_config, "description") and function_config.description:
                    function_description = function_config.description
                # Try to get description directly from the function
                elif hasattr(function, "description") and function.description:
                    function_description = function.description
                # Fall back to function docstring
                elif hasattr(function, "__doc__") and function.__doc__:
                    # Extract the first paragraph from the docstring
                    function_description = function.__doc__.strip().split("\n\n")[0]
                # Last resort - use function name
                else:
                    function_description = f"Function {function_name}"

                # Create a dynamic wrapper function that exposes the actual parameters
                def create_function_wrapper(func: Function, schema: type[BaseModel]):
                    # Check if we're dealing with AIQChatRequest - special case
                    from inspect import Parameter, Signature

                    is_chat_request = False

                    # Check if the schema name is AIQChatRequest
                    if schema.__name__ == "AIQChatRequest" or (
                        hasattr(schema, "__qualname__") and "AIQChatRequest" in schema.__qualname__
                    ):
                        is_chat_request = True
                        logger.info(f"Function {function_name} uses AIQChatRequest - creating simplified interface")

                        # For AIQChatRequest, we'll create a simple wrapper with just a query parameter
                        parameters = [
                            Parameter(
                                name="query", kind=Parameter.KEYWORD_ONLY, default=Parameter.empty, annotation=str
                            )
                        ]

                        field_info = {
                            "query": {"type": str, "description": "Input query or prompt text", "required": True}
                        }
                    else:
                        # Regular case - extract parameter information from the input schema
                        # Extract parameter information from the input schema
                        param_fields = schema.model_fields

                        from pydantic import create_model
                        import inspect
                        import re

                        # Parse docstring if available to get parameter descriptions
                        param_descriptions = {}
                        if func.__doc__:
                            docstring = func.__doc__
                            # Look for parameter descriptions in the docstring (Args: section)
                            args_section = re.search(r"Args:(.*?)(?:Returns:|Yields:|Raises:|$)", docstring, re.DOTALL)
                            if args_section:
                                args_text = args_section.group(1)
                                # Extract parameter descriptions using regex
                                param_matches = re.findall(r"(\w+):\s*(.*?)(?=\n\s*\w+:|$)", args_text, re.DOTALL)
                                for param_name, param_desc in param_matches:
                                    param_descriptions[param_name] = param_desc.strip()

                        # For simple input handling, extract fields from the schema definition
                        # This is needed because some AIQ functions might have a single composite input
                        field_info = {}

                        # If we only have one field and it's a complex type, try to extract its fields
                        parameters = []

                        # Check if we're dealing with a simple wrapper schema with a single field
                        # This is common in AIQ where functions often have a single field that contains the actual parameters
                        if len(param_fields) == 1:
                            # Get the field and check if it's a complex type
                            field_name, field = next(iter(param_fields.items()))
                            field_type = field.annotation

                            # If it's a string type, keep it simple - just expose the single parameter
                            if field_type == str or getattr(field_type, "__origin__", None) == str:
                                # This is a simple string field, use it directly
                                parameters.append(
                                    Parameter(
                                        name=field_name,
                                        kind=Parameter.KEYWORD_ONLY,
                                        default=Parameter.empty if field.is_required else None,
                                        annotation=field_type,
                                    )
                                )
                                # Get description from docstring if available
                                desc = param_descriptions.get(
                                    field_name, field.description or f"Parameter {field_name}"
                                )
                                field_info[field_name] = {
                                    "type": field_type,
                                    "description": desc,
                                    "required": field.is_required,
                                }
                            # If it's a complex type, try to extract its fields
                            # This is common with AIQ functions that accept a single object with multiple fields
                            elif hasattr(field_type, "model_fields"):
                                # This is a complex type, extract its fields
                                sub_fields = field_type.model_fields
                                for sub_name, sub_field in sub_fields.items():
                                    parameters.append(
                                        Parameter(
                                            name=sub_name,
                                            kind=Parameter.KEYWORD_ONLY,
                                            default=Parameter.empty if sub_field.is_required else None,
                                            annotation=sub_field.annotation,
                                        )
                                    )
                                    # Get description from docstring if available
                                    desc = param_descriptions.get(
                                        sub_name, sub_field.description or f"Parameter {sub_name}"
                                    )
                                    field_info[sub_name] = {
                                        "type": sub_field.annotation,
                                        "description": desc,
                                        "required": sub_field.is_required,
                                    }
                            else:
                                # This is some other type, use it directly
                                parameters.append(
                                    Parameter(
                                        name=field_name,
                                        kind=Parameter.KEYWORD_ONLY,
                                        default=Parameter.empty if field.is_required else None,
                                        annotation=field_type,
                                    )
                                )
                                # Get description from docstring if available
                                desc = param_descriptions.get(
                                    field_name, field.description or f"Parameter {field_name}"
                                )
                                field_info[field_name] = {
                                    "type": field_type,
                                    "description": desc,
                                    "required": field.is_required,
                                }
                        else:
                            # Multiple fields in the schema, use them directly
                            for name, field in param_fields.items():
                                # Get the field type and convert to appropriate Python type
                                field_type = field.annotation

                                # Get description from docstring if available
                                desc = param_descriptions.get(name, field.description or f"Parameter {name}")

                                # Store field info for documentation
                                field_info[name] = {
                                    "type": field_type,
                                    "description": desc,
                                    "required": field.is_required,
                                }

                                # Add the parameter to our list
                                parameters.append(
                                    Parameter(
                                        name=name,
                                        kind=Parameter.KEYWORD_ONLY,
                                        default=Parameter.empty if field.is_required else None,
                                        annotation=field_type,
                                    )
                                )

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
                                ctx.info(
                                    f"Calling function {function_name} with args: {json.dumps(kwargs, default=str)}"
                                )
                                await ctx.report_progress(0, 100)

                            try:
                                # Special handling for AIQChatRequest
                                if is_chat_request:
                                    from aiq.data_models.api_server import AIQChatRequest

                                    # Create a chat request from the query string
                                    query = kwargs.get("query", "")
                                    chat_request = AIQChatRequest.from_string(query)

                                    # Special handling for Workflow objects
                                    if is_workflow:
                                        # Workflows have a run method that is an async context manager
                                        # that returns an AIQRunner
                                        async with func.run(chat_request) as runner:
                                            # Get the result from the runner
                                            result = await runner.result(to_type=str)
                                    else:
                                        # Regular functions use ainvoke
                                        result = await func.ainvoke(chat_request, to_type=str)
                                else:
                                    # Regular handling
                                    # Handle complex input schema - if we extracted fields from a nested schema,
                                    # we need to reconstruct the input
                                    if len(schema.model_fields) == 1 and len(field_info) > 1:
                                        # Get the field name from the original schema
                                        field_name = next(iter(schema.model_fields.keys()))
                                        field_type = schema.model_fields[field_name].annotation

                                        # If it's a pydantic model, we need to create an instance
                                        if hasattr(field_type, "model_validate"):
                                            # Create the nested object
                                            nested_obj = field_type.model_validate(kwargs)
                                            # Call with the nested object
                                            kwargs = {field_name: nested_obj}

                                    # Call the AIQ function with the parameters - special handling for Workflow
                                    if is_workflow:
                                        # For workflow with regular input, we'll assume the first parameter is the input
                                        input_value = list(kwargs.values())[0] if kwargs else ""

                                        # Workflows have a run method that is an async context manager
                                        # that returns an AIQRunner
                                        async with func.run(input_value) as runner:
                                            # Get the result from the runner
                                            result = await runner.result(to_type=str)
                                    else:
                                        # Regular function call
                                        result = await func.acall_invoke(**kwargs)

                                # Report completion
                                if ctx:
                                    await ctx.report_progress(100, 100)

                                # Handle different result types for proper formatting
                                if isinstance(result, str):
                                    return result
                                elif isinstance(result, (dict, list)):
                                    return json.dumps(result, default=str)
                                else:
                                    return str(result)
                            except Exception as e:
                                if ctx:
                                    ctx.error(f"Error calling function {function_name}: {str(e)}")
                                raise

                        return wrapper_with_ctx

                    # Create the wrapper function
                    wrapper = create_wrapper()

                    # Set the signature on the wrapper function (WITHOUT ctx)
                    wrapper.__signature__ = sig
                    wrapper.__name__ = function_name

                    # Create a proper docstring
                    doc_lines = []
                    doc_lines.append(function_description)

                    # Add parameter documentation
                    for name, info in field_info.items():
                        doc_lines.append(f"  {name}: {info['description']}")

                    wrapper.__doc__ = "\n".join(doc_lines)

                    # Return the wrapper with proper signature
                    return wrapper

                # Create and register the wrapper function with MCP
                wrapper_func = create_function_wrapper(function, input_schema)
                mcp.tool(name=function_name, description=function_description)(wrapper_func)

            # Add a simple fallback function if no functions were found
            if not functions:
                logger.warning("No functions found in workflow, adding placeholder function")

                @mcp.tool()
                def add(a: int, b: int) -> int:
                    """Add two numbers."""
                    return a + b

            # Start the MCP server
            await mcp.run_sse_async()

    def _get_all_functions(self, workflow: Workflow) -> Dict[str, Function]:
        """Get all functions from the workflow.

        Args:
            workflow: The AIQ workflow.

        Returns:
            Dict mapping function names to Function objects.
        """
        functions: Dict[str, Function] = {}

        # Extract all functions from the workflow
        for function_name, function in workflow.functions.items():
            functions[function_name] = function

        functions[workflow.config.workflow.type] = workflow

        return functions 