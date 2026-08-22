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

import json
import keyword
import logging
import re
from inspect import Parameter
from inspect import Signature
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context
from nat.builder.function import Function
from nat.builder.function_base import FunctionBase

if TYPE_CHECKING:
    from nat.plugins.mcp.server.memory_profiler import MemoryProfiler
    from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)

# Reserved wrapper parameter for MCP request Context injection.
INJECTED_CONTEXT_PARAM = "_nat_mcp_context"

_USE_PYDANTIC_DEFAULT = object()


def _use_pydantic_default() -> object:
    """Mark an omitted `FastMCP` argument for removal before model validation."""
    return _USE_PYDANTIC_DEFAULT


def _build_mcp_field_annotation(field: FieldInfo) -> Any:
    """Build a field annotation for `FastMCP`'s intermediate argument model.

    `FastMCP` materializes default-factory values before calling the wrapper. Use
    a private marker factory in that intermediate model so the declared schema
    remains responsible for applying its original factory and tracking whether
    the caller supplied the field.
    """
    field_metadata: list[Any] = [field]
    if field.default_factory is not None:
        # The type-agnostic marker need not satisfy the field type. The wrapper
        # removes it before the declared schema validates the original factory's value.
        field_metadata.append(Field(default_factory=_use_pydantic_default, validate_default=False))

    return Annotated[(field.annotation or Any, *field_metadata)]


def _sanitize_parameter_name(name: str) -> str:
    """Sanitize a JSON schema property name into a valid Python identifier.

    The MCP specification places no restriction on property names, but Python's
    ``inspect.Parameter`` requires valid identifiers.  This function:

    1. Replaces non-alphanumeric/underscore characters with ``_``.
    2. Prepends ``_`` when the result starts with a digit.
    3. Appends ``_`` when the result is a Python keyword (PEP 8 convention).

    Args:
        name: The original property name from the MCP tool schema.

    Returns:
        A valid Python identifier derived from *name*.
    """
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if not safe or safe[0].isdigit():
        safe = '_' + safe
    if keyword.iskeyword(safe):
        safe = safe + '_'
    return safe


def _build_name_mapping(field_names: list[str]) -> dict[str, str]:
    """Map each original schema field name to a unique, valid Python identifier.

    Handles collisions by appending numeric suffixes when multiple original names
    sanitize to the same Python identifier (e.g. ``from`` and ``from_`` both
    sanitize to ``from_``; the second one becomes ``from_2``).

    Names that are already valid Python identifiers map to themselves.

    Args:
        field_names: Original field names from the Pydantic model.

    Returns:
        A dict mapping ``{original_name: safe_name}`` for every field.
    """
    name_map: dict[str, str] = {}
    used: set[str] = set()

    # First pass: reserve names that need no sanitization
    for name in field_names:
        safe = _sanitize_parameter_name(name)
        if safe == name:
            name_map[name] = name
            used.add(name)

    # Second pass: assign safe names for fields that need sanitization
    for name in field_names:
        if name in name_map:
            continue
        safe = _sanitize_parameter_name(name)
        candidate = safe
        suffix = 2
        while candidate in used:
            candidate = f"{safe}{suffix}"
            suffix += 1
        name_map[name] = candidate
        used.add(candidate)

    return name_map


def _validate_input_schema_for_context_injection(schema: type[BaseModel], name_map: dict[str, str]) -> None:
    """Reject schemas whose fields collide with the injected Context parameter."""
    if INJECTED_CONTEXT_PARAM in name_map.values():
        raise ValueError(
            f"Workflow input schema cannot declare a field that maps to reserved MCP parameter "
            f"{INJECTED_CONTEXT_PARAM!r}.", )
    if "ctx" in schema.model_fields:
        raise ValueError(
            "Workflow input schema cannot declare a field named 'ctx'; that name is reserved for "
            "MCP request context injection.", )


def is_field_optional(field: FieldInfo) -> tuple[bool, Any]:
    """Determine if a Pydantic field is optional and extract its default value for MCP signatures.

    For MCP tool signatures, we need to distinguish:

    - Required fields: marked with Parameter.empty
    - Optional with concrete default: use that default
    - Optional with factory: omit the signature default; the annotated field uses
      an internal marker so the declared schema can apply its factory later

    Args:
        field: The Pydantic FieldInfo to check

    Returns:
        A tuple of (is_optional, default_value):
        - (False, Parameter.empty) for required fields
        - (True, actual_default) for optional fields with explicit defaults
        - (True, Parameter.empty) for optional fields with default_factory
    """
    if field.is_required():
        return False, Parameter.empty

    # Field is optional - has either default or factory
    if field.default is not PydanticUndefined:
        return True, field.default

    # Factory case: the generated field annotation carries an internal marker factory.
    # Leaving the signature default empty avoids exposing a non-serializable
    # sentinel in the generated JSON schema.
    if field.default_factory is not None:
        return True, Parameter.empty

    # Rare corner case: non-required yet no default surfaced
    return True, Parameter.empty


async def _run_through_session_manager(session_manager: 'SessionManager',
                                       payload: BaseModel,
                                       ctx: Any | None = None,
                                       memory_profiler: 'MemoryProfiler | None' = None) -> Any:
    """Execute a payload through SessionManager, including per-user workflows."""
    try:
        if session_manager.is_workflow_per_user:
            from nat.builder.context import Context
            from nat.runtime.user_manager import UserManager

            user_id = Context.get().user_id
            http_connection = None
            if ctx is not None:
                try:
                    http_connection = ctx.request_context.request
                    if user_id is None and http_connection is not None:
                        user_info = UserManager.extract_user_from_connection(http_connection)
                        if user_info is not None:
                            user_id = user_info.get_user_id()
                except ValueError:
                    pass

            async with session_manager.session(user_id=user_id, http_connection=http_connection) as session:
                async with session.run(payload) as runner:
                    return await runner.result()

        async with session_manager.run(payload) as runner:
            return await runner.result()
    finally:
        if memory_profiler:
            memory_profiler.on_request_complete()


def create_function_wrapper(
    function_name: str,
    session_manager: 'SessionManager',
    schema: type[BaseModel],
    memory_profiler: 'MemoryProfiler | None' = None,
):
    """Create a wrapper function that exposes a NAT Function as an MCP tool using SessionManager.

    Here SessionManager.run() which is used to create a Runner that
    automatically handles observability (emits intermediate step events, starts exporters, etc).

    Args:
        function_name (str): The name of the function/tool
        session_manager (SessionManager): SessionManager wrapping the function/workflow
        schema (type[BaseModel]): The input schema of the function
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

        # Build collision-safe name mapping and derive the keys FastMCP may
        # emit for each declared schema field.
        name_map = _build_name_mapping(list(param_fields.keys()))
        argument_name_map = {safe: orig for orig, safe in name_map.items() if orig != safe}
        _validate_input_schema_for_context_injection(schema, name_map)

        parameters = []
        for name, field in param_fields.items():
            # Retain the Pydantic field metadata so FastMCP preserves its
            # constraints, descriptions, and optionality.
            # Look up the collision-safe parameter name
            safe_name = name_map[name]

            field_type = _build_mcp_field_annotation(field)
            if safe_name != name:
                # Only synthesize the original JSON name when the declared
                # field has no input alias semantics of its own.
                if field.validation_alias is None:
                    field_type = Annotated[field_type, Field(alias=name)]
                if isinstance(field.alias, str):
                    argument_name_map[field.alias] = name

            # Check if field is optional and get its default value
            _is_optional, param_default = is_field_optional(field)

            # Add the parameter to our list
            parameters.append(
                Parameter(
                    name=safe_name,
                    kind=Parameter.KEYWORD_ONLY,
                    default=param_default,
                    annotation=field_type,
                ))

    # Create the function signature WITHOUT the injected Context parameter
    # Context is declared on the wrapper for FastMCP injection but omitted from __signature__
    # so it is excluded from the client-facing tool schema.
    sig = Signature(parameters=parameters, return_annotation=str)

    # Define the actual wrapper function that accepts injected context but doesn't expose it
    def create_wrapper():

        async def wrapper_with_ctx(_nat_mcp_context: Context | None = None, **kwargs):
            """Internal wrapper that will be called by MCP.

            Uses SessionManager.run() which creates a Runner that automatically handles observability.
            """
            # FastMCP applies the wrapper field's factory to omitted arguments.
            # Remove its marker so the declared schema can apply the original
            # default factory and preserve model_fields_set semantics.
            kwargs = {k: v for k, v in kwargs.items() if v is not _USE_PYDANTIC_DEFAULT}

            # Process the function call
            if _nat_mcp_context:
                _nat_mcp_context.info("Calling function %s with args: %s",
                                      function_name,
                                      json.dumps(kwargs, default=str))
                await _nat_mcp_context.report_progress(0, 100)

            try:
                # Prepare input payload
                if is_chat_request:
                    from nat.data_models.api_server import ChatRequest
                    # Create a chat request from the query string
                    query = kwargs.get("query", "")
                    payload = ChatRequest.from_string(query)
                else:
                    # Canonicalize sanitized names and aliases emitted by FastMCP.
                    if argument_name_map:
                        kwargs = {argument_name_map.get(k, k): v for k, v in kwargs.items()}
                    # Always validate with the declared schema
                    payload = schema.model_validate(kwargs, by_name=True)

                # Use SessionManager.run() pattern - this automatically handles all observability
                # The Runner created by session_manager.run() will:
                # 1. Start the exporter manager
                # 2. Emit WORKFLOW_START/FUNCTION_START events
                # 3. Execute the function/workflow
                # 4. Emit WORKFLOW_END/FUNCTION_END events
                # 5. Stop the exporter manager
                result = await _run_through_session_manager(session_manager,
                                                            payload,
                                                            ctx=_nat_mcp_context,
                                                            memory_profiler=memory_profiler)

                # Report completion
                if _nat_mcp_context:
                    await _nat_mcp_context.report_progress(100, 100)

                # Handle different result types for proper formatting
                if isinstance(result, str):
                    return result
                if isinstance(result, dict | list):
                    return json.dumps(result, default=str)
                return str(result)
            except Exception as e:
                if _nat_mcp_context:
                    _nat_mcp_context.error("Error calling function %s: %s", function_name, str(e))

                raise

        return wrapper_with_ctx

    # Create the wrapper function
    wrapper = create_wrapper()

    # Set the signature on the wrapper function (WITHOUT injected context)
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
                               session_manager: 'SessionManager',
                               memory_profiler: 'MemoryProfiler | None' = None,
                               function: FunctionBase | None = None) -> None:
    """Register a NAT Function as an MCP tool using SessionManager.

    Each function is wrapped in a SessionManager
    so that all calls go through Runner that automatically handles observability.

    Args:
        mcp: The FastMCP instance
        function_name: The name to register the function under
        session_manager: SessionManager wrapping the function/workflow
        memory_profiler: Optional memory profiler to track requests
    """
    logger.info("Registering function %s with MCP", function_name)

    if session_manager.is_workflow_per_user:
        input_schema = session_manager.get_workflow_input_schema()
        workflow_config = session_manager.config.workflow
        function_description = getattr(workflow_config, "description", None) or function_name
    else:
        workflow = session_manager.workflow
        target_function = function or workflow
        input_schema = getattr(target_function, "input_schema", workflow.input_schema)
        function_description = get_function_description(target_function)

    logger.info("Function %s has input schema: %s", function_name, input_schema)

    # Create and register the wrapper function with MCP
    wrapper_func = create_function_wrapper(function_name, session_manager, input_schema, memory_profiler)
    mcp.tool(name=function_name, description=function_description)(wrapper_func)
