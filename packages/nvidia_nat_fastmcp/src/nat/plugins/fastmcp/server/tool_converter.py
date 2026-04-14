# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Convert NeMo Agent Toolkit functions to FastMCP tools."""

import asyncio
import json
import logging
from inspect import Parameter
from inspect import Signature
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from nat.builder.function import Function  # type: ignore[reportMissingImports]
from nat.builder.function_base import FunctionBase  # type: ignore[reportMissingImports]
from nat.data_models.mcp_tool_result import AtifToolMeta  # type: ignore[reportMissingImports]
from nat.data_models.mcp_tool_result import StructuredToolContent  # type: ignore[reportMissingImports]
from nat.data_models.mcp_tool_result import ToolRuntimeMeta  # type: ignore[reportMissingImports]
from nat.runtime.session import SessionManager  # type: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

# Sentinel: marks "optional; let Pydantic supply default/factory"
_USE_PYDANTIC_DEFAULT = object()


class FastMCPToolOutputOptions(BaseModel):
    """Internal options controlling FastMCP tool output behavior."""
    structured_tool_output: bool = False
    include_atif_meta: bool = False


def _safe_json_schema(schema: Any) -> dict[str, Any]:
    """Return a JSON schema for Pydantic models or dict-like schemas."""
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    if isinstance(schema, dict):
        return schema
    return {}


def _get_field_default(field_info: FieldInfo) -> Any:
    """Return field default or a sentinel to skip default."""
    if field_info.default is not PydanticUndefined:
        return field_info.default
    if field_info.default_factory is not None:
        return _USE_PYDANTIC_DEFAULT
    return _USE_PYDANTIC_DEFAULT


def _build_signature_from_schema(schema: Any) -> Signature:
    """Build a function signature from a Pydantic schema if possible."""
    if _is_chat_request_schema(schema):
        return Signature(parameters=[
            Parameter(name="query", kind=Parameter.KEYWORD_ONLY, annotation=str),
        ])
    if not hasattr(schema, "model_fields"):
        return Signature()

    params: list[Parameter] = []
    for name, field_info in schema.model_fields.items():  # type: ignore[attr-defined]
        annotation = field_info.annotation or Any
        default = _get_field_default(field_info)
        if default is _USE_PYDANTIC_DEFAULT:
            params.append(Parameter(name, Parameter.KEYWORD_ONLY, annotation=annotation))
        else:
            params.append(Parameter(name, Parameter.KEYWORD_ONLY, default=default, annotation=annotation))

    return Signature(parameters=params)


def _build_input_schema(schema: Any) -> Any:
    """Return an input schema for tool registration."""
    if schema is None:
        return None

    if isinstance(schema, BaseModel) or hasattr(schema, "model_json_schema"):
        return schema

    if isinstance(schema, dict):
        return schema

    return None


def _build_annotations_from_schema(schema: Any) -> dict[str, Any]:
    """Build function annotations from a Pydantic schema if possible."""
    if _is_chat_request_schema(schema):
        return {"query": str}
    if not hasattr(schema, "model_fields"):
        return {}

    annotations: dict[str, Any] = {}
    for name, field_info in schema.model_fields.items():  # type: ignore[attr-defined]
        annotations[name] = field_info.annotation or Any
    return annotations


def _is_chat_request_schema(schema: Any) -> bool:
    """Return True when the schema represents a ChatRequest."""
    schema_name = getattr(schema, "__name__", "")
    schema_qualname = getattr(schema, "__qualname__", "")
    return schema_name == "ChatRequest" or "ChatRequest" in schema_qualname


def _format_result_text(result: Any) -> str:
    """Format tool result as text for MCP content compatibility."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict | list):
        return json.dumps(result, default=str)
    return str(result)


def _build_structured_content(result: Any, result_text: str) -> dict[str, Any]:
    """Build structured content for ToolResult payload."""
    payload = StructuredToolContent(result_text=result_text)
    if isinstance(result, dict):
        payload.result_json = result
    elif isinstance(result, list):
        payload.result_json = result
    return payload.model_dump(exclude_none=True)


def _build_output_schema() -> dict[str, Any]:
    """Return output schema used when structured tool output is enabled."""
    return StructuredToolContent.model_json_schema()


def _build_atif_meta(function_name: str, intermediate_step_dicts: list[dict[str, Any]]) -> AtifToolMeta | None:
    """Convert intermediate steps to inline ATIF metadata."""
    if not intermediate_step_dicts:
        return None

    from nat.data_models.intermediate_step import IntermediateStep  # type: ignore[reportMissingImports]
    from nat.utils.atif_converter import IntermediateStepToATIFConverter  # type: ignore[reportMissingImports]

    validated_steps: list[IntermediateStep] = []
    for raw_step in intermediate_step_dicts:
        try:
            validated_steps.append(IntermediateStep.model_validate(raw_step))
        except Exception:  # pragma: no cover - defensive for malformed stream events
            logger.exception("Skipping invalid intermediate step while building ATIF metadata")

    if not validated_steps:
        return None

    converter = IntermediateStepToATIFConverter()
    trajectory = converter.convert(validated_steps, agent_name=function_name)

    return AtifToolMeta(run_id=uuid4().hex,
                        schema_version=trajectory.schema_version,
                        trajectory=trajectory)


def create_function_wrapper(
    function_name: str,
    session_manager: "SessionManager",
    input_schema: Any,
    output_options: FastMCPToolOutputOptions | None = None,
):
    """Create a wrapper function for MCP that invokes the workflow via `SessionManager`.

    Args:
        function_name: The name of the function to register.
        session_manager: The session manager for the workflow.
        input_schema: Input schema for the workflow/function.
        output_options: Optional tool output behavior flags.
    """
    output_options = output_options or FastMCPToolOutputOptions()
    signature = _build_signature_from_schema(input_schema)

    async def wrapper_func(**kwargs: Any) -> Any:
        if _is_chat_request_schema(input_schema):
            from nat.data_models.api_server import ChatRequest  # type: ignore[reportMissingImports]

            query = kwargs.get("query", "")
            payload = ChatRequest.from_string(query)
        else:
            cleaned_kwargs = {k: v for k, v in kwargs.items() if v is not _USE_PYDANTIC_DEFAULT}
            payload = input_schema.model_validate(cleaned_kwargs) if hasattr(input_schema,
                                                                             "model_validate") else cleaned_kwargs

        intermediate_task: asyncio.Task | None = None
        if output_options.include_atif_meta:
            from nat.builder.runtime_event_subscriber import pull_intermediate  # type: ignore[reportMissingImports]
            intermediate_task = asyncio.ensure_future(pull_intermediate())

        async with session_manager.run(payload) as runner:
            result = await runner.result()

        result_text = _format_result_text(result)
        use_tool_result = output_options.structured_tool_output or output_options.include_atif_meta
        if not use_tool_result:
            return result_text

        structured_content = _build_structured_content(result, result_text) if output_options.structured_tool_output else None
        meta: dict[str, Any] | None = None
        if output_options.include_atif_meta:
            intermediate_step_dicts: list[dict[str, Any]] = []
            if intermediate_task is not None:
                intermediate_step_dicts = await intermediate_task
            atif_meta = _build_atif_meta(function_name, intermediate_step_dicts)
            if atif_meta is not None:
                meta = ToolRuntimeMeta(atif=atif_meta).model_dump(exclude_none=True)

        return ToolResult(content=result_text, structured_content=structured_content, meta=meta)

    wrapper_func.__signature__ = signature  # type: ignore[attr-defined]
    wrapper_func.__annotations__ = _build_annotations_from_schema(input_schema)
    wrapper_func.__name__ = function_name
    wrapper_func.__doc__ = "Auto-generated wrapper for a NeMo Agent Toolkit workflow."
    return wrapper_func


def get_function_description(function: FunctionBase | None) -> str | None:
    """Retrieve a human-readable description for a NAT function or workflow."""
    if function is None:
        return None

    from nat.builder.workflow import Workflow  # type: ignore[reportMissingImports]

    function_description: str | None = None

    if isinstance(function, Workflow):
        config = function.config

        if hasattr(function, "description") and function.description:
            function_description = function.description
        elif hasattr(config, "description") and config.description:
            function_description = config.description
        elif hasattr(config, "topic") and config.topic:
            function_description = config.topic
        elif hasattr(config, "workflow") and hasattr(config.workflow, "description") and config.workflow.description:
            function_description = config.workflow.description
    elif isinstance(function, Function):
        function_description = function.description

    return function_description


def register_function_with_mcp(mcp: FastMCP,
                               function_name: str,
                               session_manager: 'SessionManager',
                               function: FunctionBase | None = None,
                               output_options: FastMCPToolOutputOptions | None = None) -> None:
    """Register a NeMo Agent Toolkit function as a FastMCP tool.

    Each function is wrapped in a `SessionManager` so that all calls go through
    the runner, which automatically handles observability.

    Args:
        mcp: The FastMCP instance.
        function_name: The name to register the function under.
        session_manager: SessionManager wrapping the function/workflow.
        function: Optional function metadata (for description/schema).
    """
    logger.info("Registering function %s with FastMCP", function_name)

    # Get the workflow from the session manager
    workflow = session_manager.workflow

    # Prefer the function's schema/description when available, fall back to workflow
    target_function = function or workflow

    # Get the input schema from the most specific object available
    input_schema = getattr(target_function, "input_schema", workflow.input_schema)
    logger.info("Function %s has input schema: %s", function_name, input_schema)

    # Get function description
    function_description = get_function_description(target_function)

    # Create and register the wrapper function with FastMCP
    wrapper_func = create_function_wrapper(function_name, session_manager, input_schema, output_options=output_options)
    if output_options and output_options.structured_tool_output:
        mcp.tool(name=function_name, description=function_description, output_schema=_build_output_schema())(wrapper_func)
    else:
        mcp.tool(name=function_name, description=function_description)(wrapper_func)


def format_schema_for_display(schema: Any) -> str:
    """Return a pretty JSON schema string for debug endpoints."""
    schema_dict = _safe_json_schema(schema)
    return json.dumps(schema_dict, indent=2)
