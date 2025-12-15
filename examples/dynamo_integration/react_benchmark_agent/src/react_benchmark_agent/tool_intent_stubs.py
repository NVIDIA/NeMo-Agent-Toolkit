# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tool Intent Stub System for Decision-Only Evaluation.

This module provides a mechanism to capture tool-intent decisions without executing actual tools.
Each stub:
1. Reads expected parameters from the tool schema
2. Records the invocation (tool_name, parameters) to a shared buffer
3. Returns a canned response so the agent continues reasoning
"""

import contextvars
import json
import logging
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

# NOTE: FunctionInfo import unused since register_tool_stubs_from_schemas is commented out
# from nat.builder.function_info import FunctionInfo

logger = logging.getLogger(__name__)

# Global registry for tool intents (accessible across module)
# This allows evaluators to retrieve captured intents
_GLOBAL_INTENT_REGISTRY: dict[str, list[dict[str, Any]]] = {}

# Context variable for current scenario ID (async-safe for concurrent execution isolation)
# Unlike threading.local(), contextvars work correctly with asyncio tasks
_current_scenario_id: contextvars.ContextVar[str] = contextvars.ContextVar("scenario_id", default="current")


def set_current_scenario_id(scenario_id: str) -> contextvars.Token:
    """
    Set the current scenario ID for this async context.
    
    This allows concurrent async workflows to isolate their intents.
    Call this before executing a workflow to ensure intents are recorded
    to the correct scenario.
    
    Args:
        scenario_id: Unique identifier for the current scenario/question
    
    Returns:
        Token that can be used to reset the scenario ID (for cleanup)
    """
    token = _current_scenario_id.set(scenario_id)
    # Initialize registry entry if needed
    if scenario_id not in _GLOBAL_INTENT_REGISTRY:
        _GLOBAL_INTENT_REGISTRY[scenario_id] = []
    logger.debug("Set current scenario ID to: %s", scenario_id)
    return token


# NOTE: Unused - commenting out for now
# def reset_scenario_id(token: contextvars.Token) -> None:
#     """
#     Reset the scenario ID to its previous value.
#     
#     Args:
#         token: Token returned from set_current_scenario_id
#     """
#     _current_scenario_id.reset(token)


def get_current_scenario_id() -> str:
    """
    Get the current scenario ID for this async context.
    
    Returns:
        The current scenario ID, or "current" if not set
    """
    return _current_scenario_id.get()


class ToolIntentBuffer:
    """
    Shared buffer to store tool intent captures during agent execution.
    
    This is used in decision-only mode to track which tools the agent
    decided to call and with what parameters, without actually executing them.
    
    Uses a global registry so evaluators can access intents across the codebase.
    """

    def __init__(self, scenario_id: str = "current"):
        """
        Initialize a tool intent buffer.
        
        Args:
            scenario_id: Identifier for the current scenario (for multi-scenario tracking)
        """
        self.scenario_id = scenario_id
        self.intents: list[dict[str, Any]] = []
        
        # Initialize global registry entry
        if scenario_id not in _GLOBAL_INTENT_REGISTRY:
            _GLOBAL_INTENT_REGISTRY[scenario_id] = []

    def record(self, tool_name: str, parameters: dict[str, Any]) -> None:
        """
        Record a tool intent.
        
        Args:
            tool_name: Name of the tool the agent decided to call
            parameters: Parameters the agent provided for the tool call
        """
        intent = {"tool": tool_name, "parameters": parameters}
        self.intents.append(intent)
        
        # Store in global registry using thread-local scenario ID for concurrent isolation
        current_scenario = get_current_scenario_id()
        if current_scenario not in _GLOBAL_INTENT_REGISTRY:
            _GLOBAL_INTENT_REGISTRY[current_scenario] = []
        _GLOBAL_INTENT_REGISTRY[current_scenario].append(intent)
        
        logger.debug("Recorded tool intent: %s (scenario: %s)", tool_name, current_scenario)

    def get_intents(self) -> list[dict[str, Any]]:
        """
        Get all recorded tool intents.
        
        Returns:
            List of tool intents with format [{"tool": "name", "parameters": {...}}]
        """
        return self.intents.copy()

    def clear(self) -> None:
        """Clear all recorded intents."""
        self.intents.clear()
        # Also clear from global registry
        _GLOBAL_INTENT_REGISTRY[self.scenario_id] = []
        logger.debug("Cleared tool intent buffer for scenario %s", self.scenario_id)


def get_global_intents(scenario_id: str = "current") -> list[dict[str, Any]]:
    """
    Retrieve tool intents from the global registry.
    
    This allows evaluators to access intents without needing builder access.
    
    Args:
        scenario_id: Identifier for the scenario
    
    Returns:
        List of tool intents
    """
    return _GLOBAL_INTENT_REGISTRY.get(scenario_id, []).copy()


def clear_global_intents(scenario_id: str = "current") -> None:
    """
    Clear intents from global registry.
    
    Args:
        scenario_id: Identifier for the scenario to clear
    """
    if scenario_id in _GLOBAL_INTENT_REGISTRY:
        _GLOBAL_INTENT_REGISTRY[scenario_id] = []
        logger.debug("Cleared global intents for scenario %s", scenario_id)


# =============================================================================
# NOTE: The following utility functions are unused and commented out for now.
# They may be useful for debugging or future enhancements.
# =============================================================================

# def cleanup_all_intents() -> int:
#     """
#     Clear ALL intents from the global registry across all scenarios.
#     
#     This should be called between evaluation runs to prevent memory accumulation
#     and ensure clean state for new evaluations.
#     
#     Returns:
#         Number of scenarios that were cleared
#     
#     Example:
#         >>> # Call at the start of an evaluation run
#         >>> from react_benchmark_agent.tool_intent_stubs import cleanup_all_intents
#         >>> cleared = cleanup_all_intents()
#         >>> print(f"Cleared {cleared} scenarios from intent registry")
#     """
#     global _GLOBAL_INTENT_REGISTRY
#     num_scenarios = len(_GLOBAL_INTENT_REGISTRY)
#     _GLOBAL_INTENT_REGISTRY.clear()
#     logger.info("Cleared all intents from global registry (%d scenarios)", num_scenarios)
#     return num_scenarios


# def get_all_scenario_ids() -> list[str]:
#     """
#     Get all scenario IDs currently in the global registry.
#     
#     Useful for debugging and monitoring intent accumulation.
#     
#     Returns:
#         List of scenario IDs with registered intents
#     """
#     return list(_GLOBAL_INTENT_REGISTRY.keys())


# def get_registry_stats() -> dict[str, Any]:
#     """
#     Get statistics about the global intent registry.
#     
#     Useful for monitoring memory usage and debugging.
#     
#     Returns:
#         Dictionary with registry statistics:
#         - num_scenarios: Number of scenarios tracked
#         - total_intents: Total number of intents across all scenarios
#         - intents_per_scenario: Dict mapping scenario_id to intent count
#     """
#     intents_per_scenario = {
#         scenario_id: len(intents) 
#         for scenario_id, intents in _GLOBAL_INTENT_REGISTRY.items()
#     }
#     return {
#         "num_scenarios": len(_GLOBAL_INTENT_REGISTRY),
#         "total_intents": sum(intents_per_scenario.values()),
#         "intents_per_scenario": intents_per_scenario,
#     }

# =============================================================================


class PermissiveToolInput(BaseModel):
    """
    Input schema that accepts tool parameters as either dict or JSON string.
    
    This handles the case where LangChain sometimes serializes tool inputs
    as JSON strings before passing them to the tool, while NAT expects dicts.
    """
    input_params: dict[str, Any] | str
    
    @field_validator('input_params', mode='before')
    @classmethod
    def parse_string_to_dict(cls, v: Any) -> dict[str, Any]:
        """Convert JSON string to dict if needed."""
        if isinstance(v, str):
            try:
                # Handle both single and double quotes in JSON strings
                normalized = v.replace("'", '"')
                return json.loads(normalized)
            except json.JSONDecodeError:
                logger.warning("Failed to parse input_params string as JSON: %s", v[:100])
                return {}
        elif isinstance(v, dict):
            return v
        else:
            logger.warning("Unexpected input_params type: %s", type(v))
            return {}


def create_tool_stub_function(
    tool_schema: dict[str, Any], intent_buffer: ToolIntentBuffer, canned_response: str | None = None
) -> tuple[callable, BaseModel | None, str]:
    """
    Create a stub function for a tool that captures intent without executing.
    
    Args:
        tool_schema: Tool schema from the dataset (includes title, description, properties, required)
        intent_buffer: Shared buffer to record tool intents
        canned_response: Optional canned response to return (defaults to success message)
    
    Returns:
        Tuple of (async_function, input_schema, function_description)
        Note: Returns custom input_schema with no validation to accept any parameter format
    """
    tool_name = tool_schema.get("title", "unknown_tool")
    tool_description = tool_schema.get("description", "")
    properties = tool_schema.get("properties", {})
    required_params = tool_schema.get("required", [])

    # Default canned response
    if canned_response is None:
        response_schema = tool_schema.get("response_schema", {})
        if response_schema:
            # Generate a realistic-looking response based on schema
            canned_response = json.dumps(_generate_mock_response(response_schema), indent=2)
        else:
            canned_response = f"Successfully executed {tool_name}. Operation completed."

    # Create stub function that accepts object input (broadest concrete type)
    # The PermissiveToolInput validator will handle string-to-dict conversion
    async def tool_stub_fn(input_params: object) -> str:
        """Tool stub that captures intent without executing."""
        # At this point, input_params should be a dict thanks to the Pydantic validator
        # Handle nested 'params' dict from LangChain if present
        if isinstance(input_params, dict):
            if 'params' in input_params and isinstance(input_params['params'], dict):
                params_dict = input_params['params']
            else:
                params_dict = input_params
        else:
            # Fallback in case validation didn't run
            logger.warning("input_params is not a dict: %s", type(input_params))
            params_dict = {}
        
        # Filter out None values
        if isinstance(params_dict, dict):
            params_dict = {k: v for k, v in params_dict.items() if v is not None}
        intent_buffer.record(tool_name, params_dict)
        logger.info("Tool stub executed: %s with %d parameters", tool_name, len(params_dict))
        return canned_response
    
    # Set proper attributes
    tool_stub_fn.__name__ = tool_name
    tool_stub_fn.__doc__ = tool_description
    
    # Return function WITH custom input_schema that accepts both dict and string
    return tool_stub_fn, PermissiveToolInput, tool_description


def _generate_mock_response(response_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a mock response based on the response schema.
    
    Args:
        response_schema: Response schema from the tool definition
    
    Returns:
        Dictionary with mock values matching the schema
    """
    mock_response = {}
    properties = response_schema.get("properties", {})

    for prop_name, prop_info in properties.items():
        prop_type = prop_info.get("type", "string")
        prop_desc = prop_info.get("description", "")

        # Generate mock values based on type
        if prop_type == "string":
            mock_response[prop_name] = f"mock_{prop_name}"
        elif prop_type == "integer":
            mock_response[prop_name] = 100
        elif prop_type == "number":
            mock_response[prop_name] = 100.50
        elif prop_type == "boolean":
            mock_response[prop_name] = True
        elif prop_type == "array":
            mock_response[prop_name] = []
        elif prop_type == "object":
            mock_response[prop_name] = {}
        else:
            mock_response[prop_name] = None

    return mock_response


# NOTE: Unused - banking_tools.py uses create_tool_stub_function directly with FunctionGroup
# def register_tool_stubs_from_schemas(
#     tool_schemas: list[dict[str, Any]], intent_buffer: ToolIntentBuffer
# ) -> dict[str, FunctionInfo]:
#     """
#     Register tool stubs from a list of tool schemas.
#     
#     Args:
#         tool_schemas: List of tool schemas from the dataset
#         intent_buffer: Shared buffer to record tool intents
#     
#     Returns:
#         Dictionary mapping tool names to FunctionInfo objects
#     """
#     registered_stubs = {}
#
#     for tool_schema in tool_schemas:
#         tool_name = tool_schema.get("title", "")
#         if not tool_name:
#             logger.warning("Skipping tool with no title: %s", tool_schema)
#             continue
#
#         try:
#             stub_fn, custom_input_schema, description = create_tool_stub_function(tool_schema, intent_buffer)
#             
#             # Use FunctionInfo constructor directly with custom input_schema to bypass auto-generation
#             # This prevents NAT from creating its own validation schema from function signatures
#             function_info = FunctionInfo(
#                 single_fn=stub_fn,
#                 stream_fn=None,
#                 input_schema=custom_input_schema,
#                 single_output_schema=str,  # All stubs return strings
#                 stream_output_schema=None,
#                 description=description
#             )
#             registered_stubs[tool_name] = function_info
#             logger.info("Registered tool stub: %s", tool_name)
#         except Exception as e:
#             logger.exception("Failed to register tool stub for %s: %s", tool_name, e)
#             continue
#
#     logger.info("Registered %d tool stubs", len(registered_stubs))
#     return registered_stubs

