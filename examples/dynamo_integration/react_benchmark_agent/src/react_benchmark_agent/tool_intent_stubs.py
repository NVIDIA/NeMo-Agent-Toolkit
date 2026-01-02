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
from pydantic import field_validator

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
    The buffer uses the current scenario ID from the contextvar (set via
    set_current_scenario_id) for both recording and clearing intents.
    """

    def __init__(self) -> None:
        """Initialize a tool intent buffer."""
        self.intents: list[dict[str, Any]] = []

    def record(self, tool_name: str, parameters: dict[str, Any]) -> None:
        """
        Record a tool intent.

        Args:
            tool_name: Name of the tool the agent decided to call
            parameters: Parameters the agent provided for the tool call
        """
        intent = {"tool": tool_name, "parameters": parameters}
        self.intents.append(intent)

        # Store in global registry using contextvar scenario ID for concurrent isolation
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
        """Clear all recorded intents for the current scenario."""
        self.intents.clear()
        # Clear from global registry using contextvar (aligned with record())
        current_scenario = get_current_scenario_id()
        _GLOBAL_INTENT_REGISTRY[current_scenario] = []
        logger.debug("Cleared tool intent buffer for scenario %s", current_scenario)


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


def create_tool_stub_function(tool_schema: dict[str, Any],
                              intent_buffer: ToolIntentBuffer,
                              canned_response: str | None = None) -> tuple[callable, BaseModel | None, str]:
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
