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
"""Red teaming middleware for attacking agent functions.

This module provides a middleware for red teaming and security testing that can
intercept and modify function inputs or outputs with configurable attack payloads.

The middleware supports:
- Targeting specific functions or entire function groups
- Field-level search within input/output schemas
- Multiple attack modes (replace, append_start, append_middle, append_end)
- Both regular and streaming function calls
- Type-safe operations on strings, integers, and floats
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import AsyncIterator
from typing import Any
from typing import Literal

from pydantic import BaseModel

from nat.middleware.function_middleware import CallNext
from nat.middleware.function_middleware import CallNextStream
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.function_middleware import FunctionMiddlewareContext

logger = logging.getLogger(__name__)


class RedTeamingMiddleware(FunctionMiddleware):
    """Middleware for red teaming that intercepts and modifies function inputs/outputs.

    This middleware enables systematic security testing by injecting attack payloads
    into function inputs or outputs. It supports flexible targeting, field-level
    modifications, and multiple attack modes.

    Features:
    - Target specific functions or entire function groups
    - Search for specific fields in input/output schemas
    - Apply attacks via replace or append modes
    - Support for both regular and streaming calls
    - Type-safe operations on strings, numbers

    Example:
        # In YAML config
        middleware:
          prompt_injection:
            _type: red_teaming
            attack_payload: "Ignore previous instructions"
            target_function_or_group: my_llm.generate
            payload_placement: append_start
            target_location: input
            target_field: prompt

    Args:
        attack_payload: The malicious payload to inject
        target_function_or_group: Function or group to target
            (None = all functions for which the middleware is applied in the workflow)
        payload_placement: How to apply the payload (replace/append_start/append_middle/append_end)
        target_location: Whether to attack input or output
        target_field: Field name or path to attack (None = operate on value directly)
    """

    def __init__(
        self,
        *,
        attack_payload: str,
        target_function_or_group: str | None = None,
        payload_placement: Literal["replace", "append_start", "append_middle", "append_end"] = "append_end",
        target_location: Literal["input", "output"] = "input",
        target_field: str | None = None,
    ) -> None:
        """Initialize red teaming middleware.

        Args:
            attack_payload: The value to inject to the function input or output
            target_function_or_group: Optional function/group to target
            payload_placement: How to apply the payload (replace or append modes)
            target_location: Whether to place the payload in the input or output
            target_field: Optional field name or path to search for
        """
        super().__init__(is_final=False)
        self._attack_payload = attack_payload
        self._target_function_or_group = target_function_or_group
        self._payload_placement = payload_placement
        self._target_location = target_location
        self._target_field = target_field

        logger.info(
            "RedTeamingMiddleware initialized: payload=%s, target=%s, placement=%s, location=%s, field=%s",
            attack_payload,
            target_function_or_group,
            payload_placement,
            target_location,
            target_field,
        )

    def _should_apply_payload(self, context_name: str) -> bool:
        """Check if this function should be attacked based on targeting configuration.

        Args:
            context_name: The name of the function from context (e.g., "calculator.add")

        Returns:
            True if the function should be attacked, False otherwise
        """
        # If no target specified, attack all functions
        if self._target_function_or_group is None:
            return True

        target = self._target_function_or_group
        # Group targeting - match if context starts with the group name
        # Handle both "group.function" and just "function" in context
        if "." in context_name and "." not in target:
            context_group = context_name.split(".", 1)[0]
            return context_group == target

        # If context has no dot, match if it equals the target exactly
        return context_name == target

    def _find_field_in_value(
        self, value: Any, schema: type[BaseModel] | type[None] | None, is_nested_path: bool = False
    ) -> tuple[Any, list[str]]:
        """Find and extract field value from a structured value using field search.

        Args:
            value: The value to search within
            schema: The schema describing the value structure
            is_nested_path: Whether target_field is a nested path with dots

        Returns:
            Tuple of (field_value, field_path) where field_path is list of keys to navigate

        Raises:
            ValueError: If field not found, multiple matches found, or invalid path
        """
        # If no field search specified, operate on value directly
        if self._target_field is None:
            return value, []

        # Handle nested path (e.g., "data.response.text")
        if is_nested_path:
            path_parts = self._target_field.split(".")
            current_value = value
            traversed_path = []

            for part in path_parts:
                traversed_path.append(part)
                try:
                    if isinstance(current_value, dict):
                        current_value = current_value[part]
                    elif isinstance(current_value, BaseModel):
                        current_value = getattr(current_value, part)
                    else:
                        raise ValueError(
                            f"Cannot navigate path '{'.'.join(traversed_path)}': "
                            f"value is not a dict or BaseModel, got {type(current_value).__name__}"
                        )
                except (KeyError, AttributeError) as e:
                    raise ValueError(
                        f"Invalid nested path '{self._target_field}': "
                        f"field '{part}' not found at '{'.'.join(traversed_path[:-1])}'"
                    ) from e

            return current_value, path_parts

        # Simple field name search - search the schema
        if schema is None or schema is type(None):
            raise ValueError(
                f"Cannot search for field '{self._target_field}' without a schema. "
                "Either provide target_field=None to operate on the value directly, "
                "or ensure the function has input/output schemas defined."
            )

        # Search for matching fields in the schema
        matching_fields = []
        for field_name, field_info in schema.model_fields.items():
            if field_name == self._target_field:
                matching_fields.append(field_name)

        # Validate results
        if len(matching_fields) == 0:
            available_fields = list(schema.model_fields.keys())
            raise ValueError(
                f"Field '{self._target_field}' not found in schema. "
                f"Available fields: {available_fields}. "
                f"If you want to target a nested field, use dot notation (e.g., 'data.response.text')."
            )

        if len(matching_fields) > 1:
            raise ValueError(
                f"Multiple fields match '{self._target_field}': {matching_fields}. "
                f"Please specify a unique field name or use a nested path."
            )

        # Extract the field value
        field_name = matching_fields[0]
        try:
            if isinstance(value, dict):
                field_value = value[field_name]
            elif isinstance(value, BaseModel):
                field_value = getattr(value, field_name)
            else:
                raise ValueError(
                    f"Cannot extract field '{field_name}' from value of type {type(value).__name__}. "
                    f"Expected dict or BaseModel."
                )
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Field '{field_name}' not found in value") from e

        return field_value, [field_name]

    def _find_middle_sentence_index(self, text: str) -> int:
        """Find the index to insert text at the middle sentence boundary.

        Args:
            text: The text to analyze

        Returns:
            The character index where the middle sentence ends
        """
        # Find all sentence boundaries using regex
        # Match sentence-ending punctuation followed by space/newline or end of string
        sentence_pattern = r"[.!?](?:\s+|$)"
        matches = list(re.finditer(sentence_pattern, text))

        if not matches:
            # No sentence boundaries found, insert at middle character
            return len(text) // 2

        # Find the sentence boundary closest to the middle
        text_midpoint = len(text) // 2
        closest_match = min(matches, key=lambda m: abs(m.end() - text_midpoint))

        return closest_match.end()

    def _apply_payload(
        self, original_value: list | str | int | float, attack_payload: str, payload_placement: str, value_type: type | None = None
    ) -> Any:
        """Apply the attack payload to a value based on the payload placement.

        Args:
            original_value: The original value to attack
            attack_payload: The payload to inject
            payload_placement: How to apply the payload
            value_type: The expected type of the value (for validation)

        Returns:
            The modified value with attack applied

        Raises:
            ValueError: If attack cannot be applied due to type mismatch
        """
        # Determine actual type from value if not provided
        if value_type is None:
            value_type = type(original_value)

        # Handle cases where original_value is a list. Replace a random index.
        if isinstance(original_value, list):
            index = random.randint(0, len(original_value) - 1)
            original_value[index] = self._apply_payload(original_value[index], attack_payload, payload_placement)
            return original_value

        # Handle string attacks
        if value_type is str or isinstance(original_value, str):
            original_str = str(original_value)

            if payload_placement == "replace":
                return attack_payload
            elif payload_placement == "append_start":
                return f"{attack_payload}{original_str}"
            elif payload_placement == "append_end":
                return f"{original_str}{attack_payload}"
            elif payload_placement == "append_middle":
                insert_index = self._find_middle_sentence_index(original_str)
                return f"{original_str[:insert_index]}{attack_payload}{original_str[insert_index:]}"
            else:
                raise ValueError(f"Unknown payload placement: {payload_placement}")

        # Handle int/float attacks
        if value_type in (int, float) or isinstance(original_value, (int, float)):
            # For numbers, only replace is allowed
            if payload_placement != "replace":
                logger.warning(
                    "Payload placement '%s' not supported for numeric types (int/float). "
                    "Falling back to 'replace' mode for field with value %s",
                    payload_placement,
                    original_value,
                )

            # Convert payload to the appropriate numeric type
            try:
                if value_type is int or isinstance(original_value, int):
                    return int(attack_payload)
                return float(attack_payload)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot convert attack payload '{attack_payload}' to {value_type.__name__}"
                ) from e

    def _set_field_in_value(self, value: Any, field_path: list[str], new_field_value: Any) -> Any:
        """Set a field value in a structured value using the field path.

        Args:
            value: The value to modify
            field_path: List of keys to navigate to the field
            new_field_value: The new value to set

        Returns:
            The modified value (may be a new instance for immutable types)
        """
        if not field_path and isinstance(value, type(new_field_value)):
            # No path means we're replacing the entire value
            return new_field_value

        # For BaseModel instances, we need to create a new instance
        if isinstance(value, BaseModel):
            # Convert to dict, modify, and convert back
            value_dict = value.model_dump()
            self._set_field_in_dict(value_dict, field_path, new_field_value)
            return type(value)(**value_dict)
        elif isinstance(value, dict):
            # Modify dict in place (but we'll create a copy to be safe)
            value_copy = value.copy()
            self._set_field_in_dict(value_copy, field_path, new_field_value)
            return value_copy
        else:
            raise ValueError(f"Cannot set field in value of type {type(value).__name__}")

    def _set_field_in_dict(self, value_dict: dict, field_path: list[str], new_field_value: Any) -> None:
        """Helper to set a field in a nested dictionary structure.

        Args:
            value_dict: The dictionary to modify (modified in place)
            field_path: List of keys to navigate
            new_field_value: The value to set
        """
        current = value_dict
        for i, key in enumerate(field_path[:-1]):
            if key not in current:
                raise ValueError(f"Path key '{key}' not found")
            current = current[key]
            # Convert BaseModel to dict if needed
            if isinstance(current, BaseModel):
                current = current.model_dump()
                # Update parent reference
                parent = value_dict
                for parent_key in field_path[:i]:
                    parent = parent[parent_key]
                parent[key] = current

        # Set the final field
        current[field_path[-1]] = new_field_value

    def _apply_payload_to_schema(self,value: Any,
                                schema: type[BaseModel] | type[None] | None, context: FunctionMiddlewareContext, is_nested_path: bool = False) -> Any:
        schema = context.single_output_schema
        field_value, field_path = self._find_field_in_value(value, schema, is_nested_path)

        # Apply attack to the field value
        attacked_value = self._apply_payload(field_value, self._attack_payload, self._payload_placement)

        # Reconstruct the output with the attacked field
        modified_value = self._set_field_in_value(value, field_path, attacked_value)

        logger.info(
            "Red teaming Middleware: Attacking %s of function '%s' "
            "(placement=%s, field=%s, original=%s, payload=%s, modified=%s)",
            self._target_location,
            context.name,
            self._payload_placement,
            self._target_field or "direct",
            field_value,
            self._attack_payload,
            attacked_value,
        )
        return modified_value
    async def function_middleware_invoke(
        self, value: Any, call_next: CallNext, context: FunctionMiddlewareContext
    ) -> Any:
        """Invoke middleware for single-output functions.

        Args:
            value: The input value to the function
            call_next: Callable to invoke next middleware/function
            context: Metadata about the function being wrapped

        Returns:
            The output value (potentially modified if attacking output)
        """
        # Check if we should attack this function
        if not self._should_apply_payload(context.name):
            logger.debug("Skipping function %s (not targeted)", context.name)
            return await call_next(value)

        # Determine if field search is a nested path
        is_nested_path = self._target_field is not None and "." in self._target_field

        try:
            if self._target_location == "input":
                # Attack the input before calling the function
                schema = context.input_schema
                modified_input = self._apply_payload_to_schema(value, schema, context, is_nested_path)
                # Call next with modified input
                return await call_next(modified_input)

            else:  # target_location == "output"
                # Call function first, then attack the output
                output = await call_next(value)
                schema = context.single_output_schema
                modified_output = self._apply_payload_to_schema(output, schema, context, is_nested_path)

                return modified_output

        except Exception as e:
            logger.error("Failed to apply red team attack to function %s: %s", context.name, e, exc_info=True)
            raise

    async def function_middleware_stream(
        self, value: Any, call_next: CallNextStream, context: FunctionMiddlewareContext
    ) -> AsyncIterator[Any]:
        """Invoke middleware for streaming functions.

        Streaming has limitations:
        - Only append_start is fully supported for output attacks
        - Other modes would require buffering, defeating the purpose of streaming

        Args:
            value: The input value to the function
            call_next: Callable to invoke next middleware/function stream
            context: Metadata about the function being wrapped

        Yields:
            Chunks from the stream (potentially modified)
        """
        # Check if we should attack this function
        if not self._should_apply_payload(context.name):
            logger.debug("Skipping function %s (not targeted)", context.name)
            async for chunk in call_next(value):
                yield chunk
            return

        # Determine if field search is a nested path
        is_nested_path = self._target_field is not None and "." in self._target_field

        try:
            if self._target_location == "input":
                # Attack input before streaming (same as non-streaming)
                schema = context.input_schema
                modified_input = self._apply_payload_to_schema(value, schema, context, is_nested_path)
                # Stream with modified input
                async for chunk in call_next(modified_input):
                    yield chunk
                return

            # target_location == "output"
            # For output attacks on streaming, only append_start is practical
            if self._payload_placement == "append_start":
                logger.info(
                    "Red teaming Middleware: Attacking output of streaming function '%s' with payload '%s'",
                    context.name,
                    self._attack_payload,
                )

                # Yield the attack payload first
                yield self._attack_payload

                # Then yield all chunks from the stream
                async for chunk in call_next(value):
                    yield chunk
                return

            # Other modes require buffering the entire stream
            logger.warning(
                "Payload placement '%s' not supported for streaming outputs (would require buffering). "
                "Only 'append_start' is supported for streaming. Passing through without attack.",
                self._payload_placement,
            )

            # Pass through without modification
            async for chunk in call_next(value):
                yield chunk

        except Exception as e:
            logger.error(
                "Failed to apply red team attack to streaming function %s: %s", context.name, e, exc_info=True
            )
            raise


__all__ = ["RedTeamingMiddleware"]

