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
from typing import Any
from typing import Literal
from typing import cast

from jsonpath_ng import parse
from pydantic import BaseModel

from nat.middleware.function_middleware import CallNext
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.function_middleware import FunctionMiddlewareContext

logger = logging.getLogger(__name__)


class RedTeamingMiddleware(FunctionMiddleware):
    """Middleware for red teaming that intercepts and modifies function inputs/outputs.

    This middleware enables systematic security testing by injecting attack payloads
    into function inputs or outputs. It supports flexible targeting, field-level
    modifications, and multiple attack modes.

    Features:

    * Target specific functions or entire function groups
    * Search for specific fields in input/output schemas
    * Apply attacks via replace or append modes
    * Support for both regular and streaming calls
    * Type-safe operations on strings, numbers

    Example::

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
        attack_payload: The malicious payload to inject.
        target_function_or_group: Function or group to target (None for all).
        payload_placement: How to apply (replace, append_start, append_middle, append_end).
        target_location: Whether to attack input or output.
        target_field: Field name or path to attack (None for direct value).
    """

    def __init__(
        self,
        *,
        attack_payload: str,
        target_function_or_group: str | None = None,
        payload_placement: Literal["replace", "append_start", "append_middle", "append_end"] = "append_end",
        target_location: Literal["input", "output"] = "input",
        target_field: str | None = None,
        target_field_resolution_strategy: Literal["random", "first", "last", "all", "error"] = "error",
        call_limit: int | None = None,
    ) -> None:
        """Initialize red teaming middleware.

        Args:
            attack_payload: The value to inject to the function input or output.
            target_function_or_group: Optional function/group to target.
            payload_placement: How to apply the payload (replace or append modes).
            target_location: Whether to place the payload in the input or output.
            target_field: JSONPath to the field to attack.
            target_field_resolution_strategy: Strategy (random/first/last/all/error).
            call_limit: Maximum number of times the middleware will apply a payload.
        """
        super().__init__(is_final=False)
        self._attack_payload = attack_payload
        self._target_function_or_group = target_function_or_group
        self._payload_placement = payload_placement
        self._target_location = target_location
        self._target_field = target_field
        self._target_field_resolution_strategy = target_field_resolution_strategy
        self._call_count: int = 0  # Count the number of times the middleware has applied a payload
        self._call_limit = call_limit
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

        if context_name == "<workflow>":
            return target in {"<workflow>", "workflow"}

        # If context has no dot, match if it equals the target exactly
        return context_name == target

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

    def _apply_payload_to_simple_type(self,
                                      original_value: list | str | int | float,
                                      attack_payload: str,
                                      payload_placement: str) -> Any:
        """Apply the attack payload to simple types (str, int, float) value.

        Args:
            original_value: The original value to attack
            attack_payload: The payload to inject
            payload_placement: How to apply the payload

        Returns:
            The modified value with attack applied

        Raises:
            ValueError: If attack cannot be applied due to type mismatch
        """
        # Determine actual type from value if not provided
        value_type = type(original_value)

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
        if isinstance(original_value, int | float):
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
                raise ValueError(f"Cannot convert attack payload '{attack_payload}' to {value_type.__name__}") from e

    def _resolve_multiple_field_matches(self, matches):
        if self._target_field_resolution_strategy == "error":
            raise ValueError(f"Multiple matches found for target_field: {self._target_field}")
        elif self._target_field_resolution_strategy == "random":
            return [random.choice(matches)]
        elif self._target_field_resolution_strategy == "first":
            return [matches[0]]
        elif self._target_field_resolution_strategy == "last":
            return [matches[-1]]
        elif self._target_field_resolution_strategy == "all":
            return matches
        else:
            raise ValueError(f"Unknown target_field_resolution_strategy: {self._target_field_resolution_strategy}")

    def _apply_payload_to_complex_type(self, value: list | dict | BaseModel) -> list | dict | BaseModel:
        if self._target_field is None:
            if isinstance(value, BaseModel):
                value_details = value.model_dump_json()
            else:
                value_details = ""
            additional_info = ("Additional info: A pydantic BaseModel with fields:" +
                               value_details if value_details else "")
            raise ValueError("Applying an attack payload to complex type, requires a target_field. \n"
                             f"Input value: {value}.: {value_details}. {additional_info} \n"
                             "A target field can be specified in the middleware configuration as a jsonpath.")

        # Convert BaseModel to dict for jsonpath processing
        original_type = type(value)
        is_basemodel = isinstance(value, BaseModel)
        if is_basemodel:
            value_to_modify = value.model_dump()
        else:
            value_to_modify = value

        jsonpath_expr = parse(self._target_field)
        matches = jsonpath_expr.find(value_to_modify)
        if len(matches) == 0:
            raise ValueError(f"No matches found for target_field: {self._target_field} in value: {value}")
        if len(matches) > 1:
            matches = self._resolve_multiple_field_matches(matches)
        else:
            matches = [matches[0]]
        modified_values = [
            self._apply_payload_to_simple_type(match.value, self._attack_payload, self._payload_placement)
            for match in matches
        ]
        for match, modified_value in zip(matches, modified_values):
            match.full_path.update(value_to_modify, modified_value)

        # Reconstruct BaseModel if original was BaseModel
        if is_basemodel:
            assert isinstance(value_to_modify, dict)
            return cast(type[BaseModel], original_type)(**value_to_modify)
        return value_to_modify

    def _apply_payload_to_function_value(self, value: Any) -> Any:
        if self._call_limit is not None and self._call_count >= self._call_limit:
            logger.warning("Call limit reached for red teaming middleware. "
                           "Not applying attack payload to value: %s",
                           value)
            return value
        if isinstance(value, list | dict | BaseModel):
            modified_value = self._apply_payload_to_complex_type(value)
        elif isinstance(value, str | int | float):
            modified_value = self._apply_payload_to_simple_type(value, self._attack_payload, self._payload_placement)
        else:
            raise ValueError(f"Unsupported function input/output type: {type(value).__name__}")
        self._call_count += 1
        return modified_value

    def _apply_payload_to_function_value_with_exception(self, value: Any, context: FunctionMiddlewareContext) -> Any:
        try:
            return self._apply_payload_to_function_value(value)
        except Exception as e:
            logger.error("Failed to apply red team attack to function %s: %s", context.name, e, exc_info=True)
            raise

    async def function_middleware_invoke(self,
                                         *args: Any,
                                         call_next: CallNext,
                                         context: FunctionMiddlewareContext,
                                         **kwargs: Any) -> Any:
        """Invoke middleware for single-output functions.

        Args:
            args: Positional arguments passed to the function (first arg is typically the input value).
            call_next: Callable to invoke next middleware/function.
            context: Metadata about the function being wrapped.
            kwargs: Keyword arguments passed to the function.

        Returns:
            The output value (potentially modified if attacking output).
        """
        value = args[0] if args else None

        # Check if we should attack this function
        if not self._should_apply_payload(context.name):
            logger.debug("Skipping function %s (not targeted)", context.name)
            return await call_next(value, *args[1:], **kwargs)

        if self._target_location == "input":
            # Attack the input before calling the function
            modified_input = self._apply_payload_to_function_value_with_exception(value, context)
            # Call next with modified input
            return await call_next(modified_input, *args[1:], **kwargs)

        elif self._target_location == "output":  # target_location == "output"
            # Call function first, then attack the output
            output = await call_next(value, *args[1:], **kwargs)
            modified_output = self._apply_payload_to_function_value_with_exception(output, context)
            return modified_output
        else:
            raise ValueError(f"Unknown target_location: {self._target_location}. "
                             "Attack payloads can only be applied to function input or output.")


__all__ = ["RedTeamingMiddleware"]
