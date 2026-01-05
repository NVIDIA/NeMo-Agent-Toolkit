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
Base Defense Middleware.

This module provides a utility base class for defense middleware with common
configuration and helper methods. Each defense middleware implements its own
core logic based on its specific defense strategy (LLM-based, rule-based, etc.).
"""

import logging
import secrets
from typing import Any
from typing import Literal
from typing import cast

from jsonpath_ng import parse
from pydantic import BaseModel
from pydantic import Field

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.middleware import FunctionMiddlewareBaseConfig
from nat.middleware.function_middleware import FunctionMiddleware

logger = logging.getLogger(__name__)


class MultipleTargetFieldMatchesError(ValueError):
    """Raised when a JSONPath matches multiple fields and strategy='error'."""

    def __init__(self, target_field: str | None) -> None:
        super().__init__(f"Multiple matches found for target_field={target_field!r}")


class UnknownTargetFieldResolutionStrategyError(ValueError):
    """Raised when an unknown target_field_resolution_strategy is configured."""

    def __init__(self, strategy: str) -> None:
        super().__init__(f"Unknown target_field_resolution_strategy={strategy!r}")


class DefenseMiddlewareConfig(FunctionMiddlewareBaseConfig):
    """Base configuration for defense middleware.

    Actions use safety domain terminology:
    - 'partial_compliance': Comply with user request with warning (monitoring mode)
    - 'refusal': Refuse user request (hard refusal)
    - 'redirection': Redirect user request to a safe place; provide a safer response
    """

    action: Literal["partial_compliance", "refusal", "redirection"] = Field(
        default="partial_compliance",
        description=("Action to take when threat detected. "
                     "Options: 'partial_compliance' (log with warning), 'refusal' (block), "
                     "'redirection' (sanitize/replace with safe content)"))

    llm_wrapper_type: LLMFrameworkEnum | str = Field(
        default=LLMFrameworkEnum.LANGCHAIN,
        description="Framework wrapper type for LLM (langchain, llama_index, crewai, etc.). "
        "Only needed for LLM-based defenses.")

    target_function_or_group: str | None = Field(
        default=None,
        description="Optional function or function group to target. "
        "If None, defense applies to all functions. "
        "Examples: 'my_calculator', 'my_calculator.divide', 'llm_agent.generate'")

    target_location: Literal["output"] = Field(
        default="output",
        description=("Whether to analyze function input or output. "
                     "Currently only 'output' is supported (analyze after function call). "
                     "Input analysis is not yet supported."))

    target_field: str | None = Field(
        default=None,
        description=(
            "Optional JSONPath expression to target specific fields within complex types (dict/list/BaseModel). "
            "If None and value is complex type, defense applies to entire value. "
            "If None and value is simple type (str/int/float), defense applies directly. "
            "Examples: '$.result', '[0]', '$.data.message', 'numbers[0]'"))

    target_field_resolution_strategy: Literal["error", "first", "last", "random", "all"] = Field(
        default="error",
        description=("Strategy for handling multiple JSONPath matches when target_field is specified. "
                     "Options: 'error' (raise error if multiple matches), 'first' (use first match), "
                     "'last' (use last match), 'random' (use random match), 'all' (analyze all matches)"))


class DefenseMiddleware(FunctionMiddleware):
    """Utility base class for defense middleware.

    This base class provides:

    * Common configuration fields (action, check_input, check_output, llm_wrapper_type)
    * Helper methods for LLM loading (for LLM-based defenses)
    * Access to builder for any resources needed

    Unlike an abstract base class, this does NOT enforce a specific pattern.
    Each defense middleware implements its own invoke/stream logic based on
    its specific defense strategy:

    * LLM-based analysis (guard models, verifiers)
    * Rule-based detection (regex, signatures)
    * Heuristic-based checks
    * Statistical anomaly detection
    * etc.

    Each defense owns its core logic, just like red_teaming_middleware does.

    LLM Wrapper Types:
        The ``llm_wrapper_type`` config field supports different framework wrappers:
        langchain (default) for LangChain/LangGraph-based workflows,
        llama_index for LlamaIndex-based workflows, crewai for CrewAI-based
        workflows, semantic_kernel for Semantic Kernel-based workflows, and
        agno, adk, strands for other supported frameworks.
    """

    def __init__(self, config: DefenseMiddlewareConfig, builder):
        """Initialize defense middleware.

        Args:
            config: Configuration for the defense middleware
            builder: Builder instance for loading LLMs and other resources
        """
        super().__init__(is_final=False)
        self.config = config
        self.builder = builder

        logger.info(f"{self.__class__.__name__} initialized: "
                    f"action={config.action}, target={config.target_function_or_group}")

    def _should_apply_defense(self, context_name: str) -> bool:
        """Check if defense should be applied to this function based on targeting configuration.

        This method mirrors the targeting logic from RedTeamingMiddleware to provide
        consistent behavior between attack and defense middleware.

        Args:
            context_name: The name of the function from context (e.g., "calculator.add").
                For workflow-level middleware, this will be "<workflow>"

        Returns:
            True if defense should be applied, False otherwise

        Examples:
            - target=None → defends all functions and workflow
            - target="my_calculator" → defends all functions in my_calculator group
            - target="my_calculator.divide" → defends only the divide function
            - target="<workflow>" or "workflow" → defends only at workflow level
        """
        # If no target specified, defend all functions
        if self.config.target_function_or_group is None:
            return True

        target = self.config.target_function_or_group

        # Group targeting - match if context starts with the group name
        # Handle both "group.function" and just "function" in context
        if "." in context_name and "." not in target:
            context_group = context_name.split(".", 1)[0]
            return context_group == target

        if context_name == "<workflow>":
            return target in {"<workflow>", "workflow"}

        # Exact match for specific function or group
        return context_name == target

    async def _get_llm_for_defense(self, llm_name: str, wrapper_type: LLMFrameworkEnum | str | None = None):
        """Helper to lazy load an LLM for defense purposes.

        This is a utility method for LLM-based defenses. Not all defenses
        will use this - some may use rule-based or other detection methods.

        Args:
            llm_name: Name of the LLM to load
            wrapper_type: Framework wrapper type (defaults to config.llm_wrapper_type if not specified)

        Returns:
            The loaded LLM instance with the specified framework wrapper
        """
        if wrapper_type is None:
            wrapper_type = self.config.llm_wrapper_type

        return await self.builder.get_llm(llm_name, wrapper_type=wrapper_type)

    def _resolve_multiple_field_matches(self, matches):
        """Resolve multiple JSONPath matches based on resolution strategy.

        Args:
            matches: List of JSONPath match objects

        Returns:
            List of matches based on resolution strategy
        """
        strategy = self.config.target_field_resolution_strategy

        if strategy == "error":
            raise MultipleTargetFieldMatchesError(self.config.target_field)
        elif strategy == "first":
            return [matches[0]]
        elif strategy == "last":
            return [matches[-1]]
        elif strategy == "random":
            return [secrets.choice(matches)]
        elif strategy == "all":
            return matches
        else:
            raise UnknownTargetFieldResolutionStrategyError(strategy)

    def _extract_field_from_value(self, value: Any) -> tuple[Any, dict | None]:
        """Extract field(s) from value using JSONPath if target_field is specified.

        Args:
            value: The value to extract fields from (can be simple or complex type).

        Returns:
            A tuple of (content_to_analyze, field_info_dict) where content_to_analyze
            is the extracted field value(s) or original value if no targeting, and
            field_info_dict contains target_field, matches, and original_value if
            field was extracted, or None otherwise.
        """
        # If no target_field specified, analyze entire value
        if self.config.target_field is None:
            return value, None

        # If value is simple type, target_field doesn't apply (analyze entire value)
        if isinstance(value, str | int | float | bool):
            logger.debug(
                "target_field '%s' specified but value is simple type (%s). "
                "Analyzing entire value instead.",
                self.config.target_field,
                type(value).__name__)
            return value, None

        # For complex types, extract field using JSONPath
        if not isinstance(value, dict | list | BaseModel):
            logger.warning(
                "target_field '%s' specified but value type '%s' is not supported for field extraction. "
                "Analyzing entire value instead.",
                self.config.target_field,
                type(value).__name__)
            return value, None

        # Convert BaseModel to dict for JSONPath processing
        original_type = type(value)
        is_basemodel = isinstance(value, BaseModel)
        if is_basemodel:
            value_dict = value.model_dump()
        else:
            value_dict = value

        # Parse JSONPath and find matches
        try:
            jsonpath_expr = parse(self.config.target_field)
            matches = jsonpath_expr.find(value_dict)

            if len(matches) == 0:
                logger.warning("No matches found for target_field '%s' in value. Analyzing entire value instead.",
                               self.config.target_field)
                return value, None

            # Resolve multiple matches based on strategy
            if len(matches) > 1:
                matches = self._resolve_multiple_field_matches(matches)

            # Extract field values
            if len(matches) == 1:
                # Single match - return the value directly
                extracted_value = matches[0].value
            else:
                # Multiple matches (strategy="all") - return list of values
                extracted_value = [match.value for match in matches]

            field_info = {
                "target_field": self.config.target_field,
                "matches": matches,
                "original_value": value,
                "is_basemodel": is_basemodel,
                "original_type": original_type
            }

            logger.debug("Extracted field '%s' from value: %s -> %s", self.config.target_field, value, extracted_value)

            return extracted_value, field_info

        except Exception as e:  # noqa: BLE001 - jsonpath-ng may raise multiple exception types; fallback is intentional.
            logger.warning("Failed to extract field '%s' from value: %s. Analyzing entire value instead.",
                           self.config.target_field,
                           e)
            return value, None

    def _apply_field_result_to_value(self, original_value: Any, field_info: dict, analysis_result: Any) -> Any:
        """Apply analysis result back to original value if field was extracted.

        This is used when defense needs to modify the value based on field analysis.
        For example, if analyzing $.result and need to replace it with sanitized value.

        Args:
            original_value: The original complex value
            field_info: Field info dict from _extract_field_from_value (None if no field extraction)
            analysis_result: The result from defense analysis (could be sanitized value)

        Returns:
            Modified value with field updated, or original value if no field extraction
        """
        if field_info is None:
            # No field extraction - return analysis result directly
            return analysis_result

        # Reconstruct value with updated field
        matches = field_info["matches"]
        is_basemodel = field_info["is_basemodel"]
        original_type = field_info["original_type"]

        # Get the dict representation
        if is_basemodel:
            value_dict = original_value.model_dump()
        # Create a copy to avoid modifying original
        elif isinstance(original_value, dict):
            value_dict = original_value.copy()
        elif isinstance(original_value, list):
            value_dict = list(original_value)
        else:
            value_dict = original_value

        # Update field(s) with analysis result
        if len(matches) == 1:
            # Single match - update single field
            matches[0].full_path.update(value_dict, analysis_result)
        # Multiple matches - update all fields (analysis_result should be a list)
        elif isinstance(analysis_result, list) and len(analysis_result) == len(matches):
            for match, result_value in zip(matches, analysis_result, strict=True):
                match.full_path.update(value_dict, result_value)
        else:
            logger.warning("Cannot apply analysis result to multiple fields: "
                           "expected list of %d values, got %s",
                           len(matches),
                           type(analysis_result).__name__)
            return original_value

        # Reconstruct BaseModel if original was BaseModel
        if is_basemodel:
            assert isinstance(value_dict, dict)
            return cast(type[BaseModel], original_type)(**value_dict)

        return value_dict


__all__ = ["DefenseMiddleware", "DefenseMiddlewareConfig"]
