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
Content Safety Guard Middleware.

This middleware uses guard models to classify content as safe or harmful
with simple Yes/No answers.
"""

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from pydantic import Field

from nat.middleware.defense.defense_middleware import DefenseMiddleware
from nat.middleware.defense.defense_middleware import DefenseMiddlewareConfig
from nat.middleware.defense.defense_middleware_data_models import ContentAnalysisResult
from nat.middleware.defense.defense_middleware_data_models import GuardResponseResult
from nat.middleware.function_middleware import CallNext
from nat.middleware.function_middleware import CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext

logger = logging.getLogger(__name__)


class ContentSafetyGuardMiddlewareConfig(DefenseMiddlewareConfig, name="content_safety_guard"):
    """Configuration for Content Safety Guard middleware.

    This middleware uses guard models to classify content as safe or harmful.

    Actions: partial_compliance (log warning but allow), refusal (block content),
    or redirection (replace with polite refusal message).

    Note: Only output analysis is currently supported (target_location='output').
    """

    llm_name: str = Field(description="Name of the guard model LLM (must be defined in llms section)")


class ContentSafetyGuardMiddleware(DefenseMiddleware):
    """Safety guard middleware using guard models to classify content as safe or unsafe.

    This middleware analyzes content using guard models (e.g., NVIDIA Nemoguard, Qwen Guard)
    that return "Safe" or "Unsafe" classifications. The middleware extracts safety categories
    when unsafe content is detected.

    Only output analysis is currently supported (``target_location='output'``).

    Streaming Behavior:
        For 'refusal' and 'redirection' actions, chunks are buffered and checked
        before yielding to prevent unsafe content from being streamed to clients.
        For 'partial_compliance' action, chunks are yielded immediately; violations
        are logged but content passes through.
    """

    def __init__(self, config: ContentSafetyGuardMiddlewareConfig, builder):
        """Initialize content safety guard middleware.

        Args:
            config: Configuration for content safety guard middleware
            builder: Builder instance for loading LLMs
        """
        super().__init__(config, builder)
        # Store config with correct type for linter
        self.config: ContentSafetyGuardMiddlewareConfig = config
        self._llm = None  # Lazy loaded LLM

        # Content Safety Guard only supports output analysis
        if config.target_location == "input":
            raise ValueError("ContentSafetyGuardMiddleware only supports target_location='output'. "
                             "Input analysis is not yet supported.")

    async def _get_llm(self):
        """Lazy load the guard model LLM when first needed."""
        if self._llm is None:
            self._llm = await self._get_llm_for_defense(self.config.llm_name)
        return self._llm

    def _extract_unsafe_categories(self, response_text: str, is_safe: bool) -> list[str]:
        """Extract safety categories only if content is unsafe.

        Supports both JSON formats (Safety Categories field) and text formats
        (Categories: line).

        Args:
            response_text: Raw response from guard model.
            is_safe: Whether the content was detected as safe.

        Returns:
            List of category strings if unsafe, empty list otherwise or on parsing error.
        """
        if is_safe:
            return []

        try:
            categories = []

            # Try parsing as JSON first (for Nemoguard)
            try:
                json_data = json.loads(response_text)
                # Look for common category field names
                category_field = None
                for field in ["Safety Categories", "Categories", "Category", "safety_categories", "categories"]:
                    if field in json_data:
                        category_field = json_data[field]
                        break

                if category_field:
                    if isinstance(category_field, str):
                        # Split by comma if it's a comma-separated string
                        categories = [cat.strip() for cat in category_field.split(",")]
                    elif isinstance(category_field, list):
                        categories = [str(cat).strip() for cat in category_field]
            except (json.JSONDecodeError, ValueError, AttributeError):
                # Not JSON, try text parsing (for Qwen Guard)
                # Look for "Categories:" or "Category:" followed by text
                category_patterns = [
                    r'Categories?:\s*([^\n]+)',  # Categories: Violent
                    r'Categories?\s*=\s*([^\n]+)',  # Categories = Violent
                    r'"Safety Categories":\s*"([^"]+)"',  # JSON-like in text
                ]

                for pattern in category_patterns:
                    match = re.search(pattern, response_text, re.IGNORECASE)
                    if match:
                        category_text = match.group(1).strip()
                        # Split by comma if comma-separated
                        categories = [cat.strip() for cat in category_text.split(",")]
                        break

            return categories
        except Exception:
            # If any error occurs during category extraction, return empty list
            logger.debug("Failed to extract categories from guard response, returning empty list")
            return []

    def _parse_guard_response(self, response_text: str) -> GuardResponseResult:
        """Parse guard model response.

        Searches for Safe or Unsafe keywords anywhere in the response (case-insensitive).
        Works with any guard model format (JSON, structured text, or plain text).
        Also extracts safety categories from both JSON and text formats.
        If neither keyword is found, falls back to implicit refusal detection.
        Prioritizes Unsafe if both keywords are present.

        Args:
            response_text: Raw response from guard model.

        Returns:
            GuardResponseResult with is_safe boolean, categories list, and raw response.
        """
        cleaned_text = re.sub(r'[*_]+', '', response_text).strip()
        response_lower = cleaned_text.lower()

        # Search for "Unsafe" or "Safe" anywhere in the response (case-insensitive)
        # Prioritize "Unsafe" if both are present
        if re.search(r'\bunsafe\b', response_lower):
            is_safe = False
        elif re.search(r'\bsafe\b', response_lower):
            is_safe = True
        else:
            # Detect implicit refusals (model refuses = harmful content detected)
            refusal_indicators = [
                "i cannot",
                "i can't",
                "i won't",
                "i will not",
                "i'm unable",
                "i am unable",
                "i refuse",
                "cannot provide",
                "can't provide",
                "won't provide",
                "illegal",
                "harmful",
                "dangerous",
                "inappropriate"
            ]

            if any(indicator in response_lower for indicator in refusal_indicators):
                is_safe = False  # Model refused = harmful detected
            else:
                is_safe = True  # Default to safe if unclear

        # Extract categories only if unsafe
        categories = self._extract_unsafe_categories(response_text, is_safe)

        return GuardResponseResult(is_safe=is_safe, categories=categories, raw_response=response_text)

    def _should_refuse(self, parsed_result: GuardResponseResult) -> bool:
        """Determine if content should be refused.

        Args:
            parsed_result: Result from _parse_guard_response.

        Returns:
            True if content should be refused.
        """
        return not parsed_result.is_safe

    async def _analyze_content(self,
                               content: Any,
                               original_input: Any = None,
                               context: FunctionMiddlewareContext | None = None) -> ContentAnalysisResult:
        """Check content safety using guard model.

        Args:
            content: The content to analyze
            original_input: The original input to the function (for context)
            context: Function metadata

        Returns:
            Safety classification result with should_refuse flag
        """
        try:
            # Get the guard model LLM
            llm = await self._get_llm()

            content_str = str(content)

            # Call the guard model using messages format to ensure chat template is applied
            # Format matches: messages = [{"role": "user", "content": prompt}]
            messages = [{"role": "user", "content": content_str}]
            response = await llm.ainvoke(messages)

            # Extract text from response
            if hasattr(response, 'content'):
                response_text = response.content.strip()
            elif isinstance(response, str):
                response_text = response.strip()
            else:
                response_text = str(response).strip()
            # Parse the guard model response

            parsed = self._parse_guard_response(response_text)
            should_refuse = self._should_refuse(parsed)

            return ContentAnalysisResult(is_safe=parsed.is_safe,
                                         categories=parsed.categories,
                                         raw_response=parsed.raw_response,
                                         should_refuse=should_refuse,
                                         error=False,
                                         error_message=None)

        except Exception as e:
            logger.exception("Content Safety Guard analysis failed: %s", e)
            return ContentAnalysisResult(is_safe=True,
                                         categories=[],
                                         raw_response="",
                                         should_refuse=False,
                                         error=True,
                                         error_message=str(e))

    async def _handle_threat(self,
                             content: Any,
                             analysis_result: ContentAnalysisResult,
                             context: FunctionMiddlewareContext) -> Any:
        """Handle unsafe content based on configured action.

        Args:
            content: The unsafe content
            analysis_result: Safety classification result.
            context: Function context

        Returns:
            Handled content (blocked, sanitized, or original)
        """
        action = self.config.action

        categories = analysis_result.categories
        logger.warning("Content Safety Guard detected unsafe content in %s (categories: %s)",
                       context.name,
                       ", ".join(categories) if categories else "none")

        if action == "refusal":
            logger.error("Content Safety Guard refusing output of %s", context.name)
            raise ValueError("Content blocked by safety policy")

        elif action == "redirection":
            return "I'm sorry, I cannot help you with that request."

        else:  # action == "partial_compliance"
            logger.warning("Safety violation logged but content passed through")
            return content

    async def _process_content_safety_detection(
        self,
        value: Any,
        location: str,
        context: FunctionMiddlewareContext,
        original_input: Any = None,
    ) -> Any:
        """Process content safety detection and handling for a given value.

        Handles field extraction, content safety analysis, threat handling,
        and applying sanitized value back to original structure.

        Args:
            value: The value to analyze (input or output).
            location: Either input or output (for logging).
            context: Function context metadata.
            original_input: Original function input (for output analysis context).

        Returns:
            The value after content safety handling (may be unchanged, sanitized, or raise).
        """
        # Extract field from value if target_field is specified
        content_to_analyze, field_info = self._extract_field_from_value(value)

        logger.info("ContentSafetyGuardMiddleware: Checking %s %s for %s",
                    f"field '{self.config.target_field}'" if field_info else "entire",
                    location,
                    context.name)
        analysis_result = await self._analyze_content(content_to_analyze,
                                                      original_input=original_input,
                                                      context=context)

        if not analysis_result.should_refuse:
            # Content is safe, return original value
            logger.info("ContentSafetyGuardMiddleware: Verified %s of %s as safe", location, context.name)
            return value

        # Unsafe content detected - handle based on action
        logger.warning("ContentSafetyGuardMiddleware: Blocking %s for %s (unsafe content detected)",
                       location,
                       context.name)
        sanitized_content = await self._handle_threat(content_to_analyze, analysis_result, context)

        # If field was extracted, apply sanitized value back to original structure
        if field_info is not None:
            return self._apply_field_result_to_value(value, field_info, sanitized_content)
        else:
            # No field extraction - return sanitized content directly
            return sanitized_content

    async def function_middleware_invoke(self,
                                         *args: Any,
                                         call_next: CallNext,
                                         context: FunctionMiddlewareContext,
                                         **kwargs: Any) -> Any:
        """Apply content safety guard check to function invocation.

        This is the core logic for content safety guard defense - each defense implements
        its own invoke/stream based on its specific strategy.

        Args:
            args: Positional arguments passed to the function (first arg is typically the input value).
            call_next: Next middleware/function to call.
            context: Function metadata (provides context state).
            kwargs: Keyword arguments passed to the function.

        Returns:
            Function output (potentially blocked or sanitized).
        """
        value = args[0] if args else None

        # Check if defense should apply to this function
        if not self._should_apply_defense(context.name):
            logger.debug("ContentSafetyGuardMiddleware: Skipping %s (not targeted)", context.name)
            return await call_next(value, *args[1:], **kwargs)

        try:
            # Call the function
            output = await call_next(value, *args[1:], **kwargs)

            # Handle output analysis (only output is supported)
            output = await self._process_content_safety_detection(output, "output", context, original_input=value)

            return output

        except Exception as e:
            logger.error("Failed to apply content safety guard to function %s: %s", context.name, e, exc_info=True)
            raise

    async def function_middleware_stream(self,
                                         *args: Any,
                                         call_next: CallNextStream,
                                         context: FunctionMiddlewareContext,
                                         **kwargs: Any) -> AsyncIterator[Any]:
        """Apply content safety guard check to streaming function.

        For 'refusal' and 'redirection' actions: Chunks are buffered and checked before yielding.
        For 'partial_compliance' action: Chunks are yielded immediately; violations are logged.

        Args:
            args: Positional arguments passed to the function (first arg is typically the input value).
            call_next: Next middleware/function to call.
            context: Function metadata.
            kwargs: Keyword arguments passed to the function.

        Yields:
            Function output chunks (potentially blocked or sanitized).
        """
        value = args[0] if args else None

        # Check if defense should apply to this function
        if not self._should_apply_defense(context.name):
            logger.debug("ContentSafetyGuardMiddleware: Skipping %s (not targeted)", context.name)
            async for chunk in call_next(value, *args[1:], **kwargs):
                yield chunk
            return

        try:
            buffer_chunks = self.config.action in ("refusal", "redirection")
            accumulated_chunks: list[Any] = []

            async for chunk in call_next(value, *args[1:], **kwargs):
                if buffer_chunks:
                    accumulated_chunks.append(chunk)
                else:
                    # partial_compliance: stream through, but still accumulate for analysis/logging
                    yield chunk
                    accumulated_chunks.append(chunk)

            # Join chunks efficiently (only convert to string if needed)
            full_output = "".join(chunk if isinstance(chunk, str) else str(chunk) for chunk in accumulated_chunks)

            processed_output = await self._process_content_safety_detection(full_output,
                                                                            "output",
                                                                            context,
                                                                            original_input=value)

            processed_str = str(processed_output)
            if self.config.action == "redirection" and processed_str != full_output:
                # Redirected: yield replacement once (and stop).
                yield processed_output
                return

            if buffer_chunks:
                # refusal: would have raised; safe content: preserve chunking
                for chunk in accumulated_chunks:
                    yield chunk

        except Exception:
            logger.error(
                "Failed to apply content safety guard to streaming function %s",
                context.name,
                exc_info=True,
            )
            raise
