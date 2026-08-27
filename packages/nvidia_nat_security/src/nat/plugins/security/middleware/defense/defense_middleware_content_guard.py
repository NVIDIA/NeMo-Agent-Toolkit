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

This middleware uses guard models to classify content as safe, unsafe, or
controversial.
"""

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from pydantic import Field

from nat.middleware.common import TargetLocation
from nat.middleware.function_middleware import CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.middleware import InvocationContext
from nat.plugins.security.middleware.defense.defense_middleware import DefenseMiddleware
from nat.plugins.security.middleware.defense.defense_middleware import DefenseMiddlewareConfig
from nat.plugins.security.middleware.defense.defense_middleware_data_models import ContentAnalysisResult
from nat.plugins.security.middleware.defense.defense_middleware_data_models import GuardResponseResult

logger = logging.getLogger(__name__)


class ContentSafetyGuardMiddlewareConfig(DefenseMiddlewareConfig, name="content_safety_guard"):
    """Configuration for Content Safety Guard middleware.

    This middleware uses guard models to classify content as safe or harmful.

    Actions: partial_compliance (log warning but allow), refusal (block content),
    or redirection (replace with polite refusal message).
    """

    llm_name: str = Field(description="Name of the guard model LLM (must be defined in llms section)")

    max_content_length: int = Field(
        default=32000,
        gt=0,
        description="Maximum number of characters sent to the guard model. Content exceeding this limit stops "
        "protected execution instead of being truncated or sent to the model.")


class ContentSafetyGuardMiddleware(DefenseMiddleware):
    """Safety guard middleware using guard models to classify content as safe or unsafe.

    This middleware analyzes content using guard models (e.g., NVIDIA Nemoguard, Qwen Guard)
    that return "Safe", "Unsafe", or "Controversial" classifications. The middleware extracts
    safety categories when unsafe content is detected.

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

        Accepts exact Safe, Unsafe, or Controversial verdicts from supported JSON,
        Qwen Guard text, or plain-text response formats. Also extracts safety
        categories from JSON and text formats. Treats Controversial, malformed,
        and unrecognized responses as unsafe.

        Args:
            response_text: Raw response from guard model.

        Returns:
            GuardResponseResult with is_safe boolean, categories list, and raw response.
        """
        verdict = self._extract_guard_verdict(response_text)
        is_safe = verdict == "safe"

        # Extract categories only if unsafe
        categories = self._extract_unsafe_categories(response_text, is_safe)

        return GuardResponseResult(is_safe=is_safe, categories=categories, raw_response=response_text)

    @staticmethod
    def _extract_guard_verdict(response_text: str) -> str | None:
        """Extract an exact verdict from a supported guard response format.

        Args:
            response_text: Raw guard response.

        Returns:
            A lowercase ``safe``, ``unsafe``, or ``controversial`` verdict when recognized; otherwise ``None``.
        """

        def reject_duplicate_keys(pairs):
            json_object = {}
            for key, value in pairs:
                if key in json_object:
                    raise ValueError(f"Duplicate JSON field: {key}")
                json_object[key] = value
            return json_object

        try:
            json_data = json.loads(response_text.strip(), object_pairs_hook=reject_duplicate_keys)
        except (json.JSONDecodeError, TypeError, ValueError):
            json_data = None

        if isinstance(json_data, dict):
            verdict_fields = ("User Safety", "Safety", "user_safety", "safety")
            candidates = [str(json_data[field]).strip().lower() for field in verdict_fields if field in json_data]
            valid_verdicts = {"safe", "unsafe", "controversial"}
            if not candidates or any(candidate not in valid_verdicts for candidate in candidates):
                return None
            return candidates[0] if len(set(candidates)) == 1 else None

        response_lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        first_line = response_lines[0].strip("*_").strip() if response_lines else ""
        match = re.fullmatch(r'(?:Safety\s*:\s*)?(Safe|Unsafe|Controversial)', first_line, re.IGNORECASE)
        if match is None:
            return None

        for line in response_lines[1:]:
            cleaned_line = line.strip("*_").strip()
            if re.fullmatch(r'(?:Safety\s*:\s*)?(Safe|Unsafe|Controversial)', cleaned_line, re.IGNORECASE):
                return None
            if re.match(r'Safety\s*:', cleaned_line, re.IGNORECASE):
                return None

        return match.group(1).lower()

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

        Raises:
            ValueError: If content exceeds ``max_content_length``.
            Exception: Propagates guard loading and invocation failures so protected execution stops.
        """
        content_str = str(content)
        if len(content_str) > self.config.max_content_length:
            raise ValueError(f"Content Safety Guard input length {len(content_str)} exceeds configured "
                             f"max_content_length={self.config.max_content_length}")

        llm = await self._get_llm()

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

        parsed = self._parse_guard_response(response_text)
        should_refuse = self._should_refuse(parsed)

        return ContentAnalysisResult(is_safe=parsed.is_safe,
                                     categories=parsed.categories,
                                     raw_response=parsed.raw_response,
                                     should_refuse=should_refuse,
                                     error=False,
                                     error_message=None)

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
            logger.error("Content Safety Guard refusing function output of %s", context.name)
            raise ValueError("Content blocked by safety policy")

        elif action == "redirection":
            return "I'm sorry, I cannot help you with that request."

        else:  # action == "partial_compliance"
            logger.warning("Safety violation logged but content passed through")
            return content

    async def _process_content_safety_detection(
        self,
        value: Any,
        context: FunctionMiddlewareContext,
        original_input: Any = None,
    ) -> Any:
        """Process content safety detection and handling for a given value.

        Handles field extraction, content safety analysis, threat handling,
        and applying sanitized value back to original structure.

        Args:
            value: The value to analyze.
            context: Function context metadata.
            original_input: Original function input (for output analysis context).

        Returns:
            The value after content safety handling (may be unchanged, sanitized, or raise).
        """
        # Extract field from value if target_field is specified
        content_to_analyze, field_info = self._extract_field_from_value(value)

        logger.info("ContentSafetyGuardMiddleware: Checking %s function output for %s",
                    f"field '{self.config.target_field}' of" if field_info else "entire",
                    context.name)
        analysis_result = await self._analyze_content(content_to_analyze,
                                                      original_input=original_input,
                                                      context=context)

        if not analysis_result.should_refuse:
            # Content is safe, return original value
            logger.info("ContentSafetyGuardMiddleware: %s function output verified as safe", context.name)
            return value

        # Unsafe content detected - handle based on action
        logger.warning("ContentSafetyGuardMiddleware: Blocking %s function output (unsafe content detected)",
                       context.name)
        sanitized_content = await self._handle_threat(content_to_analyze, analysis_result, context)

        # If field was extracted, apply sanitized value back to original structure
        if field_info is not None:
            return self._apply_field_result_to_value(value, field_info, sanitized_content)
        else:
            # No field extraction - return sanitized content directly
            return sanitized_content

    async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Analyze function output for content safety after execution.

        Args:
            context: Invocation context with function metadata and output.

        Returns:
            Modified context if output was processed, None to pass through.
        """
        if self.config.target_location != TargetLocation.OUTPUT:
            return None

        # Check if defense should apply to this function
        func_ctx: FunctionMiddlewareContext = context.function_context
        if not self._should_apply_defense(func_ctx.name):
            logger.debug("ContentSafetyGuardMiddleware: Skipping %s (not targeted)", func_ctx.name)
            return None

        try:
            # Handle function output analysis
            original_input = context.original_args[0] if context.original_args else None
            context.output = await self._process_content_safety_detection(context.output,
                                                                          func_ctx,
                                                                          original_input=original_input)
            return context
        except Exception as error:
            logger.error("Failed to apply content safety guard to function %s (%s); protected execution stopped",
                         func_ctx.name,
                         type(error).__name__)
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

            processed_output = await self._process_content_safety_detection(full_output, context, original_input=value)

            processed_str = str(processed_output)
            if self.config.action == "redirection" and processed_str != full_output:
                # Redirected: yield replacement once (and stop).
                yield processed_output
                return

            if buffer_chunks:
                # refusal: would have raised; safe content: preserve chunking
                for chunk in accumulated_chunks:
                    yield chunk

        except Exception as error:
            logger.error(
                "Failed to apply content safety guard to streaming function %s (%s); protected execution stopped",
                context.name,
                type(error).__name__,
            )
            raise
