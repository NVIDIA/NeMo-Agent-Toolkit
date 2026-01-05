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
PII Defense Middleware using Microsoft Presidio.

This middleware detects and anonymizes Personally Identifiable Information (PII)
in function outputs using Microsoft Presidio.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from pydantic import Field

from nat.middleware.defense.defense_middleware import DefenseMiddleware
from nat.middleware.defense.defense_middleware import DefenseMiddlewareConfig
from nat.middleware.defense.defense_middleware_data_models import PIIAnalysisResult
from nat.middleware.function_middleware import CallNext
from nat.middleware.function_middleware import CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext

logger = logging.getLogger(__name__)


class PIIDefenseMiddlewareConfig(DefenseMiddlewareConfig, name="pii_defense"):
    """Configuration for PII Defense Middleware using Microsoft Presidio.

    Detects PII in function outputs using Presidio's rule-based entity recognition (no LLM required).

    See <https://github.com/microsoft/presidio> for more information about Presidio.

    Actions:
    - 'partial_compliance': Detect and log PII, but allow content to pass through
    - 'refusal': Block content if PII detected (hard stop)
    - 'redirection': Replace PII with anonymized placeholders (e.g., <EMAIL_ADDRESS>)

    Note: Only output analysis is currently supported (target_location='output').
    """

    llm_name: str | None = Field(default=None, description="Not used for PII defense (Presidio is rule-based)")
    entities: list[str] = Field(default_factory=lambda: [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "US_SSN",
        "LOCATION",
        "IP_ADDRESS", ],
                                description="List of PII entities to detect")
    score_threshold: float = Field(default=0.01, description="Minimum confidence score (0.0-1.0) for PII detection")


class PIIDefenseMiddleware(DefenseMiddleware):
    """PII Defense Middleware using Microsoft Presidio.

    Detects PII in function outputs using Presidio's rule-based entity recognition.
    Only output analysis is currently supported (``target_location='output'``).

    See https://github.com/microsoft/presidio for more information about Presidio.

    Streaming Behavior:
        For 'refusal' and 'redirection' actions, chunks are buffered and checked
        before yielding to prevent PII from being streamed to clients.
        For 'partial_compliance' action, chunks are yielded immediately; violations
        are logged but content passes through.
    """

    def __init__(self, config: PIIDefenseMiddlewareConfig, builder):
        super().__init__(config, builder)
        self.config: PIIDefenseMiddlewareConfig = config
        self._analyzer = None
        self._anonymizer = None

        # PII Defense only supports output analysis
        if config.target_location == "input":
            raise ValueError("PIIDefenseMiddleware only supports target_location='output'. "
                             "Input analysis is not yet supported.")

        logger.info(f"PIIDefenseMiddleware initialized: "
                    f"action={config.action}, entities={config.entities}, "
                    f"score_threshold={config.score_threshold}, target={config.target_function_or_group}")

    def _lazy_load_presidio(self):
        """Lazy load Presidio components when first needed."""
        if self._analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine

                self._analyzer = AnalyzerEngine()
                self._anonymizer = AnonymizerEngine()
                logger.info("Presidio engines loaded successfully")
            except ImportError as err:
                raise ImportError("Microsoft Presidio is not installed. "
                                  "Install it with: pip install presidio-analyzer presidio-anonymizer") from err

    def _analyze_content(self, text: str) -> PIIAnalysisResult:
        """Analyze content for PII entities using Presidio.

        Args:
            text: The text to analyze

        Returns:
            PIIAnalysisResult with detection results and anonymized text.
        """
        self._lazy_load_presidio()
        from presidio_anonymizer.entities import OperatorConfig

        # Analyze for PII with NO score threshold first (to see everything)
        all_results = self._analyzer.analyze(text=text, entities=self.config.entities, language="en")

        # Log ALL detections before filtering (without PII text for privacy)
        logger.debug("PII Defense raw detections: %s", [(r.entity_type, r.score, r.start, r.end) for r in all_results])

        # Filter by score threshold
        results = [r for r in all_results if r.score >= self.config.score_threshold]

        # Group by entity type (without PII text for privacy)
        detected_entities = {}
        for result in results:
            entity_type = result.entity_type
            if entity_type not in detected_entities:
                detected_entities[entity_type] = []
            detected_entities[entity_type].append({"score": result.score, "start": result.start, "end": result.end})

        # Generate anonymized version (used when action='sanitize')
        anonymized_text = text
        if results:
            # Use custom replacement operators for each entity type
            operators = {}
            for result in results:
                operators[result.entity_type] = OperatorConfig("replace", {"new_value": f"<{result.entity_type}>"})

            anonymized_text = self._anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text

        return PIIAnalysisResult(pii_detected=len(results) > 0,
                                 entities=detected_entities,
                                 anonymized_text=anonymized_text,
                                 original_text=text)

    def _process_pii_detection(
        self,
        value: Any,
        location: str,
        context: FunctionMiddlewareContext,
    ) -> Any:
        """Process PII detection and sanitization for a given value.

        This is a common helper method that handles:
        - Field extraction (if target_field is specified)
        - PII analysis
        - Action handling (refusal, redirection, partial_compliance)
        - Applying sanitized value back to original structure

        Args:
            value: The value to analyze (input or output)
            location: Either "input" or "output" (for logging)
            context: Function context metadata

        Returns:
            The value after PII handling (may be unchanged, sanitized, or raise exception)
        """
        # Extract field from value if target_field is specified
        content_to_analyze, field_info = self._extract_field_from_value(value)

        logger.info("PIIDefenseMiddleware: Checking %s %s for %s",
                    f"field '{self.config.target_field}'" if field_info else "entire",
                    location,
                    context.name)
        # Analyze for PII (convert to string for Presidio)
        content_text = str(content_to_analyze)
        analysis_result = self._analyze_content(content_text)

        if not analysis_result.pii_detected:
            logger.info("PIIDefenseMiddleware: Verified %s of %s: No PII detected", location, context.name)
            return value

        # PII detected - handle based on action
        entities = analysis_result.entities
        # Build entities string efficiently without intermediate list
        entities_str = ", ".join(f"{k}({len(v)})" for k, v in entities.items())
        sanitized_content = self._handle_threat(content_to_analyze, analysis_result, context, location, entities_str)

        # If field was extracted, apply sanitized value back to original structure
        if field_info is not None:
            return self._apply_field_result_to_value(value, field_info, sanitized_content)
        else:
            # No field extraction - return sanitized content directly
            return sanitized_content

    def _handle_threat(
        self,
        content: Any,
        analysis_result: PIIAnalysisResult,
        context: FunctionMiddlewareContext,
        location: str,
        entities_str: str,
    ) -> Any:
        """Handle detected PII threat based on configured action.

        Args:
            content: The content with PII
            analysis_result: Detection result from Presidio
            context: Function context
            location: Either "input" or "output" (for logging)
            entities_str: String representation of detected entities

        Returns:
            Handled content (anonymized, original, or raises exception for refusal)
        """
        if self.config.action == "refusal":
            logger.error("PII Defense refusing %s of %s: %s", location, context.name, entities_str)
            raise ValueError(f"PII detected in {location}: {entities_str}. Output refused.")

        elif self.config.action == "redirection":
            logger.warning("PII Defense detected PII in %s of %s: %s", location, context.name, entities_str)
            logger.info("PII Defense anonymizing %s for %s", location, context.name)
            str(content)
            anonymized_content = analysis_result.anonymized_text

            # Convert anonymized_text back to original type if needed
            redirected_value = anonymized_content
            if isinstance(content, int | float):
                try:
                    redirected_value = type(content)(anonymized_content)
                except (ValueError, TypeError):
                    logger.warning("Could not convert anonymized text '%s' to %s",
                                   anonymized_content,
                                   type(content).__name__)
                    redirected_value = anonymized_content

            return redirected_value

        else:  # action == "partial_compliance"
            logger.warning("PII Defense detected PII in %s of %s: %s", location, context.name, entities_str)
            return content  # No modification, just log

    async def function_middleware_invoke(
        self,
        *args: Any,
        call_next: CallNext,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> Any:
        """Intercept function calls to detect and anonymize PII in inputs or outputs.

        Args:
            args: Positional arguments passed to the function (first arg is typically the input value).
            call_next: Function to call the next middleware or the actual function.
            context: Context containing function metadata.
            kwargs: Keyword arguments passed to the function.

        Returns:
            The function result, with PII anonymized if action='redirection'.
        """
        value = args[0] if args else None

        # Check if this defense should apply to this function
        if not self._should_apply_defense(context.name):
            logger.debug("PIIDefenseMiddleware: Skipping %s (not targeted)", context.name)
            return await call_next(value, *args[1:], **kwargs)

        try:
            # Call the actual function
            result = await call_next(value, *args[1:], **kwargs)

            # Handle output analysis (only output is supported)
            result = self._process_pii_detection(result, "output", context)

            return result

        except Exception:
            logger.error(
                "Failed to apply PII defense to function %s",
                context.name,
                exc_info=True,
            )
            raise

    async def function_middleware_stream(
        self,
        *args: Any,
        call_next: CallNextStream,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Intercept streaming calls to detect and anonymize PII in inputs or outputs.

        For 'refusal' and 'redirection' actions: Chunks are buffered and checked before yielding.
        For 'partial_compliance' action: Chunks are yielded immediately; violations are logged.

        Args:
            args: Positional arguments passed to the function (first arg is typically the input value).
            call_next: Function to call the next middleware or the actual function.
            context: Context containing function metadata.
            kwargs: Keyword arguments passed to the function.

        Yields:
            The function result chunks, with PII anonymized if action='redirection'.
        """
        value = args[0] if args else None

        # Check if this defense should apply to this function
        if not self._should_apply_defense(context.name):
            logger.debug("PIIDefenseMiddleware: Skipping %s (not targeted)", context.name)
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

            # Analyze the full output for PII
            full_output = "".join(chunk if isinstance(chunk, str) else str(chunk) for chunk in accumulated_chunks)
            processed_output = self._process_pii_detection(full_output, "output", context)

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
                "Failed to apply PII defense to streaming function %s",
                context.name,
                exc_info=True,
            )
            raise
