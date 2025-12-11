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

"""
PII Defense Middleware using Microsoft Presidio.

This middleware detects and anonymizes Personally Identifiable Information (PII)
in function outputs using Microsoft Presidio.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from pydantic import Field

from nat.middleware.defense_middleware import DefenseMiddleware
from nat.middleware.defense_middleware import DefenseMiddlewareConfig
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

    llm_name: str | None = Field(
        default=None,
        description="Not used for PII defense (Presidio is rule-based)"
    )
    entities: list[str] = Field(
        default=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN", "LOCATION", "IP_ADDRESS"],
        description="List of PII entities to detect"
    )
    score_threshold: float = Field(
        default=0.01,
        description="Minimum confidence score (0.0-1.0) for PII detection"
    )


class PIIDefenseMiddleware(DefenseMiddleware):
    """PII Defense Middleware using Microsoft Presidio.

    Detects PII in function outputs using Presidio's rule-based entity recognition.
    Only output analysis is currently supported (target_location='output').

    See <https://github.com/microsoft/presidio> for more information about Presidio.
    """

    def __init__(self, config: PIIDefenseMiddlewareConfig, builder):
        super().__init__(config, builder)
        self.config: PIIDefenseMiddlewareConfig = config
        self._analyzer = None
        self._anonymizer = None

        # PII Defense only supports output analysis
        if config.target_location == "input":
            raise ValueError(
                "PIIDefenseMiddleware only supports target_location='output'. "
                "Input analysis is not yet supported."
            )

        logger.info(
            f"PIIDefenseMiddleware initialized: "
            f"action={config.action}, entities={config.entities}, "
            f"score_threshold={config.score_threshold}, target={config.target_function_or_group}"
        )

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
                raise ImportError(
                    "Microsoft Presidio is not installed. "
                    "Install it with: pip install presidio-analyzer presidio-anonymizer"
                ) from err

    def _analyze_text(self, text: str) -> dict:
        """Analyze text for PII entities using Presidio.

        Args:
            text: The text to analyze

        Returns:
            Dictionary with detection results and anonymized text
        """
        self._lazy_load_presidio()
        from presidio_anonymizer.entities import OperatorConfig

        # Analyze for PII with NO score threshold first (to see everything)
        all_results = self._analyzer.analyze(
            text=text,
            entities=self.config.entities,
            language="en"
        )

        # Log ALL detections before filtering
        logger.debug(
            "PII Defense raw detections: %s",
            [(r.entity_type, r.score, text[r.start:r.end]) for r in all_results]
        )

        # Filter by score threshold
        results = [r for r in all_results if r.score >= self.config.score_threshold]

        # Group by entity type
        detected_entities = {}
        for result in results:
            entity_type = result.entity_type
            if entity_type not in detected_entities:
                detected_entities[entity_type] = []
            detected_entities[entity_type].append({
                "text": text[result.start:result.end],
                "score": result.score,
                "start": result.start,
                "end": result.end
            })

        # Generate anonymized version (used when action='sanitize')
        anonymized_text = text
        if results:
            # Use custom replacement operators for each entity type
            operators = {}
            for result in results:
                operators[result.entity_type] = OperatorConfig(
                    "replace",
                    {"new_value": f"<{result.entity_type}>"}
                )

            anonymized_text = self._anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=operators
            ).text

        return {
            "pii_detected": len(results) > 0,
            "entities": detected_entities,
            "anonymized_text": anonymized_text,
            "original_text": text
        }

    def _process_pii_detection(
        self,
        value: Any,
        location: str,
        context: FunctionMiddlewareContext,
    ) -> tuple[Any, bool, str]:
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
            Tuple of (sanitized_value, should_block, entities_str)
            - sanitized_value: The value after PII handling (may be unchanged)
            - should_block: True if refusal action should block (caller should raise error)
            - entities_str: String representation of detected entities (for error messages)
        """
        # Extract field from value if target_field is specified
        content_to_analyze, field_info = self._extract_field_from_value(value)

        # Analyze for PII (convert to string for Presidio)
        content_text = str(content_to_analyze)
        analysis_result = self._analyze_text(content_text)

        if not analysis_result.get("pii_detected", False):
            return value, False, ""

        # PII detected - handle based on action
        entities = analysis_result.get("entities", {})
        entities_str = ", ".join([f"{k}({len(v)})" for k, v in entities.items()])

        if self.config.action == "refusal":
            logger.error("PII Defense refusing %s of %s: %s", location, context.name, entities_str)
            return value, True, entities_str  # Signal to block

        elif self.config.action == "redirection":
            logger.warning("PII Defense detected PII in %s of %s: %s", location, context.name, entities_str)
            logger.info("PII Defense anonymizing %s for %s", location, context.name)
            anonymized_content = analysis_result.get("anonymized_text", content_text)

            # Convert anonymized_text back to original type if needed
            sanitized_value = anonymized_content
            if isinstance(content_to_analyze, (int, float)):
                try:
                    sanitized_value = type(content_to_analyze)(anonymized_content)
                except (ValueError, TypeError):
                    logger.warning(
                        "Could not convert anonymized text '%s' to %s",
                        anonymized_content,
                        type(content_to_analyze).__name__
                    )
                    sanitized_value = anonymized_content

            # If field was extracted, apply sanitized value back to original structure
            if field_info is not None:
                return self._apply_field_result_to_value(value, field_info, sanitized_value), False, entities_str
            else:
                # No field extraction - return sanitized content directly
                return sanitized_value, False, entities_str

        else:  # action == "partial_compliance"
            logger.warning("PII Defense detected PII in %s of %s: %s", location, context.name, entities_str)
            return value, False, entities_str  # No modification, just log

    async def function_middleware_invoke(
        self,
        value: Any,
        call_next: CallNext,
        context: FunctionMiddlewareContext,
    ) -> Any:
        """Intercept function calls to detect and anonymize PII in inputs or outputs.

        Args:
            value: The input value to the function
            call_next: Function to call the next middleware or the actual function
            context: Context containing function metadata

        Returns:
            The function result, with PII anonymized if action='redirection'
        """
        # Check if this defense should apply to this function
        if not self._should_apply_defense(context.name):
            return await call_next(value)

        # Call the actual function
        result = await call_next(value)

        # Handle output analysis (only output is supported)
        sanitized_output, should_block, entities_str = self._process_pii_detection(result, "output", context)
        if should_block:
            raise ValueError(f"PII detected in output: {entities_str}. Output blocked.")
        result = sanitized_output

        return result

    async def function_middleware_stream(
        self,
        value: Any,
        call_next: CallNextStream,
        context: FunctionMiddlewareContext,
    ) -> AsyncIterator[Any]:
        """Intercept streaming calls to detect and anonymize PII in inputs or outputs.

        Note: PII detection requires full text, so we collect all chunks before analyzing.

        Args:
            value: The input value to the function
            call_next: Function to call the next middleware or the actual function
            context: Context containing function metadata

        Yields:
            The function result chunks, with PII anonymized if action='redirection'
        """
        # Check if this defense should apply to this function
        if not self._should_apply_defense(context.name):
            async for chunk in call_next(value):
                yield chunk
            return

        # Collect all chunks for output PII analysis (only output is supported)
        collected_chunks = []
        async for chunk in call_next(value):
            collected_chunks.append(chunk)

        # Handle output analysis
        if collected_chunks:
            # For streaming, we need to reconstruct the full output value
            output_value = "".join(str(chunk) for chunk in collected_chunks)
            sanitized_output, should_block, entities_str = self._process_pii_detection(output_value, "output", context)

            if should_block:
                raise ValueError(f"PII detected in output: {entities_str}. Output blocked.")

            # If sanitized (redirection), yield as single chunk
            if sanitized_output != output_value:
                yield sanitized_output
                return

        # Yield original chunks (no PII detected)
        for chunk in collected_chunks:
            yield chunk

