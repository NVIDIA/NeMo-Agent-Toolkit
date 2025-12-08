# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PII Defense Middleware using Microsoft Presidio.

This middleware detects and anonymizes Personally Identifiable Information (PII)
in function outputs using Microsoft Presidio.
"""

import logging
from typing import Any, AsyncIterator, List, Optional

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from pydantic import Field

from nat.middleware.defense_middleware import DefenseMiddleware, DefenseMiddlewareConfig
from nat.middleware.function_middleware import CallNext, CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext

logger = logging.getLogger(__name__)


class PIIDefenseMiddlewareConfig(DefenseMiddlewareConfig, name="pii_defense"):
    """Configuration for PII Defense Middleware using Microsoft Presidio.

    Detects PII in function outputs. Actions: 'log', 'block', or 'sanitize'.
    Uses Presidio's rule-based entity recognition (no LLM required).
    """

    llm_name: Optional[str] = Field(
        default=None,
        description="Not used for PII defense (Presidio is rule-based)"
    )
    entities: List[str] = Field(
        default=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN", "LOCATION", "IP_ADDRESS"],
        description="List of PII entities to detect"
    )
    score_threshold: float = Field(
        default=0.01,
        description="Minimum confidence score (0.0-1.0) for PII detection"
    )


class PIIDefenseMiddleware(DefenseMiddleware):
    """PII Defense Middleware using Microsoft Presidio.

    Detects PII in function outputs and takes action:
    - 'block': Raises an error if PII is detected
    - 'sanitize': Replaces PII with placeholders (e.g., <EMAIL_ADDRESS>)
    """

    def __init__(self, config: PIIDefenseMiddlewareConfig, builder):
        super().__init__(config, builder)
        self.config: PIIDefenseMiddlewareConfig = config
        self._analyzer = None
        self._anonymizer = None

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
            except ImportError:
                raise ImportError(
                    "Microsoft Presidio is not installed. "
                    "Install it with: pip install presidio-analyzer presidio-anonymizer"
                )

    def _analyze_text(self, text: str) -> dict:
        """Analyze text for PII entities using Presidio.

        Args:
            text: The text to analyze

        Returns:
            Dictionary with detection results and anonymized text
        """
        self._lazy_load_presidio()

        # Analyze for PII with NO score threshold first (to see everything)
        all_results = self._analyzer.analyze(
            text=text,
            entities=self.config.entities,
            language="en"
        )
        
        # Log ALL detections before filtering
        logger.info(f"PII Defense raw detections: {[(r.entity_type, r.score, text[r.start:r.end]) for r in all_results]}")
        
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


    async def function_middleware_invoke(
        self,
        value: Any,
        call_next: CallNext,
        context: FunctionMiddlewareContext,
    ) -> Any:
        """Intercept function calls to detect and anonymize PII in outputs.

        Args:
            value: The input value to the function
            call_next: Function to call the next middleware or the actual function
            context: Context containing function metadata

        Returns:
            The function result, with PII anonymized if action='sanitize'
        """
        # Check if this defense should apply to this function
        if not self._should_apply_defense(context.name):
            return await call_next(value)

        # Call the actual function
        result = await call_next(value)

        # Analyze output for PII
        output_text = str(result)
        analysis_result = self._analyze_text(output_text)

        if analysis_result["pii_detected"]:
            entities_str = ", ".join([f"{k}({len(v)})" for k, v in analysis_result["entities"].items()])
            
            if self.config.action == "block":
                logger.error(f"PII Defense blocking output of {context.name}: {entities_str}")
                raise ValueError(f"PII detected in output: {entities_str}. Output blocked.")
            elif self.config.action == "sanitize":
                logger.warning(f"PII Defense detected PII in output of {context.name}: {entities_str}")
                logger.info(f"PII Defense anonymizing output for {context.name}")
                result = analysis_result["anonymized_text"]
            else:  # action == "log"
                logger.warning(f"PII Defense detected PII in output of {context.name}: {entities_str}")
                # No modification, just log

        return result

    async def function_middleware_stream(
        self,
        value: Any,
        call_next: CallNextStream,
        context: FunctionMiddlewareContext,
    ) -> AsyncIterator[Any]:
        """Intercept streaming calls to detect and anonymize PII in outputs.

        Note: PII detection requires full text, so we collect all chunks before analyzing.

        Args:
            value: The input value to the function
            call_next: Function to call the next middleware or the actual function
            context: Context containing function metadata

        Yields:
            The function result chunks, with PII anonymized if action='sanitize'
        """
        # Check if this defense should apply to this function
        if not self._should_apply_defense(context.name):
            async for chunk in call_next(value):
                yield chunk
            return

        # Collect all chunks for PII analysis
        collected_chunks = []
        async for chunk in call_next(value):
            collected_chunks.append(chunk)

        if collected_chunks:
            output_text = "".join(str(chunk) for chunk in collected_chunks)
            analysis_result = self._analyze_text(output_text)

            if analysis_result["pii_detected"]:
                entities_str = ", ".join([f"{k}({len(v)})" for k, v in analysis_result["entities"].items()])
                
                if self.config.action == "block":
                    logger.error(f"PII Defense blocking output of {context.name}: {entities_str}")
                    raise ValueError(f"PII detected in output: {entities_str}. Output blocked.")
                elif self.config.action == "sanitize":
                    logger.warning(f"PII Defense detected PII in output of {context.name}: {entities_str}")
                    logger.info(f"PII Defense anonymizing output for {context.name}")
                    # Yield the anonymized text as a single chunk
                    yield analysis_result["anonymized_text"]
                    return
                else:  # action == "log"
                    logger.warning(f"PII Defense detected PII in output of {context.name}: {entities_str}")
                    # No modification, just log

        # Yield original chunks (no PII detected)
        for chunk in collected_chunks:
            yield chunk

