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
"""Data models for defense middleware output."""

from typing import Any

from pydantic import BaseModel


class PIIAnalysisResult(BaseModel):
    """Result of PII analysis using Presidio.

    Attributes:
        pii_detected: Whether PII was detected in the analyzed text.
        entities: Dictionary mapping entity types to lists of detection metadata (score, start, end).
        anonymized_text: Text with PII replaced by entity type placeholders (e.g., <EMAIL_ADDRESS>).
        original_text: The unmodified original text that was analyzed.
    """

    pii_detected: bool
    entities: dict[str, list[dict[str, Any]]]
    anonymized_text: str
    original_text: str


class GuardResponseResult(BaseModel):
    """Result of parsing guard model response.

    Attributes:
        is_safe: Whether the content is classified as safe by the guard model.
        categories: List of unsafe content categories detected (empty if safe).
        raw_response: The unprocessed response text from the guard model.
    """

    is_safe: bool
    categories: list[str]
    raw_response: str


class ContentAnalysisResult(BaseModel):
    """Result of content safety analysis with guard models.

    Attributes:
        is_safe: Whether the content is classified as safe by the guard model.
        categories: List of unsafe content categories detected (empty if safe).
        raw_response: The unprocessed response text from the guard model.
        should_refuse: Whether the content should be refused based on the analysis.
        error: Whether an error occurred during analysis.
        error_message: Error message if error occurred, otherwise None.
    """

    is_safe: bool
    categories: list[str]
    raw_response: str
    should_refuse: bool
    error: bool = False
    error_message: str | None = None


class OutputVerificationResult(BaseModel):
    """Result of output verification using LLM.

    Attributes:
        threat_detected: Whether a threat (incorrect or manipulated output) was detected.
        confidence: Confidence score (0.0-1.0) in the threat detection.
        reason: Explanation for the detection result.
        correct_answer: The correct output value if threat detected, otherwise None.
        content_type: Type of content analyzed ('input' or 'output').
        should_refuse: Whether the content should be refused based on threshold.
        error: Whether an error occurred during verification.
    """

    threat_detected: bool
    confidence: float
    reason: str
    correct_answer: Any | None
    content_type: str
    should_refuse: bool
    error: bool = False
