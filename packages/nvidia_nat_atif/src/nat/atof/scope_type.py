# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scope type enumeration for ATOF events.

Serializes as lowercase strings. See ATOF spec Section 3.1.
"""

from __future__ import annotations

from enum import StrEnum


class ScopeType(StrEnum):
    """Semantic scope type for ScopeStart/ScopeEnd events."""

    AGENT = "agent"
    FUNCTION = "function"
    TOOL = "tool"
    LLM = "llm"
    RETRIEVER = "retriever"
    EMBEDDER = "embedder"
    RERANKER = "reranker"
    GUARDRAIL = "guardrail"
    EVALUATOR = "evaluator"
    CUSTOM = "custom"
    UNKNOWN = "unknown"
