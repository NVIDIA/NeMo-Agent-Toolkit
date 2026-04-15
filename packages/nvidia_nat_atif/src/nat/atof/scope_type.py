# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scope type enumeration for ATOF events.

Mirrors NeMo-Flow's ``ScopeType`` enum (``crates/core/src/types/scope.rs``)
without importing it. Serializes as lowercase strings.
"""

from __future__ import annotations

from enum import Enum


class ScopeType(str, Enum):
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
