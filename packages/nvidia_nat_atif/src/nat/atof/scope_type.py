# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scope type vocabulary for ATOF events (spec §4).

ATOF v0.1 uses a CLOSED vocabulary for ``scope_type`` with a ``custom`` +
``profile.subtype`` escape hatch for vendor extensions and an ``unknown``
value for tier-1 pass-through producers that cannot classify a scope.

The ``ScopeType`` name is retained as a ``Literal`` type alias so that typed
Python consumers can annotate expected values; validator logic in ``events.py``
tolerates any non-empty string on the wire (consumers MUST NOT reject unknown
``scope_type`` values per spec §4.3), but producers using the canonical
vocabulary through ``ScopeType`` get static-analysis coverage.

Canonical vocabulary:

- ``agent``      — top-level agent or workflow scope
- ``function``   — generic function or application step
- ``llm``        — LLM call scope (populates ``profile.model_name`` + optional ``schema``)
- ``tool``       — tool invocation scope (populates ``profile.tool_call_id`` + optional ``schema``)
- ``retriever``  — retrieval step (document search, index lookup)
- ``embedder``   — embedding-generation step
- ``reranker``   — result reranking step
- ``guardrail``  — guardrail or validation step
- ``evaluator``  — evaluation or scoring step
- ``custom``     — vendor-defined category; REQUIRES ``profile.subtype`` to name it
- ``unknown``    — producer does not know or cannot classify the scope
"""

from __future__ import annotations

from typing import Literal

ScopeType = Literal[
    "agent",
    "function",
    "llm",
    "tool",
    "retriever",
    "embedder",
    "reranker",
    "guardrail",
    "evaluator",
    "custom",
    "unknown",
]
