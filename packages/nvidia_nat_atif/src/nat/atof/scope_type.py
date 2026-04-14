# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scope type alias for ATOF events.

In ATOF v0.2, ``scope_type`` is an open-vocabulary non-empty string. The
``ScopeType`` name is retained as a type alias for documentation purposes
only; it does NOT enumerate valid values. Common conventions (spec §3.1):
agent, function, tool, llm, retriever, embedder, reranker, guardrail,
evaluator, custom, unknown.

See ATOF spec Section 3.1.
"""

from __future__ import annotations

# Documentation-only type alias. Any non-empty string is a valid scope_type.
ScopeType = str
