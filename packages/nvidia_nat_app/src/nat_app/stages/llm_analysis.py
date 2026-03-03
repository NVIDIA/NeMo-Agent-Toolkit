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
"""LLMAnalysisStage: detect LLM call sites per node using the adapter's LLMDetector."""

from __future__ import annotations

import logging
from typing import Any

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.graph.adapter import AbstractFrameworkAdapter
from nat_app.graph.llm_detection import LLMCallInfo
from nat_app.graph.llm_detection import count_llm_calls

logger = logging.getLogger(__name__)


class LLMAnalysisStage:
    """Count LLM invocation sites per node for downstream priority assignment.

    Reads: ``node_funcs`` (from ``NodeAnalysisStage``)
    Writes: ``llm_analysis`` — ``dict[str, LLMCallInfo]``

    When the adapter's ``get_llm_detector``
    returns ``None``, writes an empty dict and returns immediately (no-op).
    """

    def __init__(self, adapter: AbstractFrameworkAdapter) -> None:
        self._adapter = adapter

    @property
    def name(self) -> str:
        return "llm_analysis"

    def apply(self, context: CompilationContext, **kwargs: Any) -> CompilationContext:
        """Count LLM invocation sites per node for priority assignment.

        Args:
            context: Current compilation context with ``node_funcs`` in metadata.
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            The updated context with ``llm_analysis`` (dict of node name to
            LLMCallInfo) in metadata. Empty dict if no LLM detector available.
        """
        detector = self._adapter.get_llm_detector()

        if detector is None:
            context.metadata["llm_analysis"] = {}
            return context

        node_funcs: dict[str, Any] = context.metadata.get("node_funcs", {})
        results: dict[str, LLMCallInfo] = {}

        for node_name, func in node_funcs.items():
            if not callable(func):
                results[node_name] = LLMCallInfo()
                continue
            results[node_name] = count_llm_calls(func, detector)

        total_llm_nodes = sum(1 for r in results.values() if r.call_count > 0)
        total_calls = sum(r.call_count for r in results.values())
        logger.info(
            "LLM analysis: %d/%d nodes have LLM calls (%d total call sites)",
            total_llm_nodes,
            len(results),
            total_calls,
        )

        context.metadata["llm_analysis"] = results
        return context
