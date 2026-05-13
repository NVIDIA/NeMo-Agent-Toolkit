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
"""EdgeClassificationStage: classify edges and find parallel groups."""

from __future__ import annotations

import logging
from typing import Any

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.graph.analysis import build_dependency_graph
from nat_app.graph.analysis import find_parallel_groups
from nat_app.graph.scheduling import classify_edges

logger = logging.getLogger(__name__)


class EdgeClassificationStage:
    """Classify edges as necessary/unnecessary and find parallel groups.

    Reads: ``graph``, ``node_analyses``, ``reducer_fields``
    Writes: ``edge_analyses``, ``necessary_edges``, ``unnecessary_edges``, ``parallel_groups``
    """

    @property
    def name(self) -> str:
        return "edge_classification"

    def apply(self, context: CompilationContext, **kwargs: Any) -> CompilationContext:
        """Classify edges as necessary/unnecessary and find parallel groups.

        Args:
            context: Current compilation context with ``graph``,
                ``node_analyses``, ``reducer_fields`` in metadata.
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            The updated context with ``edge_analyses``, ``necessary_edges``,
            ``unnecessary_edges``, and ``parallel_groups`` in metadata.
        """
        graph = context.metadata["graph"]
        node_analyses = context.metadata["node_analyses"]
        reducer_fields = context.metadata.get("reducer_fields") or {}

        edge_analyses = classify_edges(graph, node_analyses, reducer_fields)

        necessary: set[tuple[str, str]] = set()
        unnecessary: set[tuple[str, str]] = set()
        for ea in edge_analyses:
            edge = (ea.source, ea.target)
            if ea.edge_type.value == "necessary":
                necessary.add(edge)
            elif ea.edge_type.value == "unnecessary":
                unnecessary.add(edge)

        logger.info("Edge classification: %d necessary, %d unnecessary", len(necessary), len(unnecessary))

        parallel_groups = find_parallel_groups(
            node_analyses,
            build_dependency_graph(node_analyses, reducer_fields),
            reducer_fields,
        )

        context.metadata["edge_analyses"] = edge_analyses
        context.metadata["necessary_edges"] = necessary
        context.metadata["unnecessary_edges"] = unnecessary
        context.metadata["parallel_groups"] = parallel_groups

        return context
