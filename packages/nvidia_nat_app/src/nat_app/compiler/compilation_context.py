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
CompilationContext: shared state that travels through the optimization pipeline.

Stages read and write metadata here so that downstream stages can reuse
analysis from upstream stages without re-computing it.

Standard metadata keys (written by built-in stages):

- ``graph``: The abstract ``Graph``
- ``topology``: ``GraphTopology``
- ``node_analyses``: ``dict[str, NodeAnalysis]``
- ``node_funcs``: ``dict[str, Callable]``
- ``reducer_fields``: ``ReducerSet``
- ``edge_analyses``: list of edge analysis results
- ``necessary_edges``: ``set[tuple[str, str]]``
- ``unnecessary_edges``: ``set[tuple[str, str]]``
- ``optimized_order``: ``list[set[str]]``
- ``branch_info``: dict of branch domain info
- ``warnings``: ``list[str]``

Framework-specific stages should namespace their own keys
(e.g. ``parallel.node_rw``, ``speculative.analysis``).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import TypeVar

if TYPE_CHECKING:
    from nat_app.graph.analysis import NodeAnalysis
    from nat_app.graph.topology import GraphTopology
    from nat_app.graph.types import Graph

_CompiledArtifactType = TypeVar("_CompiledArtifactType")


@dataclass
class CompilationContext(Generic[_CompiledArtifactType]):
    """Mutable context that flows through the optimization pipeline.

    Attributes:
        compiled: The current compiled artifact (updated by each stage).
        metadata: Free-form dict for inter-stage communication.
    """

    compiled: _CompiledArtifactType
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def graph(self) -> Graph | None:
        """The abstract Graph, or None if ExtractStage hasn't run."""
        return self.metadata.get("graph")

    @property
    def topology(self) -> GraphTopology | None:
        """The GraphTopology, or None if TopologyStage hasn't run."""
        return self.metadata.get("topology")

    @property
    def node_analyses(self) -> dict[str, NodeAnalysis] | None:
        """Per-node analysis results, or None if NodeAnalysisStage hasn't run."""
        return self.metadata.get("node_analyses")

    @property
    def optimized_order(self) -> list[set[str]] | None:
        """Parallel stage groupings, or None if SchedulingStage hasn't run."""
        return self.metadata.get("optimized_order")

    @property
    def necessary_edges(self) -> set[tuple[str, str]] | None:
        """Necessary edges, or None if EdgeClassificationStage hasn't run."""
        return self.metadata.get("necessary_edges")

    @property
    def unnecessary_edges(self) -> set[tuple[str, str]] | None:
        """Unnecessary edges, or None if EdgeClassificationStage hasn't run."""
        return self.metadata.get("unnecessary_edges")
