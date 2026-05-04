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
First-class Graph type — the central interchange format for all nat_app algorithms.

Framework adapter packages produce ``Graph`` objects via the
`GraphExtractor` protocol.  All analysis,
scheduling, and optimization algorithms accept ``Graph`` as input.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any


class PriorityLevel(Enum):
    """Discrete scheduling priority tiers for inference requests.

    Used by `PriorityAssignmentStage`
    to assign priorities relative to branch groups.  The numeric values are
    written directly to ``NodeInfo.priority`` and propagated to the inference
    cluster via ``nvext.agent_hints.priority``.

    Higher float value = higher scheduling priority on the cluster.
    """

    HIGH = 1.0
    MEDIUM = 0.5
    LOW = 0.1


@dataclass(frozen=True)
class ProfiledNodeCost:
    """Observed cost metrics for a graph node, aggregated from profiling runs.

    This is the format contract for profiled priority assignment.
    Framework adapters populate this from their profiler output and
    inject it via ``seed_context`` into
    ``context.metadata["profiled_node_costs"]``.

    Fields split into "self" metrics (this node only) and "subtree"
    metrics (includes downstream costs computed by the profiler):

    Self metrics (need graph-based subtree propagation):
        llm_call_count, total_prompt_tokens, total_completion_tokens,
        total_tokens, self_time_ms

    Subtree metrics (already propagated, use directly):
        subtree_time_ms, total_latency_ms
    """

    llm_call_count: int = 0
    total_latency_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    self_time_ms: float = 0.0
    subtree_time_ms: float = 0.0


class CostMetric(Enum):
    """Named cost metric presets for profiled priority assignment.

    Used to select which field of `ProfiledNodeCost` drives the
    priority algorithm.  Pass to ``PriorityAssignmentStage(cost_metric=...)``.

    Each preset has an associated ``pre_propagated`` flag (see
    ``_COST_METRIC_INFO`` in ``priority_assignment.py``) indicating
    whether the metric already includes downstream subtree costs
    (from the profiler) or needs graph-based propagation by the stage.
    """

    LLM_CALLS = "llm_calls"
    WALL_CLOCK_MS = "wall_clock_ms"
    PROMPT_TOKENS = "prompt_tokens"
    COMPLETION_TOKENS = "completion_tokens"
    TOTAL_TOKENS = "total_tokens"
    SUBTREE_TIME = "subtree_time"


class BranchGroupType(Enum):
    """Classification of a branch group in the priority assignment algorithm."""

    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    LINEAR = "linear"


@dataclass
class BranchGroup:
    """A group of nodes that share a common branching point.

    Used by ``PriorityAssignmentStage`` and ``PriorityStrategy`` to represent
    nodes grouped by conditional routers, parallel fan-out, or linear chains.
    """

    name: str
    group_type: BranchGroupType
    node_names: list[str] = field(default_factory=list)
    subtree_costs: list[float] = field(default_factory=list)
    priorities: list[PriorityLevel] = field(default_factory=list)
    ceiling: PriorityLevel | None = None


class EdgeKind(Enum):
    """Classification of an edge in the graph."""

    DIRECT = "direct"
    CONDITIONAL = "conditional"


@dataclass(frozen=True)
class Edge:
    """A directed edge between two nodes."""

    source: str
    target: str
    kind: EdgeKind = EdgeKind.DIRECT
    branch: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


@dataclass
class NodeInfo:
    """Metadata attached to a graph node."""

    func: Callable | Any | None = None
    """The callable associated with this node (if available)."""

    priority: float | None = None
    """Scheduling priority for inference requests. Higher values are
    scheduled first on the inference cluster. None means not yet
    assigned (cluster uses its own default). Set by the priority
    compilation stage or explicitly by the user."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary key-value metadata (analysis results, labels, etc.)."""


class Graph:
    """
    Directed graph with typed node/edge metadata.

    This is the interchange type that all ``nat_app.graph`` algorithms accept.
    Framework adapter packages produce ``Graph`` via the ``GraphExtractor`` protocol.

    Nodes are identified by string names.  Edges carry a `EdgeKind` and
    optional metadata.  Conditional edges (routers) are represented as edges
    with ``kind=EdgeKind.CONDITIONAL``.

    Minimum required for optimization:

        graph.add_node(name, func=callable)   # for each node
        graph.add_edge(source, target)        # for each dependency
        graph.entry_point = "start_node"      # entry point
        graph.terminal_nodes = {"end_node"}   # terminal nodes

    Optional (unlocks advanced analysis):

        graph.add_conditional_edges(...)      # for routers / branching
        graph.metadata on nodes               # for framework-specific data

    See `minimal` for a one-call factory that builds a valid graph
    from plain Python data.

    Example:

        g = Graph()
        g.add_node("fetch", func=fetch_fn)
        g.add_node("process", func=process_fn)
        g.add_edge("fetch", "process")
        g.add_conditional_edges("router", {"branch_a": "tool_a", "branch_b": "tool_b"})

        g.entry_point = "fetch"
        g.terminal_nodes.add("process")
    """

    def __init__(self) -> None:
        self._nodes: dict[str, NodeInfo] = {}
        self._edges: list[Edge] = []
        self._edge_keys: set[tuple[str, str, str | None]] = set()

        self._successors: dict[str, set[str]] = {}
        self._predecessors: dict[str, set[str]] = {}
        self._conditional_targets: dict[str, dict[str, list[str]]] = {}

        self.entry_point: str = ""
        """The graph's entry node (set by the framework adapter)."""

        self.terminal_nodes: set[str] = set()
        """Nodes that lead to graph termination."""

    @classmethod
    def minimal(
        cls,
        nodes: dict[str, Callable | None],
        edges: list[tuple[str, str]],
        entry: str | None = None,
    ) -> Graph:
        """Create a Graph with the minimum fields needed for optimization.

        This factory builds a valid graph from plain Python data -- handy for
        framework teams that want ``DefaultGraphCompiler`` integration without
        thinking about which fields matter.

        Args:
            nodes: Mapping of node name to callable (or ``None``).
            edges: List of ``(source, target)`` dependency edges.
            entry: Entry-point node.  Defaults to the first key in *nodes*.

        Returns:
            A fully-wired ``Graph`` ready for compilation.
        """
        g = cls()
        for name, func in nodes.items():
            g.add_node(name, func=func)
        for src, tgt in edges:
            g.add_edge(src, tgt)

        if entry:
            g.entry_point = entry
        elif nodes:
            g.entry_point = next(iter(nodes))

        nodes_with_downstream = {src for src, _ in edges}
        for name in nodes:
            if name not in nodes_with_downstream:
                g.terminal_nodes.add(name)

        return g

    # -- Node operations ---------------------------------------------------

    def add_node(
        self,
        name: str,
        func: Callable | Any | None = None,
        priority: float | None = None,
        **metadata: Any,
    ) -> None:
        """Add a node with optional function reference, priority, and metadata.

        Args:
            name: Unique node identifier.
            func: The callable associated with this node.
            priority: Scheduling priority for inference requests.
            **metadata: Arbitrary key-value metadata.
        """
        self._nodes[name] = NodeInfo(func=func, priority=priority, metadata=metadata)
        self._successors.setdefault(name, set())
        self._predecessors.setdefault(name, set())

    def has_node(self, name: str) -> bool:
        """Check whether a node exists in the graph.

        Args:
            name: The node name to look up.

        Returns:
            True if the node exists, False otherwise.
        """
        return name in self._nodes

    def get_node(self, name: str) -> NodeInfo:
        """Return the NodeInfo for a node by name.

        Args:
            name: The node name to look up.

        Returns:
            The NodeInfo for the node.

        Raises:
            KeyError: If the node does not exist.
        """
        return self._nodes[name]

    @property
    def node_names(self) -> set[str]:
        """All node names in the graph.

        Returns:
            Set of all node names.
        """
        return set(self._nodes)

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph.

        Returns:
            The count of nodes.
        """
        return len(self._nodes)

    def nodes(self) -> Iterator[tuple[str, NodeInfo]]:
        """Iterate over ``(name, NodeInfo)`` pairs.

        Returns:
            Iterator of (name, NodeInfo) pairs.
        """
        yield from self._nodes.items()

    # -- Edge operations ---------------------------------------------------

    def add_edge(self, source: str, target: str, **metadata: Any) -> None:
        """Add a direct edge between two nodes.

        Args:
            source: Source node name.
            target: Target node name.
            **metadata: Arbitrary key-value metadata for the edge.
        """
        key = (source, target, None)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        edge = Edge(source=source, target=target, kind=EdgeKind.DIRECT, metadata=metadata)
        self._edges.append(edge)
        self._successors.setdefault(source, set()).add(target)
        self._predecessors.setdefault(target, set()).add(source)

    def _remove_conditional_edges_for_source(self, source: str) -> None:
        """Remove all conditional edges from source. Keeps _edges, _edge_keys, _successors, _predecessors consistent."""
        to_remove = [e for e in self._edges if e.source == source and e.kind == EdgeKind.CONDITIONAL]
        for e in to_remove:
            self._edges.remove(e)
            key = (e.source, e.target, e.branch)
            self._edge_keys.discard(key)
            self._successors.get(source, set()).discard(e.target)
            self._predecessors.get(e.target, set()).discard(source)
        if source in self._conditional_targets:
            del self._conditional_targets[source]

    def add_conditional_edges(
        self,
        source: str,
        branch_targets: dict[str, str | list[str]],
        **metadata: Any,
    ) -> None:
        """
        Add conditional (router) edges from *source* to multiple targets.

        Args:
            source: The router node name.
            branch_targets: Mapping of ``branch_name -> target_node(s)``.
                Each value may be a single node name or a list for
                1-to-many routing (one label triggers multiple targets).
            **metadata: Attached to each created edge.
        """
        self._remove_conditional_edges_for_source(source)
        normalized: dict[str, list[str]] = {
            label: [t] if isinstance(t, str) else list(t)
            for label, t in branch_targets.items()
        }
        self._conditional_targets[source] = normalized
        for branch_name, targets in normalized.items():
            for target in targets:
                key = (source, target, branch_name)
                if key in self._edge_keys:
                    continue
                self._edge_keys.add(key)
                edge = Edge(
                    source=source,
                    target=target,
                    kind=EdgeKind.CONDITIONAL,
                    branch=branch_name,
                    metadata=metadata,
                )
                self._edges.append(edge)
                self._successors.setdefault(source, set()).add(target)
                self._predecessors.setdefault(target, set()).add(source)

    @property
    def edges(self) -> list[Edge]:
        """All edges in the graph.

        Returns:
            List of all edges.
        """
        return list(self._edges)

    @property
    def edge_pairs(self) -> list[tuple[str, str]]:
        """All edges as ``(source, target)`` tuples (convenience).

        Returns:
            List of (source, target) tuples.
        """
        return [(e.source, e.target) for e in self._edges]

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph.

        Returns:
            The count of edges.
        """
        return len(self._edges)

    def get_conditional_targets(self, node: str) -> dict[str, list[str]] | None:
        """Return the branch_name -> target list mapping for a conditional node, or None.

        Args:
            node: The node name to look up.

        Returns:
            Branch mapping if the node is conditional, or None.
        """
        return self._conditional_targets.get(node)

    @property
    def conditional_edge_sources(self) -> dict[str, dict[str, list[str]]]:
        """All conditional edge sources and their branch mappings.

        Returns:
            Mapping of source node to branch target mappings.
        """
        return dict(self._conditional_targets)

    # -- Adjacency ---------------------------------------------------------

    def successors(self, node: str) -> list[str]:
        """Direct successors of *node*.

        Args:
            node: The node name.

        Returns:
            List of successor node names.
        """
        return list(self._successors.get(node, set()))

    def predecessors(self, node: str) -> list[str]:
        """Direct predecessors of *node*.

        Args:
            node: The node name.

        Returns:
            List of predecessor node names.
        """
        return list(self._predecessors.get(node, set()))

    def to_adjacency(self) -> dict[str, list[str]]:
        """Full forward adjacency dict.

        Returns:
            Mapping of each node to its list of successors.
        """
        return {n: list(succs) for n, succs in self._successors.items()}

    # -- Subgraph ----------------------------------------------------------

    def subgraph(self, nodes: set[str]) -> Graph:
        """Return a new Graph containing only the specified nodes and their inter-edges.

        Args:
            nodes: Set of node names to include.

        Returns:
            A new Graph with the specified nodes and edges between them.
        """
        sub = Graph()
        for name in nodes:
            if name in self._nodes:
                info = self._nodes[name]
                sub.add_node(name, func=info.func, priority=info.priority, **info.metadata)
        for edge in self._edges:
            if edge.source in nodes and edge.target in nodes:
                if edge.kind == EdgeKind.CONDITIONAL:
                    branch = edge.branch or ""
                    existing = sub._conditional_targets.setdefault(edge.source, {})
                    existing.setdefault(branch, []).append(edge.target)
                key = (
                    edge.source,
                    edge.target,
                    edge.branch if edge.kind == EdgeKind.CONDITIONAL else None,
                )
                sub._edge_keys.add(key)
                sub._edges.append(edge)
                sub._successors.setdefault(edge.source, set()).add(edge.target)
                sub._predecessors.setdefault(edge.target, set()).add(edge.source)
        if self.entry_point in nodes:
            sub.entry_point = self.entry_point
        sub.terminal_nodes = self.terminal_nodes & nodes
        return sub

    # -- Validation --------------------------------------------------------

    def validate(self) -> list[str]:
        """Check structural invariants.

        Checks:
        - ``entry_point`` is set and exists in nodes
        - All ``terminal_nodes`` exist in nodes
        - All edge endpoints exist in nodes
        - No orphan nodes (unreachable from entry point)

        Returns:
            List of issues found (empty means valid).
        """
        issues: list[str] = []

        if not self.entry_point:
            issues.append("No entry_point set")
        elif self.entry_point not in self._nodes:
            issues.append(f"entry_point '{self.entry_point}' not in nodes")

        for name in self.terminal_nodes:
            if name not in self._nodes:
                issues.append(f"Terminal node '{name}' not in nodes")

        for edge in self._edges:
            if edge.source not in self._nodes:
                issues.append(f"Edge source '{edge.source}' not in nodes")
            if edge.target not in self._nodes:
                issues.append(f"Edge target '{edge.target}' not in nodes")

        if self.entry_point and self.entry_point in self._nodes:
            reachable = self._compute_reachable(self.entry_point)
            orphans = set(self._nodes.keys()) - reachable
            if orphans:
                issues.append(f"Unreachable nodes from entry: {sorted(orphans)}")

        return issues

    def _compute_reachable(self, start: str) -> set[str]:
        """BFS from *start* to find all reachable nodes.

        Args:
            start: The starting node name.

        Returns:
            Set of all reachable node names.
        """
        visited: set[str] = set()
        queue: deque[str] = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for succ in self._successors.get(node, set()):
                if succ not in visited:
                    queue.append(succ)
        return visited

    # -- Hashing (for caching) --------------------------------------------

    @property
    def structure_hash(self) -> str:
        """
        Content-addressable hash of the graph structure (nodes + edges).

        Two graphs with the same nodes and edges produce the same hash,
        regardless of insertion order.  Used for analysis caching.

        Returns:
            A hex digest string identifying the graph structure.
        """
        parts = sorted(self._nodes.keys())
        parts.extend(sorted(f"{e.source}->{e.target}:{e.kind.value}:{e.branch or ''}" for e in self._edges))
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    # -- Representation ----------------------------------------------------

    def __repr__(self) -> str:
        return f"Graph(nodes={self.node_count}, edges={self.edge_count}, entry={self.entry_point!r})"

    def __len__(self) -> int:
        return self.node_count
