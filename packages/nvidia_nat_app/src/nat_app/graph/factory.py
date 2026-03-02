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
Graph factory: build a ``Graph`` and lightweight adapter from raw Python data.

Used by ``nat_app.api`` (simplified API for framework teams) and
``nat_app.speculation.plan`` (speculation planning from raw graph data).
"""

from __future__ import annotations

from collections.abc import Callable

from nat_app.graph.adapter import AbstractFrameworkAdapter
from nat_app.graph.types import Graph


def build_graph_and_adapter(
    nodes: dict[str, Callable | None],
    edges: list[tuple[str, str]],
    entry: str | None = None,
    conditional_edges: dict[str, dict[str, str | list[str]]] | None = None,
    self_state_attrs: dict[str, str] | None = None,
) -> tuple[Graph, AbstractFrameworkAdapter]:
    """Build a Graph and lightweight adapter from raw data.

    Args:
        nodes: Mapping of node name to callable function (or None).
        edges: List of ``(source, target)`` dependency edges.
        entry: Entry point node name. Defaults to the first key in ``nodes``.
        conditional_edges: Router/conditional edges mapping a router node
            to its branch targets.
        self_state_attrs: For class methods, maps ``self.X`` attribute names
            to object namespaces.

    Returns:
        A tuple of ``(graph, adapter)`` where *graph* is a populated ``Graph``
        and *adapter* is a lightweight ``AbstractFrameworkAdapter`` instance.
    """
    g = Graph()
    for name, func in nodes.items():
        g.add_node(name, func=func)
    for src, tgt in edges:
        g.add_edge(src, tgt)
    if conditional_edges:
        for node, targets in conditional_edges.items():
            g.add_conditional_edges(node, targets)

    if entry:
        g.entry_point = entry
    elif nodes:
        g.entry_point = next(iter(nodes))

    node_names_set = set(nodes.keys())
    nodes_with_downstream = {src for src, _ in edges}
    if conditional_edges:
        nodes_with_downstream.update(conditional_edges.keys())
    for name in node_names_set:
        if name not in nodes_with_downstream:
            g.terminal_nodes.add(name)

    class _QuickAdapter(AbstractFrameworkAdapter):

        def extract(self, source):
            return source

        def build(self, original, result):
            return result

        def get_self_state_attrs(self):
            return self_state_attrs

    return g, _QuickAdapter()
