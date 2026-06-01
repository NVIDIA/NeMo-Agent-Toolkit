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
"""Shared test fixtures for nvidia_nat_app."""

from __future__ import annotations

import typing

import pytest

if typing.TYPE_CHECKING:
    from nat_app.graph.types import Graph


def _new_graph() -> Graph:
    from nat_app.graph.types import Graph

    return Graph()


@pytest.fixture(autouse=True)
def _suppress_experimental_warning():
    """Suppress the package-level experimental warning during tests."""
    import warnings

    from nat_app import ExperimentalWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        yield


@pytest.fixture(name="linear_graph")
def linear_graph_fixture() -> Graph:
    """A -> B -> C linear chain."""
    g = _new_graph()
    g.add_node("a")
    g.add_node("b")
    g.add_node("c")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.entry_point = "a"
    return g


@pytest.fixture(name="cycle_graph")
def cycle_graph_fixture() -> Graph:
    """A -> B -> C -> A cycle."""
    g = _new_graph()
    g.add_node("a")
    g.add_node("b")
    g.add_node("c")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    g.entry_point = "a"
    return g


@pytest.fixture(name="router_graph")
def router_graph_fixture() -> Graph:
    """Router with left/right branches merging."""
    g = _new_graph()
    g.add_node("router")
    g.add_node("left")
    g.add_node("right")
    g.add_node("merge")
    g.add_edge("router", "left")
    g.add_edge("router", "right")
    g.add_edge("left", "merge")
    g.add_edge("right", "merge")
    g.add_conditional_edges("router", {"go_left": ["left"], "go_right": ["right"]})
    g.entry_point = "router"
    return g


@pytest.fixture(name="diamond_graph")
def diamond_graph_fixture() -> Graph:
    """A fans out to B and C, both merge into D."""
    g = _new_graph()
    g.add_node("a", func=lambda s: s)
    g.add_node("b", func=lambda s: s)
    g.add_node("c", func=lambda s: s)
    g.add_node("d", func=lambda s: s)
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("b", "d")
    g.add_edge("c", "d")
    g.entry_point = "a"
    return g


@pytest.fixture(name="simple_graph")
def simple_graph_fixture() -> Graph:
    """Two-node graph with functions and terminal set, for compiler tests."""
    g = _new_graph()
    g.add_node("a", func=lambda s: {"x": 1})
    g.add_node("b", func=lambda s: {"y": s["x"]})
    g.add_edge("a", "b")
    g.entry_point = "a"
    g.terminal_nodes = {"b"}
    return g


@pytest.fixture(name="nested_cycle_graph")
def nested_cycle_graph_fixture() -> Graph:
    """Nested cycles sharing node ``evaluate``."""
    g = _new_graph()
    for name in ("parse", "search", "evaluate", "refine", "decide"):
        g.add_node(name)
    g.add_edge("parse", "search")
    g.add_edge("search", "evaluate")
    g.add_edge("evaluate", "refine")
    g.add_edge("refine", "evaluate")
    g.add_edge("evaluate", "decide")
    g.add_edge("decide", "parse")
    g.entry_point = "parse"
    return g


@pytest.fixture(name="disjoint_cycles_graph")
def disjoint_cycles_graph_fixture() -> Graph:
    """Two independent cycles behind a linear entry."""
    g = _new_graph()
    for name in ("entry", "loop_a", "check_a", "bridge", "loop_b", "check_b"):
        g.add_node(name)
    g.add_edge("entry", "loop_a")
    g.add_edge("loop_a", "check_a")
    g.add_edge("check_a", "loop_a")
    g.add_edge("check_a", "bridge")
    g.add_edge("bridge", "loop_b")
    g.add_edge("loop_b", "check_b")
    g.add_edge("check_b", "loop_b")
    g.entry_point = "entry"
    return g


@pytest.fixture(name="parallelizable_cycle_graph")
def parallelizable_cycle_graph_fixture() -> Graph:
    """Cycle with parallelizable body: entry fans out to a and b, both merge to exit."""
    g = _new_graph()
    for n in ("entry", "a", "b", "exit"):
        g.add_node(n)
    g.add_edge("entry", "a")
    g.add_edge("entry", "b")
    g.add_edge("a", "exit")
    g.add_edge("b", "exit")
    g.add_edge("exit", "entry")
    g.entry_point = "entry"
    return g


@pytest.fixture(name="overlapping_cycles_graph")
def overlapping_cycles_graph_fixture() -> Graph:
    """Two cycles sharing a common edge segment (A -> B)."""
    g = _new_graph()
    for name in ("a", "b", "c", "d"):
        g.add_node(name)
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    g.add_edge("b", "d")
    g.add_edge("d", "a")
    g.entry_point = "a"
    return g
