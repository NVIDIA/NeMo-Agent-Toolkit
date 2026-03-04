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
"""Tests for build_graph_and_adapter."""

import pytest

from nat_app.graph.factory import build_graph_and_adapter
from nat_app.graph.types import Graph


def _dummy(state):
    return state


class TestBuildGraphAndAdapter:

    def test_basic_graph_and_adapter(self):
        g, adapter = build_graph_and_adapter(
            nodes={"a": _dummy, "b": _dummy},
            edges=[("a", "b")],
        )
        assert isinstance(g, Graph)
        assert g.has_node("a")
        assert g.has_node("b")
        assert g.entry_point == "a"
        assert g.edge_count == 1

    def test_with_conditional_edges(self):
        g, adapter = build_graph_and_adapter(
            nodes={"router": _dummy, "a": _dummy, "b": _dummy},
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {"left": "a", "right": "b"}},
        )
        assert g.get_conditional_targets("router") is not None
        assert g.get_conditional_targets("router")["left"] == ["a"]
        assert g.get_conditional_targets("router")["right"] == ["b"]

    def test_adapter_get_self_state_attrs_returns_provided(self):
        attrs = {"state": "state", "memory": "memory"}
        g, adapter = build_graph_and_adapter(
            nodes={"a": _dummy},
            edges=[],
            self_state_attrs=attrs,
        )
        assert adapter.get_self_state_attrs() == attrs

    def test_adapter_get_self_state_attrs_none_when_not_provided(self):
        g, adapter = build_graph_and_adapter(
            nodes={"a": _dummy},
            edges=[],
        )
        assert adapter.get_self_state_attrs() is None

    def test_entry_point_explicit(self):
        g, adapter = build_graph_and_adapter(
            nodes={"a": _dummy, "b": _dummy},
            edges=[("a", "b")],
            entry="b",
        )
        assert g.entry_point == "b"

    def test_terminal_nodes_set(self):
        g, adapter = build_graph_and_adapter(
            nodes={"a": _dummy, "b": _dummy},
            edges=[("a", "b")],
        )
        assert "b" in g.terminal_nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
