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
"""Tests for nested path tracking in AccessSet and AST analysis."""
from typing import TypedDict

from nat_app.graph.access import AccessSet
from nat_app.graph.analysis import NodeAnalysis
from nat_app.graph.static_analysis import analyze_function_ast

# -- AccessSet tests -------------------------------------------------------


def test_accessset_flat_overlaps():
    a = AccessSet.from_fields("query", "data")
    b = AccessSet.from_fields("data", "response")
    assert a.overlaps(b) is True
    assert a.is_flat is True


def test_accessset_flat_no_overlap():
    a = AccessSet.from_fields("query")
    b = AccessSet.from_fields("response")
    assert a.overlaps(b) is False


def test_accessset_nested_overlap_parent_child():
    a = AccessSet.from_fields("user.name")
    b = AccessSet.from_fields("user")
    assert a.is_flat is False
    assert b.is_flat is True
    assert a.overlaps(b) is True


def test_accessset_nested_no_overlap_siblings():
    a = AccessSet.from_fields("user.name")
    b = AccessSet.from_fields("user.email")
    assert a.overlaps(b) is False


def test_accessset_multi_object_no_cross_overlap():
    a = AccessSet()
    a.add("state", "name")
    b = AccessSet()
    b.add("memory", "name")
    assert a.overlaps(b) is False


def test_accessset_reducers():
    a = AccessSet.from_fields("messages", "query")
    b = AccessSet.from_fields("messages", "response")
    reducers = {"state": {"messages"}}
    assert a.overlaps(b, exclude_reducers=reducers) is False


def test_accessset_add_flat():
    a = AccessSet()
    a.add_flat("query")
    a.add_flat("response")
    assert a.fields() == {"query", "response"}
    assert a.is_flat is True


def test_accessset_from_set():
    a = AccessSet.from_set({"a", "b", "c"})
    assert a.fields() == {"a", "b", "c"}
    assert a.all_fields_flat == {"a", "b", "c"}


def test_accessset_and_intersection():
    a = AccessSet.from_fields("query", "data", "x")
    b = AccessSet.from_fields("data", "response", "x")
    inter = a & b
    assert inter.fields() == {"data", "x"}


def test_accessset_sub_with_accessset():
    a = AccessSet.from_fields("query", "data", "response")
    b = AccessSet.from_fields("data")
    diff = a - b
    assert diff.fields() == {"query", "response"}


def test_accessset_sub_with_reducerset():
    a = AccessSet.from_fields("messages", "query")
    reducers = {"state": {"messages"}}
    diff = a - reducers
    assert diff.fields() == {"query"}


def test_accessset_objects():
    a = AccessSet()
    a.add("state", "query")
    a.add("memory", "cache")
    assert a.objects == {"state", "memory"}


def test_accessset_fields_per_object():
    a = AccessSet()
    a.add("state", "query")
    a.add("state", "response")
    a.add("memory", "cache")
    assert a.fields("state") == {"query", "response"}
    assert a.fields("memory") == {"cache"}
    assert a.fields("nonexistent") == set()


# -- Conflict detection with nested paths ----------------------------------


def test_conflict_write_child_read_parent():
    na = NodeAnalysis(name="a")
    na.mutations = AccessSet.from_fields("user.name")
    nb = NodeAnalysis(name="b")
    nb.reads = AccessSet.from_fields("user")
    assert na.conflicts_with(nb) is True


def test_no_conflict_write_sibling_paths():
    na = NodeAnalysis(name="a")
    na.mutations = AccessSet.from_fields("user.name")
    nb = NodeAnalysis(name="b")
    nb.mutations = AccessSet.from_fields("user.email")
    assert na.conflicts_with(nb) is False


def test_conflict_both_write_same_nested():
    na = NodeAnalysis(name="a")
    na.mutations = AccessSet.from_fields("user.name")
    nb = NodeAnalysis(name="b")
    nb.mutations = AccessSet.from_fields("user.name")
    assert na.conflicts_with(nb) is True


# -- AST analysis with nested state access ---------------------------------


class NestedState(TypedDict):
    user: dict
    config: dict
    query: str


def nested_write_fn(state: NestedState):
    """Writes to state['user']['name']."""
    state["user"]["name"] = "Alice"
    return {}


def nested_read_fn(state: NestedState):
    """Reads state['user']['preferences']['theme']."""
    theme = state["user"]["preferences"]["theme"]
    return {"result": theme}


def nested_augassign_fn(state: NestedState):
    """Augmented assignment: state['config']['count'] += 1."""
    state["config"]["count"] += 1
    return {}


def flat_fn(state: NestedState):
    """Flat access: state['query']."""
    q = state["query"]
    return {"response": q.upper()}


def mixed_fn(state: NestedState):
    """Mix of flat and nested."""
    q = state["query"]
    state["user"]["last_query"] = q
    return {"response": q}


def test_ast_nested_write():
    r = analyze_function_ast(nested_write_fn)
    assert ("state", "user.name") in list(r.mutations)


def test_ast_nested_read():
    r = analyze_function_ast(nested_read_fn)
    reads_list = list(r.reads)
    assert ("state", "user.preferences.theme") in reads_list or ("state", "user") in reads_list


def test_ast_nested_augassign():
    r = analyze_function_ast(nested_augassign_fn)
    assert ("state", "config.count") in list(r.mutations)


def test_ast_flat_stays_flat():
    r = analyze_function_ast(flat_fn)
    assert ("state", "query") in list(r.reads)
    assert r.reads.is_flat is True


def test_ast_mixed_goes_nested():
    r = analyze_function_ast(mixed_fn)
    mutations_list = list(r.mutations)
    assert ("state", "user.last_query") in mutations_list


# -- Attribute-based access on custom objects (non-dict state) -------------


def attr_write_fn(memory):
    """Attribute write: memory.last_query = ..."""
    memory.last_query = "hello"
    return {}


def attr_deep_write_fn(memory):
    """Deep attribute chain write: memory.user.preferences.theme = ..."""
    memory.user.preferences.theme = "dark"
    return {}


def attr_augassign_fn(memory):
    """Attribute augmented assignment: memory.count += 1."""
    memory.count += 1
    return {}


def attr_delete_fn(memory):
    """Attribute delete: del memory.last_query."""
    del memory.last_query
    return {}


def attr_mutating_method_fn(memory):
    """Mutating method on attribute: memory.conversations.append(x)."""
    memory.conversations.append("new message")
    return {}


def attr_read_fn(memory):
    """Attribute read: x = memory.last_query."""
    x = memory.last_query
    return {"result": x}


def attr_mixed_rw_fn(memory):
    """Read one attr, write another."""
    q = memory.last_query
    memory.response = q.upper()
    return {}


def test_attr_write():
    r = analyze_function_ast(attr_write_fn, param_to_obj={"memory": "memory"})
    assert ("memory", "last_query") in list(r.mutations)


def test_attr_deep_write():
    r = analyze_function_ast(attr_deep_write_fn, param_to_obj={"memory": "memory"})
    mutations = list(r.mutations)
    assert ("memory", "user.preferences.theme") in mutations


def test_attr_augassign():
    r = analyze_function_ast(attr_augassign_fn, param_to_obj={"memory": "memory"})
    assert ("memory", "count") in list(r.mutations)
    assert ("memory", "count") in list(r.reads)


def test_attr_delete():
    r = analyze_function_ast(attr_delete_fn, param_to_obj={"memory": "memory"})
    assert ("memory", "last_query") in list(r.mutations)


def test_attr_mutating_method():
    r = analyze_function_ast(attr_mutating_method_fn, param_to_obj={"memory": "memory"})
    mutations = list(r.mutations)
    assert ("memory", "conversations") in mutations


def test_attr_read():
    r = analyze_function_ast(attr_read_fn, param_to_obj={"memory": "memory"})
    assert ("memory", "last_query") in list(r.reads)


def test_attr_mixed_rw():
    r = analyze_function_ast(attr_mixed_rw_fn, param_to_obj={"memory": "memory"})
    assert ("memory", "last_query") in list(r.reads)
    assert ("memory", "response") in list(r.mutations)


def test_attr_no_conflict_different_objects():
    """Two nodes writing to different objects should not conflict."""
    na = NodeAnalysis(name="a")
    na.mutations = AccessSet()
    na.mutations.add("state", "query")

    nb = NodeAnalysis(name="b")
    nb.mutations = AccessSet()
    nb.mutations.add("memory", "query")

    assert na.conflicts_with(nb) is False


def test_attr_conflict_same_object_same_field():
    """Two nodes writing to same field on same object conflict."""
    na = NodeAnalysis(name="a")
    na.mutations = AccessSet()
    na.mutations.add("memory", "count")

    nb = NodeAnalysis(name="b")
    nb.mutations = AccessSet()
    nb.mutations.add("memory", "count")

    assert na.conflicts_with(nb) is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
