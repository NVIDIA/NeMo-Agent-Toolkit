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
"""Tests for constraint resolution: get_constraints, resolve_constraints, apply_constraints_to_analysis, merge_deps."""

import pytest

from nat_app.constraints.decorators import depends_on
from nat_app.constraints.decorators import has_side_effects
from nat_app.constraints.decorators import sequential
from nat_app.constraints.models import OptimizationConfig
from nat_app.constraints.resolution import apply_constraints_to_analysis
from nat_app.constraints.resolution import get_constraints
from nat_app.constraints.resolution import merge_dependencies
from nat_app.constraints.resolution import resolve_constraints
from nat_app.graph.analysis import NodeAnalysis

# -- Test functions -----------------------------------------------------------


def plain_fn(state):
    pass


@sequential(reason="DB write")
def sequential_fn(state):
    pass


@depends_on("upstream", reason="Needs upstream")
def dependent_fn(state):
    pass


@has_side_effects(reason="HTTP call")
def side_effect_fn(state):
    pass


# -- get_constraints ----------------------------------------------------------


class TestGetConstraints:

    def test_returns_none_for_undecorated(self):
        assert get_constraints(plain_fn) is None

    def test_returns_constraints_for_decorated(self):
        constraints = get_constraints(sequential_fn)
        assert constraints is not None
        assert constraints.force_sequential is True

    def test_returns_depends_on(self):
        constraints = get_constraints(dependent_fn)
        assert constraints is not None
        assert "upstream" in constraints.depends_on


# -- resolve_constraints ------------------------------------------------------


class TestResolveConstraints:

    def test_undecorated_with_default_config(self):
        config = OptimizationConfig()
        resolved = resolve_constraints("my_node", plain_fn, config)
        assert resolved.force_sequential is False
        assert resolved.has_side_effects is False

    def test_decorator_sets_force_sequential(self):
        config = OptimizationConfig()
        resolved = resolve_constraints("seq_node", sequential_fn, config)
        assert resolved.force_sequential is True
        assert resolved.source == "decorator"

    def test_decorator_sets_side_effects(self):
        config = OptimizationConfig()
        resolved = resolve_constraints("side_node", side_effect_fn, config)
        assert resolved.has_side_effects is True

    def test_config_force_sequential(self):
        config = OptimizationConfig(force_sequential={"my_node"})
        resolved = resolve_constraints("my_node", plain_fn, config)
        assert resolved.force_sequential is True
        assert any("force_sequential" in r for r in resolved.reasons)

    def test_config_explicit_dependencies(self):
        config = OptimizationConfig(explicit_dependencies={"my_node": {"dep_a"}})
        resolved = resolve_constraints("my_node", plain_fn, config)
        assert "dep_a" in resolved.explicit_dependencies

    def test_config_side_effect_nodes(self):
        config = OptimizationConfig(side_effect_nodes={"my_node"})
        resolved = resolve_constraints("my_node", plain_fn, config)
        assert resolved.has_side_effects is True

    def test_heuristic_keyword_match(self):
        config = OptimizationConfig()
        resolved = resolve_constraints("save_results", plain_fn, config)
        assert resolved.has_side_effects is True
        assert resolved.source == "heuristic"

    def test_trust_analysis_suppresses_heuristic(self):
        config = OptimizationConfig(trust_analysis=True)
        resolved = resolve_constraints("save_results", plain_fn, config)
        assert resolved.has_side_effects is False

    def test_disable_parallelization_forces_sequential(self):
        config = OptimizationConfig(disable_parallelization=True)
        resolved = resolve_constraints("any_node", plain_fn, config)
        assert resolved.force_sequential is True

    def test_none_func_skips_decorator_check(self):
        config = OptimizationConfig()
        resolved = resolve_constraints("my_node", None, config)
        assert resolved.force_sequential is False

    def test_decorator_priority_over_config(self):
        config = OptimizationConfig()
        resolved = resolve_constraints("seq_node", sequential_fn, config)
        assert resolved.source == "decorator"

    def test_decorator_depends_on_merged(self):
        config = OptimizationConfig()
        resolved = resolve_constraints("dep_node", dependent_fn, config)
        assert "upstream" in resolved.explicit_dependencies


# -- apply_constraints_to_analysis -------------------------------------------


class TestApplyConstraintsToAnalysis:

    def test_builds_per_node_constraints(self):
        node_analyses = {"a": NodeAnalysis(name="a"), "b": NodeAnalysis(name="b")}
        node_funcs = {"a": plain_fn, "b": sequential_fn}
        config = OptimizationConfig()

        constraints, warnings = apply_constraints_to_analysis(node_analyses, node_funcs, config)

        assert "a" in constraints
        assert "b" in constraints
        assert constraints["b"].force_sequential is True

    def test_warns_for_side_effect_nodes(self):
        node_analyses = {"api_call": NodeAnalysis(name="api_call")}
        node_funcs = {"api_call": side_effect_fn}
        config = OptimizationConfig()

        constraints, warnings = apply_constraints_to_analysis(node_analyses, node_funcs, config)

        assert len(warnings) >= 1
        assert any("side effects" in w for w in warnings)

    def test_no_warning_for_sequential_side_effect(self):
        node_analyses = {"db_write": NodeAnalysis(name="db_write")}
        node_funcs = {"db_write": sequential_fn}
        config = OptimizationConfig()

        _, warnings = apply_constraints_to_analysis(node_analyses, node_funcs, config)

        side_effect_warnings = [w for w in warnings if "db_write" in w and "side effects" in w]
        assert len(side_effect_warnings) == 0


# -- merge_dependencies -------------------------------------------------------


class TestMergeDependencies:

    def test_merges_data_and_constraint_deps(self):
        data_deps = {"a": {"b"}, "c": set()}
        config = OptimizationConfig()
        resolved_a = resolve_constraints("a", plain_fn, config)
        resolved_c = resolve_constraints("c", dependent_fn, config)

        merged = merge_dependencies(data_deps, {"a": resolved_a, "c": resolved_c})

        assert "b" in merged["a"]
        assert "upstream" in merged["c"]

    def test_preserves_original_data_deps(self):
        data_deps = {"a": {"b"}}
        config = OptimizationConfig()
        resolved_a = resolve_constraints("a", plain_fn, config)

        merge_dependencies(data_deps, {"a": resolved_a})

        assert data_deps["a"] == {"b"}

    def test_adds_missing_nodes(self):
        data_deps = {"a": {"b"}}
        config = OptimizationConfig(explicit_dependencies={"new_node": {"a"}})
        resolved_new = resolve_constraints("new_node", plain_fn, config)

        merged = merge_dependencies(data_deps, {
            "a": resolve_constraints("a", plain_fn, config), "new_node": resolved_new
        })

        assert "new_node" in merged
        assert "a" in merged["new_node"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
