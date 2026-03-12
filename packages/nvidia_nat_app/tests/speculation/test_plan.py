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
"""Tests for plan_speculation(), SpeculationPlan, and helper functions."""

import pytest

from nat_app.graph.topology import find_router_chains
from nat_app.speculation.plan import SpeculationPlan
from nat_app.speculation.plan import partition_targets
from nat_app.speculation.plan import plan_speculation
from nat_app.speculation.safety import SpeculationSafetyConfig
from nat_app.speculation.safety import speculation_unsafe
from nat_app.speculation.strategies.router_branch import RouterBranchResolution

# -- Test functions ----------------------------------------------------------


def route_fn(state):
    return state.get("choice", "a")


def fn_a(state):
    state["a_out"] = "done"


def fn_b(state):
    state["b_out"] = "done"


def fn_c(state):
    state["c_out"] = state["a_out"]


def fn_merge(state):
    state["merged"] = True


@speculation_unsafe
def unsafe_fn(state):
    state["side_effect"] = True


# -- Basic planning ----------------------------------------------------------


class TestPlanSpeculationBasic:

    def test_no_routers_returns_empty(self):
        plans = plan_speculation(
            nodes={
                "a": fn_a, "b": fn_b
            },
            edges=[("a", "b")],
        )
        assert plans == []

    def test_single_router_produces_plan(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
        )
        assert len(plans) == 1
        plan = plans[0]
        assert plan.decision_node == "router"

    def test_plan_is_frozen(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
        )
        plan = plans[0]
        assert isinstance(plan, SpeculationPlan)
        with pytest.raises(AttributeError):
            plan.decision_node = "changed"

    def test_returns_speculation_plan_type(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
        )
        assert all(isinstance(p, SpeculationPlan) for p in plans)


# -- Targets and cancellation ------------------------------------------------


class TestTargetsAndCancellation:

    def test_targets_include_branch_nodes(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
        )
        plan = plans[0]
        assert "a" in plan.targets_to_launch
        assert "b" in plan.targets_to_launch

    def test_cancel_map_keyed_by_label(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
        )
        plan = plans[0]
        assert "b" in plan.resolution.cancel_map.get("left", frozenset())
        assert "a" in plan.resolution.cancel_map.get("right", frozenset())

    def test_merge_nodes_identified(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b, "merge": fn_merge
            },
            edges=[
                ("router", "a"),
                ("router", "b"),
                ("a", "merge"),
                ("b", "merge"),
            ],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
        )
        plan = plans[0]
        assert "merge" in plan.merge_nodes

    def test_merge_nodes_not_in_targets(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b, "merge": fn_merge
            },
            edges=[
                ("router", "a"),
                ("router", "b"),
                ("a", "merge"),
                ("b", "merge"),
            ],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
        )
        plan = plans[0]
        assert "merge" not in plan.targets_to_launch

    def test_merge_nodes_not_in_cancel_map_values(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b, "merge": fn_merge
            },
            edges=[
                ("router", "a"),
                ("router", "b"),
                ("a", "merge"),
                ("b", "merge"),
            ],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
        )
        plan = plans[0]
        for nodes_to_cancel in plan.resolution.cancel_map.values():
            assert "merge" not in nodes_to_cancel


# -- Safety configuration ---------------------------------------------------


class TestSafetyFiltering:

    def test_unsafe_nodes_excluded(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
            safety=SpeculationSafetyConfig(unsafe_nodes={"b"}),
        )
        plan = plans[0]
        assert "b" not in plan.targets_to_launch
        assert "b" in plan.excluded_nodes

    def test_unsafe_decorator_excluded(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": unsafe_fn
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
        )
        plan = plans[0]
        assert "b" not in plan.targets_to_launch
        assert "b" in plan.excluded_nodes

    def test_safe_overrides_restore_excluded(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": unsafe_fn
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
            safety=SpeculationSafetyConfig(safe_overrides={"b"}),
        )
        plan = plans[0]
        assert "b" in plan.targets_to_launch
        assert "b" not in plan.excluded_nodes

    def test_safe_overrides_beat_unsafe_nodes(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
            safety=SpeculationSafetyConfig(unsafe_nodes={"b"}, safe_overrides={"b"}),
        )
        plan = plans[0]
        assert "b" in plan.targets_to_launch

    def test_all_excluded_skips_router(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
            safety=SpeculationSafetyConfig(unsafe_nodes={"a", "b"}),
        )
        assert plans == []

    def test_excluded_not_in_cancel_map(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
            safety=SpeculationSafetyConfig(unsafe_nodes={"b"}),
        )
        plan = plans[0]
        for nodes_to_cancel in plan.resolution.cancel_map.values():
            assert "b" not in nodes_to_cancel


# -- Multi-router graphs -----------------------------------------------------


class TestMultiRouter:

    def test_multiple_routers_produce_multiple_plans(self):
        plans = plan_speculation(
            nodes={
                "r1": route_fn,
                "a": fn_a,
                "b": fn_b,
                "r2": route_fn,
                "c": fn_c,
                "merge": fn_merge,
            },
            edges=[
                ("r1", "a"),
                ("r1", "b"),
                ("a", "r2"),
                ("b", "r2"),
                ("r2", "c"),
                ("r2", "merge"),
            ],
            conditional_edges={
                "r1": {
                    "left": "a", "right": "b"
                },
                "r2": {
                    "x": "c", "y": "merge"
                },
            },
        )
        router_names = {p.decision_node for p in plans}
        assert "r1" in router_names


# -- Branch depth and metadata -----------------------------------------------


class TestMetadata:

    def test_max_branch_depth(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
        )
        plan = plans[0]
        assert plan.max_branch_depth >= 1

    def test_frozenset_types(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
        )
        plan = plans[0]
        assert isinstance(plan.targets_to_launch, frozenset)
        assert isinstance(plan.excluded_nodes, frozenset)
        assert isinstance(plan.merge_nodes, frozenset)
        for v in plan.resolution.cancel_map.values():
            assert isinstance(v, frozenset)


# -- label_map and chain_next -----------------------------------------------


class TestLabelMap:

    def test_label_map_populated(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
        )
        plan = plans[0]
        assert plan.resolution.label_map is not None
        assert plan.resolution.label_map["left"] == frozenset({"a"})
        assert plan.resolution.label_map["right"] == frozenset({"b"})

    def test_label_map_one_to_many(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b, "c": fn_c
            },
            edges=[("router", "a"), ("router", "b"), ("router", "c")],
            conditional_edges={"router": {
                "left": ["a", "b"], "right": "c"
            }},
        )
        plan = plans[0]
        assert plan.resolution.label_map is not None
        assert plan.resolution.label_map["left"] == frozenset({"a", "b"})
        assert plan.resolution.label_map["right"] == frozenset({"c"})

    def test_one_to_many_cancel_map(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b, "c": fn_c
            },
            edges=[("router", "a"), ("router", "b"), ("router", "c")],
            conditional_edges={"router": {
                "left": ["a", "b"], "right": "c"
            }},
        )
        plan = plans[0]
        left_cancel = plan.resolution.cancel_map.get("left", frozenset())
        assert "c" in left_cancel
        assert "a" not in left_cancel
        assert "b" not in left_cancel

        right_cancel = plan.resolution.cancel_map.get("right", frozenset())
        assert "a" in right_cancel
        assert "b" in right_cancel

    def test_label_map_none_without_conditional_edges(self):
        plans = plan_speculation(
            nodes={
                "a": fn_a, "b": fn_b
            },
            edges=[("a", "b")],
        )
        assert plans == []


class TestChainNext:

    def test_chain_next_populated(self):
        plans = plan_speculation(
            nodes={
                "r1": route_fn, "r2": route_fn, "a": fn_a, "b": fn_b
            },
            edges=[("r1", "r2"), ("r2", "a"), ("r2", "b")],
            conditional_edges={
                "r1": {
                    "pass": "r2"
                },
                "r2": {
                    "left": "a", "right": "b"
                },
            },
        )
        r1_plan = next(p for p in plans if p.decision_node == "r1")
        assert r1_plan.chain_next == "r2"

    def test_chain_next_none_for_terminal(self):
        plans = plan_speculation(
            nodes={
                "router": route_fn, "a": fn_a, "b": fn_b
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
        )
        plan = plans[0]
        assert plan.chain_next is None


# -- Helper functions -------------------------------------------------------


class TestGetCancelSet:

    def test_returns_cancel_set_for_label(self):
        resolution = RouterBranchResolution(
            cancel_map={
                "left": frozenset({"b"}), "right": frozenset({"a"})
            },
            label_map={
                "left": frozenset({"a"}), "right": frozenset({"b"})
            },
            all_targets=frozenset({"a", "b"}),
        )
        plan = SpeculationPlan(
            strategy="router_branch",
            decision_node="r",
            targets_to_launch=frozenset({"a", "b"}),
            excluded_nodes=frozenset(),
            resolution=resolution,
            merge_nodes=frozenset(),
            max_branch_depth=1,
            is_cycle_exit=False,
        )
        assert plan.resolution.get_cancel_set("left") == frozenset({"b"})
        assert plan.resolution.get_cancel_set("right") == frozenset({"a"})

    def test_returns_empty_for_unknown_label(self):
        resolution = RouterBranchResolution(
            cancel_map={"left": frozenset({"b"})},
            label_map=None,
            all_targets=frozenset({"a", "b"}),
        )
        plan = SpeculationPlan(
            strategy="router_branch",
            decision_node="r",
            targets_to_launch=frozenset({"a", "b"}),
            excluded_nodes=frozenset(),
            resolution=resolution,
            merge_nodes=frozenset(),
            max_branch_depth=1,
            is_cycle_exit=False,
        )
        assert plan.resolution.get_cancel_set("unknown") == frozenset()


class TestIsOnChosenPath:

    def test_chosen_node_is_on_path(self):
        resolution = RouterBranchResolution(
            cancel_map={
                "left": frozenset({"b"}), "right": frozenset({"a"})
            },
            label_map=None,
            all_targets=frozenset({"a", "b"}),
        )
        plan = SpeculationPlan(
            strategy="router_branch",
            decision_node="r",
            targets_to_launch=frozenset({"a", "b"}),
            excluded_nodes=frozenset(),
            resolution=resolution,
            merge_nodes=frozenset(),
            max_branch_depth=1,
            is_cycle_exit=False,
        )
        assert plan.resolution.is_on_chosen_path("a", "left") is True
        assert plan.resolution.is_on_chosen_path("b", "left") is False
        assert plan.resolution.is_on_chosen_path("b", "right") is True
        assert plan.resolution.is_on_chosen_path("a", "right") is False

    def test_node_not_in_targets(self):
        resolution = RouterBranchResolution(
            cancel_map={},
            label_map=None,
            all_targets=frozenset({"a"}),
        )
        plan = SpeculationPlan(
            strategy="router_branch",
            decision_node="r",
            targets_to_launch=frozenset({"a"}),
            excluded_nodes=frozenset(),
            resolution=resolution,
            merge_nodes=frozenset(),
            max_branch_depth=1,
            is_cycle_exit=False,
        )
        assert plan.resolution.is_on_chosen_path("unknown", "left") is False


class TestPartitionTargets:

    def test_all_immediate_without_chain(self):
        resolution = RouterBranchResolution(
            cancel_map={},
            label_map=None,
            all_targets=frozenset({"a", "b"}),
        )
        plan = SpeculationPlan(
            strategy="router_branch",
            decision_node="r",
            targets_to_launch=frozenset({"a", "b"}),
            excluded_nodes=frozenset(),
            resolution=resolution,
            merge_nodes=frozenset(),
            max_branch_depth=1,
            is_cycle_exit=False,
            chain_next=None,
        )
        immediate, deferred = partition_targets(plan)
        assert immediate == frozenset({"a", "b"})
        assert deferred == frozenset()

    def test_partition_with_chain_next(self):
        resolution = RouterBranchResolution(
            cancel_map={"pass": frozenset()},
            label_map=None,
            all_targets=frozenset({"r2", "a", "b"}),
        )
        plan = SpeculationPlan(
            strategy="router_branch",
            decision_node="r1",
            targets_to_launch=frozenset({"r2", "a", "b"}),
            excluded_nodes=frozenset(),
            resolution=resolution,
            merge_nodes=frozenset(),
            max_branch_depth=2,
            is_cycle_exit=False,
            chain_next="r2",
        )
        immediate, deferred = partition_targets(plan)
        assert "r2" in immediate


# -- find_router_chains ------------------------------------------------------


class TestFindRouterChains:

    def test_chain_detected(self):
        from nat_app.graph.topology import GraphTopology
        from nat_app.graph.topology import NodeType
        from nat_app.graph.topology import RouterInfo
        topology = GraphTopology(
            nodes={"r1", "r2", "a", "b"},
            edges=[("r1", "r2"), ("r2", "a"), ("r2", "b")],
            node_types={
                "r1": NodeType.ROUTER, "r2": NodeType.ROUTER, "a": NodeType.REGULAR, "b": NodeType.REGULAR
            },
            routers=[
                RouterInfo(node="r1", branches={"pass": ["r2"]}),
                RouterInfo(node="r2", branches={
                    "left": ["a"], "right": ["b"]
                }),
            ],
            cycles=[],
            parallelizable_regions=[],
            sequential_regions=[],
        )
        chains = find_router_chains(topology)
        assert len(chains) == 1
        assert chains[0] == ["r1", "r2"]

    def test_no_chain_standalone_routers(self):
        from nat_app.graph.topology import GraphTopology
        from nat_app.graph.topology import NodeType
        from nat_app.graph.topology import RouterInfo
        topology = GraphTopology(
            nodes={"r1", "r2", "a", "b", "c", "d"},
            edges=[("r1", "a"), ("r1", "b"), ("r2", "c"), ("r2", "d")],
            node_types={
                "r1": NodeType.ROUTER,
                "r2": NodeType.ROUTER,
                "a": NodeType.REGULAR,
                "b": NodeType.REGULAR,
                "c": NodeType.REGULAR,
                "d": NodeType.REGULAR,
            },
            routers=[
                RouterInfo(node="r1", branches={
                    "left": ["a"], "right": ["b"]
                }),
                RouterInfo(node="r2", branches={
                    "left": ["c"], "right": ["d"]
                }),
            ],
            cycles=[],
            parallelizable_regions=[],
            sequential_regions=[],
        )
        chains = find_router_chains(topology)
        assert chains == []


# -- Import from public API -------------------------------------------------


class TestPublicImports:

    def test_importable_from_nat_app(self):
        from nat_app import Resolution
        from nat_app import ResolutionPolicy
        from nat_app import RouterBranchStrategy
        from nat_app import SpeculationPlan as SP
        from nat_app import SpeculationPlanner
        from nat_app import partition_targets as pt
        from nat_app import plan_speculation as ps
        assert SP is SpeculationPlan
        assert ps is plan_speculation
        assert pt is partition_targets
        assert Resolution is not None
        assert ResolutionPolicy is not None
        assert RouterBranchStrategy is not None
        assert SpeculationPlanner is not None

    def test_importable_from_api(self):
        from nat_app.api import SpeculationPlan as SP
        from nat_app.api import plan_speculation as ps
        assert SP is SpeculationPlan
        assert ps is plan_speculation

    def test_importable_from_executors(self):
        from nat_app.executors import SpeculationPlan as SP
        from nat_app.executors import partition_targets as pt
        assert SP is SpeculationPlan
        assert pt is partition_targets

    def test_importable_from_speculation(self):
        from nat_app.speculation import Resolution
        from nat_app.speculation import ResolutionPolicy
        from nat_app.speculation import RouterBranchStrategy
        from nat_app.speculation import SpeculationPlan as SP
        from nat_app.speculation import SpeculationPlanner
        from nat_app.speculation import partition_targets as pt
        from nat_app.speculation import plan_speculation as ps
        assert SP is SpeculationPlan
        assert ps is plan_speculation
        assert pt is partition_targets
        assert Resolution is not None
        assert ResolutionPolicy is not None
        assert RouterBranchStrategy is not None
        assert SpeculationPlanner is not None


# -- resolve_chosen_label ----------------------------------------------------


class TestResolveChosenLabel:

    def test_returns_label_when_in_cancel_map(self):
        resolution = RouterBranchResolution(
            cancel_map={
                "left": frozenset({"b"}), "right": frozenset({"a"})
            },
            label_map={
                "left": frozenset({"a"}), "right": frozenset({"b"})
            },
            all_targets=frozenset({"a", "b"}),
        )
        plan = SpeculationPlan(
            strategy="router_branch",
            decision_node="r",
            targets_to_launch=frozenset({"a", "b"}),
            excluded_nodes=frozenset(),
            resolution=resolution,
            merge_nodes=frozenset(),
            max_branch_depth=1,
            is_cycle_exit=False,
        )
        assert plan.resolution._resolve_label("left") == "left"
        assert plan.resolution._resolve_label("right") == "right"

    def test_reverse_maps_target_to_label(self):
        resolution = RouterBranchResolution(
            cancel_map={
                "left": frozenset({"b"}), "right": frozenset({"a"})
            },
            label_map={
                "left": frozenset({"a"}), "right": frozenset({"b"})
            },
            all_targets=frozenset({"a", "b"}),
        )
        plan = SpeculationPlan(
            strategy="router_branch",
            decision_node="r",
            targets_to_launch=frozenset({"a", "b"}),
            excluded_nodes=frozenset(),
            resolution=resolution,
            merge_nodes=frozenset(),
            max_branch_depth=1,
            is_cycle_exit=False,
        )
        assert plan.resolution._resolve_label("a") == "left"
        assert plan.resolution._resolve_label("b") == "right"

    def test_fallback_when_no_label_map(self):
        resolution = RouterBranchResolution(
            cancel_map={},
            label_map=None,
            all_targets=frozenset({"a", "b"}),
        )
        plan = SpeculationPlan(
            strategy="router_branch",
            decision_node="r",
            targets_to_launch=frozenset({"a", "b"}),
            excluded_nodes=frozenset(),
            resolution=resolution,
            merge_nodes=frozenset(),
            max_branch_depth=1,
            is_cycle_exit=False,
        )
        assert plan.resolution._resolve_label("unknown_target") == "unknown_target"

    def test_fallback_when_target_not_in_any_label(self):
        resolution = RouterBranchResolution(
            cancel_map={"left": frozenset({"b"})},
            label_map={"left": frozenset({"a"})},
            all_targets=frozenset({"a", "b"}),
        )
        plan = SpeculationPlan(
            strategy="router_branch",
            decision_node="r",
            targets_to_launch=frozenset({"a", "b"}),
            excluded_nodes=frozenset(),
            resolution=resolution,
            merge_nodes=frozenset(),
            max_branch_depth=1,
            is_cycle_exit=False,
        )
        assert plan.resolution._resolve_label("not_mapped") == "not_mapped"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
