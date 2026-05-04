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
"""Tests for the hierarchical PriorityAssignmentStage."""

import pytest

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.graph.llm_detection import LLMCallInfo
from nat_app.graph.types import BranchGroup
from nat_app.graph.types import BranchGroupType
from nat_app.graph.types import CostMetric
from nat_app.graph.types import Graph
from nat_app.graph.types import PriorityLevel
from nat_app.graph.types import ProfiledNodeCost
from nat_app.stages.priority_assignment import PriorityAssignmentStage
from nat_app.stages.priority_assignment import SJFPriorityStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx_from_graph(
    graph: Graph,
    llm_analysis: dict[str, LLMCallInfo] | None = None,
) -> CompilationContext:
    ctx = CompilationContext(compiled=None)
    ctx.metadata["graph"] = graph
    ctx.metadata["llm_analysis"] = llm_analysis or {}
    return ctx


def _linear_graph(
    node_names: list[str],
    llm_analysis: dict[str, LLMCallInfo] | None = None,
) -> CompilationContext:
    """A -> B -> C linear chain."""
    g = Graph()
    for name in node_names:
        g.add_node(name)
    if node_names:
        g.entry_point = node_names[0]
        for i in range(len(node_names) - 1):
            g.add_edge(node_names[i], node_names[i + 1])
        g.terminal_nodes.add(node_names[-1])
    return _ctx_from_graph(g, llm_analysis)


def _router_graph(
    router: str,
    branches: dict[str, str],
    llm_analysis: dict[str, LLMCallInfo] | None = None,
    extra_edges: list[tuple[str, str]] | None = None,
) -> CompilationContext:
    """
    A conditional-router graph::

        entry -> router --cond--> branch_a
                        --cond--> branch_b
    """
    g = Graph()
    g.add_node(router)
    for target in set(branches.values()):
        g.add_node(target)
    g.add_conditional_edges(router, branches)
    g.entry_point = router
    g.terminal_nodes = set(branches.values())

    for src, tgt in (extra_edges or []):
        if not g.has_node(src):
            g.add_node(src)
        if not g.has_node(tgt):
            g.add_node(tgt)
        g.add_edge(src, tgt)

    return _ctx_from_graph(g, llm_analysis)


def _parallel_graph(
    source: str,
    targets: list[str],
    llm_analysis: dict[str, LLMCallInfo] | None = None,
) -> CompilationContext:
    """
    A parallel fan-out graph (unconditional edges)::

        source --> t1
               --> t2
               --> t3
    """
    g = Graph()
    g.add_node(source)
    for t in targets:
        g.add_node(t)
        g.add_edge(source, t)
    g.entry_point = source
    g.terminal_nodes = set(targets)
    return _ctx_from_graph(g, llm_analysis)


def _profiled_ctx_from_graph(
    graph: Graph,
    profiled: dict[str, ProfiledNodeCost],
    llm_analysis: dict[str, LLMCallInfo] | None = None,
) -> CompilationContext:
    """Build a CompilationContext with profiled_node_costs (and optionally llm_analysis)."""
    ctx = CompilationContext(compiled=None)
    ctx.metadata["graph"] = graph
    ctx.metadata["profiled_node_costs"] = profiled
    if llm_analysis is not None:
        ctx.metadata["llm_analysis"] = llm_analysis
    return ctx


def _router_profiled(
    router: str,
    branches: dict[str, str],
    profiled: dict[str, ProfiledNodeCost],
    llm_analysis: dict[str, LLMCallInfo] | None = None,
    extra_edges: list[tuple[str, str]] | None = None,
) -> CompilationContext:
    """Conditional-router graph backed by ProfiledNodeCost data."""
    g = Graph()
    g.add_node(router)
    for target in set(branches.values()):
        g.add_node(target)
    g.add_conditional_edges(router, branches)
    g.entry_point = router
    g.terminal_nodes = set(branches.values())

    for src, tgt in (extra_edges or []):
        if not g.has_node(src):
            g.add_node(src)
        if not g.has_node(tgt):
            g.add_node(tgt)
        g.add_edge(src, tgt)

    return _profiled_ctx_from_graph(g, profiled, llm_analysis)


def _nested_conditional_parallel_graph(llm_analysis: dict[str, LLMCallInfo], ) -> CompilationContext:
    """
    Topology::

        router_R --cond--> X --parallel--> X1, X2
                 --cond--> Y --parallel--> Y1, Y2
    """
    g = Graph()
    for n in ("router_R", "X", "Y", "X1", "X2", "Y1", "Y2"):
        g.add_node(n)
    g.add_conditional_edges("router_R", {"a": "X", "b": "Y"})
    g.add_edge("X", "X1")
    g.add_edge("X", "X2")
    g.add_edge("Y", "Y1")
    g.add_edge("Y", "Y2")
    g.entry_point = "router_R"
    g.terminal_nodes = {"X1", "X2", "Y1", "Y2"}
    return _ctx_from_graph(g, llm_analysis)


# ---------------------------------------------------------------------------
# Tests: basics
# ---------------------------------------------------------------------------


class TestBasics:

    def test_stage_name(self):
        assert PriorityAssignmentStage().name == "priority_assignment"

    def test_no_llm_analysis_noop(self):
        ctx = _linear_graph(["a", "b"])
        result = PriorityAssignmentStage().apply(ctx)
        assert result.metadata["graph"].get_node("a").priority is None

    def test_all_zero_calls_noop(self):
        ctx = _linear_graph(
            ["a", "b"],
            {
                "a": LLMCallInfo(call_count=0), "b": LLMCallInfo(call_count=0)
            },
        )
        result = PriorityAssignmentStage().apply(ctx)
        assert result.metadata["graph"].get_node("a").priority is None
        assert result.metadata["graph"].get_node("b").priority is None

    def test_user_priority_not_overwritten(self):
        ctx = _linear_graph(
            ["a", "b"],
            {
                "a": LLMCallInfo(call_count=3), "b": LLMCallInfo(call_count=1)
            },
        )
        ctx.metadata["graph"].get_node("a").priority = 0.42
        result = PriorityAssignmentStage().apply(ctx)
        assert result.metadata["graph"].get_node("a").priority == 0.42


# ---------------------------------------------------------------------------
# Tests: pluggable strategy
# ---------------------------------------------------------------------------


class TestPluggableStrategy:

    def test_custom_strategy_used(self):
        """Custom strategy overrides default SJF behavior."""

        class AllMediumStrategy:

            def assign_group_priorities(self, group: BranchGroup, ceiling: PriorityLevel | None) -> list[PriorityLevel]:
                return [PriorityLevel.MEDIUM] * len(group.node_names)

        stage = PriorityAssignmentStage(strategy=AllMediumStrategy())
        ctx = _router_graph(
            "router",
            {
                "a": "fast", "b": "slow"
            },
            {
                "router": LLMCallInfo(call_count=0),
                "fast": LLMCallInfo(call_count=1),
                "slow": LLMCallInfo(call_count=10),
            },
        )
        result = stage.apply(ctx)
        g = result.metadata["graph"]
        assert g.get_node("fast").priority == PriorityLevel.MEDIUM.value
        assert g.get_node("slow").priority == PriorityLevel.MEDIUM.value

    def test_default_strategy_preserves_sjf_behavior(self):
        """PriorityAssignmentStage() with no strategy uses SJF and preserves behavior."""
        stage = PriorityAssignmentStage()
        ctx = _router_graph(
            "router",
            {
                "a": "fast", "b": "slow"
            },
            {
                "router": LLMCallInfo(call_count=0),
                "fast": LLMCallInfo(call_count=1),
                "slow": LLMCallInfo(call_count=10),
            },
        )
        result = stage.apply(ctx)
        g = result.metadata["graph"]
        assert g.get_node("fast").priority == PriorityLevel.HIGH.value
        assert g.get_node("slow").priority == PriorityLevel.LOW.value

    def test_explicit_sjf_strategy_same_as_default(self):
        """Passing SJFPriorityStrategy() explicitly produces same result as default."""
        default_stage = PriorityAssignmentStage()
        explicit_stage = PriorityAssignmentStage(strategy=SJFPriorityStrategy())
        ctx = _router_graph(
            "router",
            {
                "a": "fast", "b": "slow"
            },
            {
                "router": LLMCallInfo(call_count=0),
                "fast": LLMCallInfo(call_count=1),
                "slow": LLMCallInfo(call_count=10),
            },
        )
        default_result = default_stage.apply(ctx)
        explicit_result = explicit_stage.apply(ctx)
        g_default = default_result.metadata["graph"]
        g_explicit = explicit_result.metadata["graph"]
        assert g_default.get_node("fast").priority == g_explicit.get_node("fast").priority
        assert g_default.get_node("slow").priority == g_explicit.get_node("slow").priority


# ---------------------------------------------------------------------------
# Tests: parallel fan-out strategies
# ---------------------------------------------------------------------------


class TestParallelFanOut:

    def test_default_uniform_medium(self):
        """All parallel branch nodes get uniform MEDIUM priority."""
        stage = PriorityAssignmentStage()
        ctx = _parallel_graph(
            "src",
            ["w1", "w2"],
            {
                "src": LLMCallInfo(call_count=0),
                "w1": LLMCallInfo(call_count=1),
                "w2": LLMCallInfo(call_count=5),
            },
        )
        result = stage.apply(ctx)
        g = result.metadata["graph"]
        assert g.get_node("w1").priority == PriorityLevel.MEDIUM.value
        assert g.get_node("w2").priority == PriorityLevel.MEDIUM.value


# ---------------------------------------------------------------------------
# Tests: conditional router end-to-end
# ---------------------------------------------------------------------------


class TestConditionalRouterEndToEnd:

    def test_fast_branch_gets_high_slow_gets_low(self):
        """End-to-end: conditional router with large cost spread -> three tiers."""
        stage = PriorityAssignmentStage()
        ctx = _router_graph(
            "router",
            {
                "a": "fast", "b": "medium_path", "c": "slow"
            },
            {
                "router": LLMCallInfo(call_count=0),
                "fast": LLMCallInfo(call_count=1),
                "medium_path": LLMCallInfo(call_count=3),
                "slow": LLMCallInfo(call_count=10),
            },
        )
        ctx.metadata["graph"].add_node("medium_path")
        ctx.metadata["graph"].add_conditional_edges("router", {"a": "fast", "b": "medium_path", "c": "slow"})

        result = stage.apply(ctx)
        g = result.metadata["graph"]

        assert g.get_node("fast").priority == PriorityLevel.HIGH.value
        assert g.get_node("medium_path").priority == PriorityLevel.MEDIUM.value
        assert g.get_node("slow").priority == PriorityLevel.LOW.value

    def test_two_tier_conditional(self):
        """Conditional router where ratio is between 1.5 and 3.0."""
        stage = PriorityAssignmentStage()
        ctx = _router_graph(
            "router",
            {
                "a": "fast", "b": "slow"
            },
            {
                "router": LLMCallInfo(call_count=0),
                "fast": LLMCallInfo(call_count=2),
                "slow": LLMCallInfo(call_count=5),
            },
        )

        result = stage.apply(ctx)
        g = result.metadata["graph"]
        assert g.get_node("fast").priority == PriorityLevel.HIGH.value
        assert g.get_node("slow").priority == PriorityLevel.MEDIUM.value

    def test_homogeneous_conditional(self):
        """Conditional router where both branches are similar -> all MEDIUM."""
        stage = PriorityAssignmentStage()
        ctx = _router_graph(
            "router",
            {
                "a": "branch_a", "b": "branch_b"
            },
            {
                "router": LLMCallInfo(call_count=0),
                "branch_a": LLMCallInfo(call_count=3),
                "branch_b": LLMCallInfo(call_count=4),
            },
        )

        result = stage.apply(ctx)
        g = result.metadata["graph"]
        assert g.get_node("branch_a").priority == PriorityLevel.MEDIUM.value
        assert g.get_node("branch_b").priority == PriorityLevel.MEDIUM.value


# ---------------------------------------------------------------------------
# Tests: linear-group end-to-end
# ---------------------------------------------------------------------------


class TestLinearEndToEnd:

    def test_linear_three_tier(self):
        """Linear chain with wide cost spread gets tiered priorities."""
        stage = PriorityAssignmentStage()
        ctx = _linear_graph(
            ["light", "mid", "heavy"],
            {
                "light": LLMCallInfo(call_count=1),
                "mid": LLMCallInfo(call_count=3),
                "heavy": LLMCallInfo(call_count=10),
            },
        )

        result = stage.apply(ctx)
        g = result.metadata["graph"]
        assert g.get_node("light").priority == PriorityLevel.HIGH.value
        assert g.get_node("mid").priority == PriorityLevel.MEDIUM.value
        assert g.get_node("heavy").priority == PriorityLevel.LOW.value


# ---------------------------------------------------------------------------
# Tests: BranchGroup dataclass
# ---------------------------------------------------------------------------


class TestBranchGroup:

    def test_defaults(self):
        bg = BranchGroup(name="test", group_type=BranchGroupType.CONDITIONAL)
        assert bg.node_names == []
        assert bg.subtree_costs == []
        assert bg.priorities == []

    def test_populated(self):
        bg = BranchGroup(
            name="router:r",
            group_type=BranchGroupType.CONDITIONAL,
            node_names=["a", "b"],
            subtree_costs=[1, 5],
            priorities=[PriorityLevel.HIGH, PriorityLevel.LOW],
        )
        assert bg.name == "router:r"
        assert len(bg.node_names) == 2


# ---------------------------------------------------------------------------
# Tests: PriorityLevel enum
# ---------------------------------------------------------------------------


class TestPriorityLevel:

    def test_values(self):
        assert PriorityLevel.HIGH.value == 1.0
        assert PriorityLevel.MEDIUM.value == 0.5
        assert PriorityLevel.LOW.value == 0.1

    def test_ordering(self):
        assert PriorityLevel.HIGH.value > PriorityLevel.MEDIUM.value > PriorityLevel.LOW.value


# ---------------------------------------------------------------------------
# Tests: profiled cost path (public apply() tests only)
# ---------------------------------------------------------------------------


class TestProfiledCostPath:

    def test_custom_callable_cost_function(self):
        """Custom cost_fn computes a weighted blend of profiled fields."""
        g = Graph()
        g.add_node("fast")
        g.add_node("slow")
        g.add_node("router")
        g.add_conditional_edges("router", {"a": "fast", "b": "slow"})
        g.entry_point = "router"
        g.terminal_nodes = {"fast", "slow"}

        profiled = {
            "router": ProfiledNodeCost(),
            "fast": ProfiledNodeCost(total_prompt_tokens=50, total_completion_tokens=20),
            "slow": ProfiledNodeCost(total_prompt_tokens=200, total_completion_tokens=100),
        }

        custom_fn = lambda c: 0.7 * c.total_prompt_tokens + 0.3 * c.total_completion_tokens  # noqa: E731
        stage = PriorityAssignmentStage(cost_fn=custom_fn)
        ctx = _profiled_ctx_from_graph(g, profiled)
        result = stage.apply(ctx)
        rg = result.metadata["graph"]

        fast_expected = 0.7 * 50 + 0.3 * 20  # 41.0
        slow_expected = 0.7 * 200 + 0.3 * 100  # 170.0
        assert slow_expected / fast_expected > 3.0

        assert rg.get_node("fast").priority == PriorityLevel.HIGH.value
        assert rg.get_node("slow").priority == PriorityLevel.LOW.value

    def test_conditional_router_end_to_end_with_profiled_data(self):
        """Full pipeline with profiled subtree_time_ms on a conditional router."""
        profiled = {
            "router": ProfiledNodeCost(),
            "fast": ProfiledNodeCost(subtree_time_ms=100.0),
            "slow": ProfiledNodeCost(subtree_time_ms=1000.0),
        }
        ctx = _router_profiled("router", {"a": "fast", "b": "slow"}, profiled)

        stage = PriorityAssignmentStage()
        result = stage.apply(ctx)
        rg = result.metadata["graph"]

        assert rg.get_node("fast").priority == PriorityLevel.HIGH.value
        assert rg.get_node("slow").priority == PriorityLevel.LOW.value

    def test_parallel_fan_out_with_profiled_data(self):
        """Parallel fan-out with profiled data applies uniform MEDIUM by default."""
        g = Graph()
        g.add_node("src")
        for t in ["w1", "w2"]:
            g.add_node(t)
            g.add_edge("src", t)
        g.entry_point = "src"
        g.terminal_nodes = {"w1", "w2"}

        profiled = {
            "src": ProfiledNodeCost(),
            "w1": ProfiledNodeCost(subtree_time_ms=100.0),
            "w2": ProfiledNodeCost(subtree_time_ms=500.0),
        }
        ctx = _profiled_ctx_from_graph(g, profiled)

        stage = PriorityAssignmentStage()
        result = stage.apply(ctx)
        rg = result.metadata["graph"]
        assert rg.get_node("w1").priority == PriorityLevel.MEDIUM.value
        assert rg.get_node("w2").priority == PriorityLevel.MEDIUM.value

    def test_empty_profiled_falls_back_to_llm_analysis(self):
        """Empty profiled_node_costs dict falls through to llm_analysis."""
        ctx = _linear_graph(
            ["light", "heavy"],
            {
                "light": LLMCallInfo(call_count=1), "heavy": LLMCallInfo(call_count=5)
            },
        )
        ctx.metadata["profiled_node_costs"] = {}

        stage = PriorityAssignmentStage()
        result = stage.apply(ctx)
        rg = result.metadata["graph"]
        assert rg.get_node("light").priority == PriorityLevel.HIGH.value
        assert rg.get_node("heavy").priority == PriorityLevel.LOW.value

    def test_no_data_at_all_is_noop(self):
        """When neither profiled nor llm_analysis have data, no priorities set."""
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.entry_point = "a"
        g.terminal_nodes = {"b"}
        ctx = CompilationContext(compiled=None)
        ctx.metadata["graph"] = g

        stage = PriorityAssignmentStage()
        result = stage.apply(ctx)
        assert result.metadata["graph"].get_node("a").priority is None
        assert result.metadata["graph"].get_node("b").priority is None

    def test_user_priority_not_overwritten_with_profiled_data(self):
        """Profiled path still respects user-set priority."""
        g = Graph()
        g.add_node("fast")
        g.add_node("slow")
        g.add_node("router")
        g.add_conditional_edges("router", {"a": "fast", "b": "slow"})
        g.entry_point = "router"
        g.terminal_nodes = {"fast", "slow"}
        g.get_node("fast").priority = 0.42

        profiled = {
            "router": ProfiledNodeCost(),
            "fast": ProfiledNodeCost(subtree_time_ms=50.0),
            "slow": ProfiledNodeCost(subtree_time_ms=500.0),
        }
        ctx = _profiled_ctx_from_graph(g, profiled)

        stage = PriorityAssignmentStage()
        result = stage.apply(ctx)
        assert result.metadata["graph"].get_node("fast").priority == 0.42


# ---------------------------------------------------------------------------
# Tests: ProfiledNodeCost and CostMetric types
# ---------------------------------------------------------------------------


class TestProfiledNodeCost:

    def test_defaults(self):
        c = ProfiledNodeCost()
        assert c.llm_call_count == 0
        assert c.total_latency_ms == 0.0
        assert c.total_prompt_tokens == 0
        assert c.total_completion_tokens == 0
        assert c.total_tokens == 0
        assert c.self_time_ms == 0.0
        assert c.subtree_time_ms == 0.0

    def test_frozen(self):
        c = ProfiledNodeCost(llm_call_count=3)
        with pytest.raises(AttributeError):
            c.llm_call_count = 5

    def test_populated(self):
        c = ProfiledNodeCost(
            llm_call_count=2,
            total_latency_ms=150.0,
            total_prompt_tokens=100,
            total_completion_tokens=50,
            total_tokens=150,
            self_time_ms=120.0,
            subtree_time_ms=200.0,
        )
        assert c.llm_call_count == 2
        assert c.total_tokens == 150
        assert c.subtree_time_ms == 200.0


class TestCostMetric:

    def test_values(self):
        assert CostMetric.LLM_CALLS.value == "llm_calls"
        assert CostMetric.WALL_CLOCK_MS.value == "wall_clock_ms"
        assert CostMetric.PROMPT_TOKENS.value == "prompt_tokens"
        assert CostMetric.COMPLETION_TOKENS.value == "completion_tokens"
        assert CostMetric.TOTAL_TOKENS.value == "total_tokens"
        assert CostMetric.SUBTREE_TIME.value == "subtree_time"

    def test_all_members_in_cost_metric_info(self):
        from nat_app.stages.priority_assignment import _COST_METRIC_INFO
        for metric in CostMetric:
            assert metric in _COST_METRIC_INFO, f"{metric} not in _COST_METRIC_INFO"


# ---------------------------------------------------------------------------
# Tests: hierarchical ceiling propagation
# ---------------------------------------------------------------------------


class TestHierarchicalPriority:

    def test_parallel_inherits_ceiling_from_conditional_parent(self):
        """Parallel children of a HIGH branch get HIGH, LOW branch get LOW."""
        ctx = _nested_conditional_parallel_graph({
            "router_R": LLMCallInfo(call_count=0),
            "X": LLMCallInfo(call_count=1),
            "Y": LLMCallInfo(call_count=10),
            "X1": LLMCallInfo(call_count=2),
            "X2": LLMCallInfo(call_count=3),
            "Y1": LLMCallInfo(call_count=4),
            "Y2": LLMCallInfo(call_count=5),
        })
        stage = PriorityAssignmentStage()
        result = stage.apply(ctx)
        rg = result.metadata["graph"]

        assert rg.get_node("X").priority == PriorityLevel.HIGH.value
        assert rg.get_node("Y").priority == PriorityLevel.LOW.value

        assert rg.get_node("X1").priority == PriorityLevel.HIGH.value
        assert rg.get_node("X2").priority == PriorityLevel.HIGH.value

        assert rg.get_node("Y1").priority == PriorityLevel.LOW.value
        assert rg.get_node("Y2").priority == PriorityLevel.LOW.value

    def test_nested_conditional_applies_ceiling(self):
        """Nested conditional under a HIGH branch caps tiers at HIGH ceiling.

        Topology::

            router_R --cond--> X (HIGH) --> router_S --cond--> S1 (cheap), S2 (expensive)
                     --cond--> Y (LOW)

        X subtree = 1 + max(1, 2) = 3, Y subtree = 20, ratio = 6.67 -> three tiers.
        S1/S2 ratio = 2/1 = 2.0 -> two active tiers.
        S1 should get HIGH (ceiling), S2 should get MEDIUM (step below ceiling).
        """
        g = Graph()
        for n in ("router_R", "X", "Y", "router_S", "S1", "S2"):
            g.add_node(n)
        g.add_conditional_edges("router_R", {"a": "X", "b": "Y"})
        g.add_edge("X", "router_S")
        g.add_conditional_edges("router_S", {"c": "S1", "d": "S2"})
        g.entry_point = "router_R"
        g.terminal_nodes = {"Y", "S1", "S2"}
        la = {
            "router_R": LLMCallInfo(call_count=0),
            "X": LLMCallInfo(call_count=1),
            "Y": LLMCallInfo(call_count=20),
            "router_S": LLMCallInfo(call_count=0),
            "S1": LLMCallInfo(call_count=1),
            "S2": LLMCallInfo(call_count=2),
        }
        ctx = _ctx_from_graph(g, la)

        stage = PriorityAssignmentStage()
        result = stage.apply(ctx)
        rg = result.metadata["graph"]

        assert rg.get_node("X").priority == PriorityLevel.HIGH.value
        assert rg.get_node("Y").priority == PriorityLevel.LOW.value
        assert rg.get_node("S1").priority == PriorityLevel.HIGH.value
        assert rg.get_node("S2").priority == PriorityLevel.MEDIUM.value

    def test_nested_conditional_under_low_ceiling_collapses(self):
        """Nested conditional under a LOW branch collapses all tiers to LOW.

        Topology::

            router_R --cond--> X (HIGH)
                     --cond--> Y (LOW) --> router_S --cond--> S1 (cheap), S2 (expensive)

        Both S1 and S2 should get LOW (ceiling=LOW collapses everything).
        """
        g = Graph()
        for n in ("router_R", "X", "Y", "router_S", "S1", "S2"):
            g.add_node(n)
        g.add_conditional_edges("router_R", {"a": "X", "b": "Y"})
        g.add_edge("Y", "router_S")
        g.add_conditional_edges("router_S", {"c": "S1", "d": "S2"})
        g.entry_point = "router_R"
        g.terminal_nodes = {"X", "S1", "S2"}
        la = {
            "router_R": LLMCallInfo(call_count=0),
            "X": LLMCallInfo(call_count=1),
            "Y": LLMCallInfo(call_count=10),
            "router_S": LLMCallInfo(call_count=0),
            "S1": LLMCallInfo(call_count=1),
            "S2": LLMCallInfo(call_count=10),
        }
        ctx = _ctx_from_graph(g, la)

        stage = PriorityAssignmentStage()
        result = stage.apply(ctx)
        rg = result.metadata["graph"]

        assert rg.get_node("Y").priority == PriorityLevel.LOW.value
        assert rg.get_node("S1").priority == PriorityLevel.LOW.value
        assert rg.get_node("S2").priority == PriorityLevel.LOW.value

    def test_deep_nesting_three_levels(self):
        """Conditional -> parallel -> conditional propagates ceiling through all levels.

        Topology::

            router_R --cond--> X (HIGH) --parallel--> X1, router_S --cond--> S1 (cheap), S2 (expensive)
                     --cond--> Y (LOW)

        X subtree = 1 + 2 + max(1, 2) = 5, Y subtree = 30, ratio = 6.0 -> three tiers.
        X1 inherits HIGH, router_S children get ceiling from X (HIGH).
        S1/S2 ratio = 2/1 = 2.0 -> two active tiers.
        S1 (cheap) -> HIGH, S2 (expensive) -> MEDIUM.
        """
        g = Graph()
        for n in ("router_R", "X", "Y", "X1", "router_S", "S1", "S2"):
            g.add_node(n)
        g.add_conditional_edges("router_R", {"a": "X", "b": "Y"})
        g.add_edge("X", "X1")
        g.add_edge("X", "router_S")
        g.add_conditional_edges("router_S", {"c": "S1", "d": "S2"})
        g.entry_point = "router_R"
        g.terminal_nodes = {"Y", "X1", "S1", "S2"}
        la = {
            "router_R": LLMCallInfo(call_count=0),
            "X": LLMCallInfo(call_count=1),
            "Y": LLMCallInfo(call_count=30),
            "X1": LLMCallInfo(call_count=2),
            "router_S": LLMCallInfo(call_count=0),
            "S1": LLMCallInfo(call_count=1),
            "S2": LLMCallInfo(call_count=2),
        }
        ctx = _ctx_from_graph(g, la)

        stage = PriorityAssignmentStage()
        result = stage.apply(ctx)
        rg = result.metadata["graph"]

        assert rg.get_node("X").priority == PriorityLevel.HIGH.value
        assert rg.get_node("Y").priority == PriorityLevel.LOW.value
        assert rg.get_node("X1").priority == PriorityLevel.HIGH.value
        assert rg.get_node("S1").priority == PriorityLevel.HIGH.value
        assert rg.get_node("S2").priority == PriorityLevel.MEDIUM.value

    def test_intermediate_linear_nodes(self):
        """Intermediate nodes between parent group and child group are traversed.

        Topology::

            router_R --cond--> X (HIGH) --> B --> C --parallel--> C1, C2
                     --cond--> Y (LOW)

        B and C are intermediate.  Parallel group at C should inherit HIGH from X.
        """
        g = Graph()
        for n in ("router_R", "X", "Y", "B", "C", "C1", "C2"):
            g.add_node(n)
        g.add_conditional_edges("router_R", {"a": "X", "b": "Y"})
        g.add_edge("X", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "C1")
        g.add_edge("C", "C2")
        g.entry_point = "router_R"
        g.terminal_nodes = {"Y", "C1", "C2"}
        la = {
            "router_R": LLMCallInfo(call_count=0),
            "X": LLMCallInfo(call_count=1),
            "Y": LLMCallInfo(call_count=10),
            "B": LLMCallInfo(call_count=0),
            "C": LLMCallInfo(call_count=0),
            "C1": LLMCallInfo(call_count=2),
            "C2": LLMCallInfo(call_count=3),
        }
        ctx = _ctx_from_graph(g, la)

        stage = PriorityAssignmentStage()
        result = stage.apply(ctx)
        rg = result.metadata["graph"]

        assert rg.get_node("X").priority == PriorityLevel.HIGH.value
        assert rg.get_node("C1").priority == PriorityLevel.HIGH.value
        assert rg.get_node("C2").priority == PriorityLevel.HIGH.value

    def test_post_merge_no_ceiling(self):
        """Groups after a merge point (multiple predecessors) get no ceiling.

        Topology::

            router_R --cond--> X (HIGH)
                     --cond--> Y (LOW)
            X --> Join
            Y --> Join
            Join --parallel--> J1, J2

        Join has two predecessors from different branches, so parallel:Join
        gets no ceiling -> defaults to MEDIUM.
        """
        g = Graph()
        for n in ("router_R", "X", "Y", "Join", "J1", "J2"):
            g.add_node(n)
        g.add_conditional_edges("router_R", {"a": "X", "b": "Y"})
        g.add_edge("X", "Join")
        g.add_edge("Y", "Join")
        g.add_edge("Join", "J1")
        g.add_edge("Join", "J2")
        g.entry_point = "router_R"
        g.terminal_nodes = {"J1", "J2"}
        la = {
            "router_R": LLMCallInfo(call_count=0),
            "X": LLMCallInfo(call_count=1),
            "Y": LLMCallInfo(call_count=10),
            "Join": LLMCallInfo(call_count=0),
            "J1": LLMCallInfo(call_count=2),
            "J2": LLMCallInfo(call_count=3),
        }
        ctx = _ctx_from_graph(g, la)

        stage = PriorityAssignmentStage()
        result = stage.apply(ctx)
        rg = result.metadata["graph"]

        assert rg.get_node("J1").priority == PriorityLevel.MEDIUM.value
        assert rg.get_node("J2").priority == PriorityLevel.MEDIUM.value

    def test_top_level_parallel_unchanged(self):
        """Top-level parallel group (no parent) defaults to MEDIUM as before."""
        ctx = _parallel_graph(
            "src",
            ["w1", "w2"],
            {
                "src": LLMCallInfo(call_count=0),
                "w1": LLMCallInfo(call_count=1),
                "w2": LLMCallInfo(call_count=5),
            },
        )
        stage = PriorityAssignmentStage()
        result = stage.apply(ctx)
        rg = result.metadata["graph"]
        assert rg.get_node("w1").priority == PriorityLevel.MEDIUM.value
        assert rg.get_node("w2").priority == PriorityLevel.MEDIUM.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
