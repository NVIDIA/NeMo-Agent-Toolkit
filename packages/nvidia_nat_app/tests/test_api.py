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
"""Tests for the nat_app.api embeddable functions."""

import pytest

from nat_app.api import analyze_function
from nat_app.api import benchmark
from nat_app.api import classify_edge
from nat_app.api import find_parallel_stages
from nat_app.api import quick_optimize
from nat_app.api import speculative_opportunities

# -- Test functions (defined in a source file so inspect.getsource works) --


def step_a(state):
    state["ticker"] = "MSFT"
    state["thesis"] = "growth"


def step_b(state):
    state["revenue"] = "245B"
    state["target"] = state["ticker"]


def step_c(state):
    state["support"] = 400
    state["rsi"] = 58


def step_d(state):
    state["model"] = state["revenue"] + "_model"
    state["based_on"] = state["support"]


def step_e(state):
    state["recommendation"] = state["model"]
    state["risk"] = state["support"]


def step_return_dict(state):
    return {"ticker": "MSFT", "thesis": "growth"}


def step_reads_ticker(state):
    state["revenue"] = state["ticker"]


# -- quick_optimize tests ---------------------------------------------------


class TestQuickOptimize:

    def test_basic_parallel_detection(self):
        stages = quick_optimize(
            nodes={
                "a": step_a, "b": step_b, "c": step_c, "d": step_d
            },
            edges=[("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
        )
        assert len(stages) > 0
        assert any(len(s) > 1 for s in stages), "Should detect parallel stage"

    def test_sequential_chain(self):
        stages = quick_optimize(
            nodes={
                "a": step_a, "b": step_b, "d": step_d
            },
            edges=[("a", "b"), ("b", "d")],
        )
        assert len(stages) >= 2

    def test_returns_list_of_sets(self):
        stages = quick_optimize(
            nodes={
                "a": step_a, "b": step_b
            },
            edges=[("a", "b")],
        )
        assert isinstance(stages, list)
        for s in stages:
            assert isinstance(s, set)

    def test_all_nodes_present(self):
        nodes = {"a": step_a, "b": step_b, "c": step_c}
        stages = quick_optimize(
            nodes=nodes,
            edges=[("a", "b"), ("a", "c")],
        )
        all_scheduled = set()
        for s in stages:
            all_scheduled.update(s)
        assert all_scheduled == set(nodes.keys())


# -- analyze_function tests -------------------------------------------------


class TestAnalyzeFunction:

    def test_reads_detected(self):
        info = analyze_function(step_b)
        assert "ticker" in info["reads"]

    def test_writes_detected(self):
        info = analyze_function(step_a)
        assert "ticker" in info["writes"]

    def test_confidence_full_for_clean_function(self):
        info = analyze_function(step_a)
        assert info["confidence"] == "full"

    def test_returns_plain_types(self):
        info = analyze_function(step_a)
        assert isinstance(info["reads"], set)
        assert isinstance(info["writes"], set)
        assert info["confidence"] in ("full", "partial", "opaque")
        assert isinstance(info["warnings"], list)
        assert isinstance(info["source_available"], bool)

    def test_source_available(self):
        info = analyze_function(step_a)
        assert info["source_available"] is True

    def test_confidence_opaque_when_source_unavailable(self):
        info = analyze_function(len)
        assert info["confidence"] == "opaque"
        assert info["source_available"] is False

    def test_confidence_partial_when_dynamic_keys(self):

        def fn_with_dynamic_key(state):
            key = some_func()  # noqa: F821
            return {key: "val"}

        info = analyze_function(fn_with_dynamic_key)
        assert info["confidence"] == "partial"

    def test_confidence_partial_when_writes_empty_but_warnings(self):

        def fn_no_params():
            return {}

        info = analyze_function(fn_no_params)
        assert info["confidence"] == "partial"
        assert not info["writes"]
        assert info["warnings"]

    def test_confidence_partial_for_exec(self):

        def exec_call(state):
            exec("x=1")  # noqa: S102
            return {}

        info = analyze_function(exec_call)
        assert info["confidence"] == "partial"

    def test_confidence_partial_for_closure_mutation(self):
        outer = {}

        def closure_mutation(state):
            outer["x"] = state.get("input", 1)
            return {}

        info = analyze_function(closure_mutation)
        assert info["confidence"] == "partial"

    def test_confidence_partial_for_global_mutable(self):

        def global_mutable(state):
            module_var["x"] = state.get("input", 1)  # noqa: F821
            return {}

        info = analyze_function(global_mutable)
        assert info["confidence"] == "partial"

    def test_return_dict_writes_included(self):
        info = analyze_function(step_return_dict)
        assert "ticker" in info["writes"]
        assert "thesis" in info["writes"]


# -- classify_edge tests ----------------------------------------------------


class TestClassifyEdge:

    def test_necessary_edge(self):
        result = classify_edge(step_a, step_b)
        assert result == "necessary"

    def test_unnecessary_edge(self):
        result = classify_edge(step_b, step_c)
        assert result == "unnecessary"

    def test_returns_string(self):
        result = classify_edge(step_a, step_b)
        assert result in ("necessary", "unnecessary", "unknown")

    def test_necessary_edge_return_dict_writes(self):
        result = classify_edge(step_return_dict, step_reads_ticker)
        assert result == "necessary"


# -- find_parallel_stages tests ---------------------------------------------


class TestFindParallelStages:

    def test_returns_stages_and_info(self):
        stages, info = find_parallel_stages(
            nodes={"a": step_a, "b": step_b, "c": step_c},
            edges=[("a", "b"), ("a", "c")],
        )
        assert isinstance(stages, list)
        assert isinstance(info, dict)

    def test_info_has_reads_writes(self):
        _, info = find_parallel_stages(
            nodes={"a": step_a, "b": step_b},
            edges=[("a", "b")],
        )
        for name, analysis in info.items():
            assert "reads" in analysis
            assert "writes" in analysis
            assert "confidence" in analysis

    def test_parallel_detected(self):
        stages, _ = find_parallel_stages(
            nodes={"a": step_a, "b": step_b, "c": step_c, "d": step_d},
            edges=[("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
        )
        parallel = [s for s in stages if len(s) > 1]
        assert len(parallel) >= 1


# -- benchmark tests ---------------------------------------------------------


def route_fn(state):
    return state.get("choice", "a")


def fn_merge(state):
    state["merged"] = True


class TestBenchmark:

    async def test_returns_expected_keys(self):

        async def execute_node(name, state):
            return {f"{name}_done": True, **state}

        result = await benchmark(
            nodes={
                "a": step_a, "b": step_b
            },
            edges=[("a", "b")],
            execute_node=execute_node,
            n_runs=1,
        )
        assert "sequential_ms" in result
        assert "parallel_ms" in result
        assert "parallel_speedup" in result
        assert "stages" in result
        assert "n_runs" in result
        assert "outputs" in result
        assert result["n_runs"] == 1

    async def test_output_propagation(self):

        async def execute_node(name, state):
            state[f"{name}_done"] = True
            return state

        result = await benchmark(
            nodes={
                "a": step_a, "b": step_b
            },
            edges=[("a", "b")],
            execute_node=execute_node,
            n_runs=1,
        )
        assert "a_done" in result["outputs"]["sequential"]
        assert "b_done" in result["outputs"]["sequential"]

    async def test_custom_strategy(self):

        async def execute_node(name, state):
            return state

        async def my_strategy(state):
            state["strategy_ran"] = True
            return state

        result = await benchmark(
            nodes={
                "a": step_a, "b": step_b
            },
            edges=[("a", "b")],
            execute_node=execute_node,
            strategies={"custom": my_strategy},
            n_runs=1,
        )
        assert "custom" in result["strategies"]
        strat = result["strategies"]["custom"]
        assert "median_ms" in strat
        assert "speedup_vs_sequential" in strat
        assert "speedup_vs_parallel" in strat
        assert result["outputs"]["custom"]["strategy_ran"] is True

    async def test_stages_populated(self):

        async def execute_node(name, state):
            return state

        result = await benchmark(
            nodes={
                "a": step_a, "b": step_b, "c": step_c
            },
            edges=[("a", "b"), ("a", "c")],
            execute_node=execute_node,
            n_runs=1,
        )
        assert isinstance(result["stages"], list)
        assert len(result["stages"]) > 0

    async def test_custom_strategy_returns_none_uses_state(self):

        async def execute_node(name, state):
            return state

        async def strategy_returns_none(state):
            state["ran"] = True

        result = await benchmark(
            nodes={
                "a": step_a, "b": step_b
            },
            edges=[("a", "b")],
            execute_node=execute_node,
            strategies={"returns_none": strategy_returns_none},
            n_runs=1,
        )
        assert "returns_none" in result["outputs"]
        assert result["outputs"]["returns_none"]["ran"] is True

    async def test_custom_strategy_returns_non_dict(self):

        async def execute_node(name, state):
            return state

        async def strategy_returns_string(state):
            return "custom_result"

        result = await benchmark(
            nodes={
                "a": step_a, "b": step_b
            },
            edges=[("a", "b")],
            execute_node=execute_node,
            strategies={"returns_string": strategy_returns_string},
            n_runs=1,
        )
        assert result["outputs"]["returns_string"] == "custom_result"

    async def test_execute_node_non_dict_sequential_raises(self):

        async def execute_node(name, state):
            return None  # invalid

        with pytest.raises(TypeError, match="execute_node must return a dict.*got NoneType.*node"):
            await benchmark(
                nodes={
                    "a": step_a, "b": step_b
                },
                edges=[("a", "b")],
                execute_node=execute_node,
                n_runs=1,
            )

    async def test_execute_node_non_dict_parallel_raises(self):
        call_count = [0]

        async def execute_node(name, state):
            call_count[0] += 1
            # Sequential runs first (a, b); parallel then runs (a, b, c). Fail on 4th call (b in parallel).
            if call_count[0] >= 4:
                return "error"
            return {f"{name}_done": True, **state}

        with pytest.raises(TypeError, match="execute_node must return a dict.*got str.*node"):
            await benchmark(
                nodes={
                    "a": step_a, "b": step_b, "c": step_c
                },
                edges=[("a", "b"), ("a", "c")],
                execute_node=execute_node,
                n_runs=1,
            )


# -- speculative_opportunities tests -----------------------------------------


class TestSpeculativeOpportunities:

    def test_no_routers_returns_empty(self):
        result = speculative_opportunities(
            nodes={
                "a": step_a, "b": step_b
            },
            edges=[("a", "b")],
        )
        assert result == []

    def test_single_router_returns_opportunity(self):
        result = speculative_opportunities(
            nodes={
                "router": route_fn, "a": step_a, "b": step_b
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
        )
        assert len(result) >= 1
        opp = result[0]
        assert opp["decision_node"] == "router"
        assert "branches" in opp
        assert "merge_nodes" in opp
        assert "speculatable_nodes" in opp
        assert "max_branch_depth" in opp
        assert "is_cycle_exit" in opp

    def test_merge_node_identified(self):
        result = speculative_opportunities(
            nodes={
                "router": route_fn, "a": step_a, "b": step_b, "merge": fn_merge
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
        assert len(result) >= 1
        opp = result[0]
        assert "merge" in opp["merge_nodes"]

    def test_speculatable_count(self):
        result = speculative_opportunities(
            nodes={
                "router": route_fn, "a": step_a, "b": step_b
            },
            edges=[("router", "a"), ("router", "b")],
            conditional_edges={"router": {
                "left": "a", "right": "b"
            }},
        )
        opp = result[0]
        assert opp["speculatable_nodes"] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
