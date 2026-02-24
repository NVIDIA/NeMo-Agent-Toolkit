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
"""Tests for run_speculation() and SpeculativeResult."""

import asyncio

import pytest

from nat_app.executors.execution_state import ExecutionState
from nat_app.executors.runner import SpeculativeResult
from nat_app.executors.runner import run_speculation
from nat_app.speculation.plan import SpeculationPlan
from nat_app.speculation.strategies.router_branch import RouterBranchResolution

# -- Helpers -----------------------------------------------------------------


def _make_plan(
    decision_node: str = "router",
    targets: frozenset[str] = frozenset({"a", "b"}),
    cancel_map: dict[str, frozenset[str]] | None = None,
) -> SpeculationPlan:
    if cancel_map is None:
        cancel_map = {"left": frozenset({"b"}), "right": frozenset({"a"})}
    resolution = RouterBranchResolution(
        cancel_map=cancel_map,
        label_map=None,
        all_targets=targets,
    )
    return SpeculationPlan(
        strategy="router_branch",
        decision_node=decision_node,
        targets_to_launch=targets,
        excluded_nodes=frozenset(),
        resolution=resolution,
        merge_nodes=frozenset(),
        max_branch_depth=1,
        is_cycle_exit=False,
    )


async def _slow_node(name: str, delay: float = 0.05, result: str | None = None) -> dict:
    await asyncio.sleep(delay)
    return {f"{name}_out": result or f"{name}_done"}


class _RaisingResolution:

    def resolve(self, _label: str):
        raise RuntimeError("resolve failed")


# -- Tests -------------------------------------------------------------------


class TestRunSpeculativeRouter:

    async def test_basic_left_chosen(self):
        plan = _make_plan()
        state = ExecutionState()
        state.execution_start_time = 0.0

        async def run_node(name: str):
            if name == "router":
                await asyncio.sleep(0.02)
                return {"decision": "left"}
            return await _slow_node(name)

        result = await run_speculation(
            plan,
            state,
            run_node=run_node,
            get_decision=lambda _: "left",
        )

        assert isinstance(result, SpeculativeResult)
        assert result.chosen_label == "left"
        assert result.decision_result == {"decision": "left"}
        assert "a" in result.chosen_results
        assert "b" not in result.chosen_results
        assert "b" in result.cancelled_nodes

    async def test_basic_right_chosen(self):
        plan = _make_plan()
        state = ExecutionState()
        state.execution_start_time = 0.0

        result = await run_speculation(
            plan,
            state,
            run_node=lambda name: _slow_node(name),
            get_decision=lambda _: "right",
        )

        assert result.chosen_label == "right"
        assert "b" in result.chosen_results
        assert "a" not in result.chosen_results
        assert "a" in result.cancelled_nodes

    async def test_metrics_tracked(self):
        plan = _make_plan()
        state = ExecutionState()
        state.execution_start_time = 0.0

        await run_speculation(
            plan,
            state,
            run_node=lambda name: _slow_node(name),
            get_decision=lambda _: "left",
        )

        # router + 2 targets launched
        assert state.tools_launched == 3
        # router + chosen target completed
        assert state.tools_completed == 2
        # unchosen target cancelled
        assert state.tools_cancelled == 1

    async def test_router_decision_recorded(self):
        plan = _make_plan()
        state = ExecutionState()
        state.execution_start_time = 0.0

        await run_speculation(
            plan,
            state,
            run_node=lambda name: _slow_node(name),
            get_decision=lambda _: "left",
        )

        assert "router" in state.speculation_decisions
        assert state.speculation_decisions["router"] == "left"

    async def test_timeline_events_recorded(self):
        plan = _make_plan()
        state = ExecutionState()
        state.execution_start_time = 0.0

        await run_speculation(
            plan,
            state,
            run_node=lambda name: _slow_node(name),
            get_decision=lambda _: "left",
        )

        nodes_in_timeline = {e["node"] for e in state.execution_timeline}
        assert "router" in nodes_in_timeline
        assert "a" in nodes_in_timeline
        assert "b" in nodes_in_timeline

        b_events = [e for e in state.execution_timeline if e["node"] == "b"]
        assert b_events[0]["status"] == "cancelled"

    async def test_no_cancel_map_entry(self):
        """When chosen label has no cancel_map entry, nothing is cancelled."""
        plan = _make_plan(cancel_map={"left": frozenset({"b"})}, )
        state = ExecutionState()
        state.execution_start_time = 0.0

        result = await run_speculation(
            plan,
            state,
            run_node=lambda name: _slow_node(name),
            get_decision=lambda _: "right",
        )

        assert result.cancelled_nodes == frozenset()
        assert "a" in result.chosen_results
        assert "b" in result.chosen_results

    async def test_single_target(self):
        plan = _make_plan(
            targets=frozenset({"only_target"}),
            cancel_map={},
        )
        state = ExecutionState()
        state.execution_start_time = 0.0

        result = await run_speculation(
            plan,
            state,
            run_node=lambda name: _slow_node(name),
            get_decision=lambda _: "only_target",
        )

        assert result.cancelled_nodes == frozenset()
        assert "only_target" in result.chosen_results

    async def test_fast_target_already_done_before_router(self):
        """Target completes before router -- should still be collected."""
        plan = _make_plan()
        state = ExecutionState()
        state.execution_start_time = 0.0

        async def run_node(name: str):
            if name == "router":
                await asyncio.sleep(0.1)
                return "left"
            # Targets finish instantly
            return {f"{name}_out": "fast"}

        result = await run_speculation(
            plan,
            state,
            run_node=run_node,
            get_decision=lambda _: "left",
        )

        assert "a" in result.chosen_results
        assert result.chosen_results["a"] == {"a_out": "fast"}

    async def test_non_dict_results(self):
        """Non-dict results are still collected correctly."""
        plan = _make_plan()
        state = ExecutionState()
        state.execution_start_time = 0.0

        async def run_node(name: str):
            await asyncio.sleep(0.01)
            return f"{name}_string_result"

        result = await run_speculation(
            plan,
            state,
            run_node=run_node,
            get_decision=lambda _: "left",
        )

        assert result.decision_result == "router_string_result"
        assert result.chosen_results["a"] == "a_string_result"

    async def test_cancel_map_references_node_not_in_targets(self):
        """When cancel_map references a node not in targets_to_launch, task is None."""
        plan = _make_plan(
            targets=frozenset({"a"}),
            cancel_map={"left": frozenset({"b"})},
        )
        state = ExecutionState()
        state.execution_start_time = 0.0

        result = await run_speculation(
            plan,
            state,
            run_node=lambda name: _slow_node(name),
            get_decision=lambda _: "left",
        )

        assert "a" in result.chosen_results
        assert result.cancelled_nodes == frozenset()

    async def test_cancelled_task_already_done_before_cancel(self):
        """Cancelled target completes before cancel request -- task.result() path."""
        plan = _make_plan()
        state = ExecutionState()
        state.execution_start_time = 0.0

        async def run_node(name: str):
            if name == "router":
                await asyncio.sleep(0.1)
                return "left"
            await asyncio.sleep(0.01)
            return {f"{name}_out": "done"}

        result = await run_speculation(
            plan,
            state,
            run_node=run_node,
            get_decision=lambda _: "left",
        )

        assert "a" in result.chosen_results
        assert "b" in result.cancelled_nodes

    async def test_chosen_target_raises_continues_without_crash(self):
        """Chosen target raises during await -- exception caught, not in chosen_results."""
        plan = _make_plan()
        state = ExecutionState()
        state.execution_start_time = 0.0

        async def run_node(name: str):
            if name == "router":
                await asyncio.sleep(0.02)
                return "left"
            if name == "a":
                raise ValueError("simulated failure")
            return await _slow_node(name)

        result = await run_speculation(
            plan,
            state,
            run_node=run_node,
            get_decision=lambda _: "left",
        )

        assert result.chosen_label == "left"
        assert "a" not in result.chosen_results
        assert "b" in result.cancelled_nodes

        # Metrics reconcile: launched == completed + cancelled
        assert state.tools_launched == 3
        assert state.tools_completed == 2  # router + failed node
        assert state.tools_cancelled == 1
        a_events = [e for e in state.execution_timeline if e["node"] == "a"]
        assert len(a_events) == 1
        assert a_events[0]["status"] == "failed"

    async def test_decision_task_raises_cancels_targets(self):
        """When decision task raises, target tasks are cancelled before re-raising."""
        plan = _make_plan()
        state = ExecutionState()
        state.execution_start_time = 0.0

        async def run_node(name: str):
            if name == "router":
                raise ValueError("decision failed")
            await asyncio.sleep(10)  # Long sleep - would hang if not cancelled
            return {f"{name}_out": "done"}

        with pytest.raises(ValueError, match="decision failed"):
            await asyncio.wait_for(
                run_speculation(plan, state, run_node=run_node, get_decision=lambda _: "left"),
                timeout=2.0,
            )

    async def test_get_decision_raises_cancels_targets(self):
        """When get_decision raises, target tasks are cancelled before re-raising."""
        plan = _make_plan()
        state = ExecutionState()
        state.execution_start_time = 0.0

        async def run_node(name: str):
            if name == "router":
                await asyncio.sleep(0.02)
                return {"x": 1}
            await asyncio.sleep(10)  # Long sleep - would hang if not cancelled
            return {f"{name}_out": "done"}

        def get_decision(_result):
            raise RuntimeError("bad decision")

        with pytest.raises(RuntimeError, match="bad decision"):
            await asyncio.wait_for(
                run_speculation(plan, state, run_node=run_node, get_decision=get_decision),
                timeout=2.0,
            )

    async def test_resolution_resolve_raises_cancels_targets(self):
        """When plan.resolution.resolve raises, target tasks are cancelled before re-raising."""
        plan = SpeculationPlan(
            strategy="router_branch",
            decision_node="router",
            targets_to_launch=frozenset({"a", "b"}),
            excluded_nodes=frozenset(),
            resolution=_RaisingResolution(),
            merge_nodes=frozenset(),
            max_branch_depth=1,
            is_cycle_exit=False,
        )
        state = ExecutionState()
        state.execution_start_time = 0.0

        async def run_node(name: str):
            if name == "router":
                await asyncio.sleep(0.02)
                return "left"
            await asyncio.sleep(10)  # Long sleep - would hang if not cancelled
            return {f"{name}_out": "done"}

        with pytest.raises(RuntimeError, match="resolve failed"):
            await asyncio.wait_for(
                run_speculation(plan, state, run_node=run_node, get_decision=lambda _: "left"),
                timeout=2.0,
            )


class TestSpeculativeResultDataclass:

    def test_frozen(self):
        result = SpeculativeResult(
            chosen_label="left",
            decision_result={"x": 1},
            chosen_results={"a": {
                "a_out": "done"
            }},
            cancelled_nodes=frozenset({"b"}),
            rerun_nodes=frozenset(),
        )
        with pytest.raises(AttributeError):
            result.chosen_label = "right"

    def test_fields(self):
        result = SpeculativeResult(
            chosen_label="left",
            decision_result=None,
            chosen_results={},
            cancelled_nodes=frozenset(),
            rerun_nodes=frozenset(),
        )
        assert result.chosen_label == "left"
        assert result.decision_result is None
        assert result.chosen_results == {}
        assert result.cancelled_nodes == frozenset()
        assert result.rerun_nodes == frozenset()


class TestPublicRunnerImports:

    def test_importable_from_nat_app(self):
        from nat_app import SpeculativeResult as SR
        from nat_app import run_speculation as rs
        assert SR is SpeculativeResult
        assert rs is run_speculation

    def test_importable_from_executors(self):
        from nat_app.executors import SpeculativeResult as SR
        from nat_app.executors import run_speculation as rs
        assert SR is SpeculativeResult
        assert rs is run_speculation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
