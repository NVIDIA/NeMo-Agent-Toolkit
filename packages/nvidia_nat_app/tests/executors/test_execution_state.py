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
"""Tests for ExecutionState mutation methods and defaults."""

import pytest

from nat_app.executors.execution_state import ExecutionState


class TestExecutionStateDefaults:

    def test_fresh_state_has_zeroed_counters(self):
        state = ExecutionState()
        assert state.tools_launched == 0
        assert state.tools_completed == 0
        assert state.tools_cancelled == 0

    def test_fresh_state_has_empty_collections(self):
        state = ExecutionState()
        assert state.ready_nodes == set()
        assert state.running_tasks == {}
        assert state.completed_nodes == {}
        assert state.speculation_decisions == {}
        assert state.cancelled_nodes == set()
        assert state.execution_timeline == []
        assert state.node_start_times == {}

    def test_execution_start_time_defaults_to_zero(self):
        state = ExecutionState()
        assert state.execution_start_time == 0.0


class TestMarkNodeReady:

    def test_adds_to_ready_nodes(self):
        state = ExecutionState()
        state.mark_node_ready("node_a")
        assert "node_a" in state.ready_nodes

    def test_multiple_nodes(self):
        state = ExecutionState()
        state.mark_node_ready("a")
        state.mark_node_ready("b")
        assert state.ready_nodes == {"a", "b"}

    def test_idempotent(self):
        state = ExecutionState()
        state.mark_node_ready("a")
        state.mark_node_ready("a")
        assert state.ready_nodes == {"a"}


class TestMarkNodeCompleted:

    def test_stores_result(self):
        state = ExecutionState()
        state.mark_node_completed("a", {"key": "value"})
        assert state.completed_nodes["a"] == {"key": "value"}

    def test_increments_tools_completed(self):
        state = ExecutionState()
        state.mark_node_completed("a")
        assert state.tools_completed == 1

    def test_increments_execution_count(self):
        state = ExecutionState()
        state.mark_node_completed("a")
        state.mark_node_completed("a")
        assert state.node_execution_count["a"] == 2

    def test_none_result_stores_empty_dict(self):
        state = ExecutionState()
        state.mark_node_completed("a", None)
        assert state.completed_nodes["a"] == {}


class TestMarkNodeCancelled:

    def test_adds_to_cancelled_nodes(self):
        state = ExecutionState()
        state.mark_node_cancelled("a")
        assert "a" in state.cancelled_nodes

    def test_increments_tools_cancelled(self):
        state = ExecutionState()
        state.mark_node_cancelled("a")
        assert state.tools_cancelled == 1


class TestRecordDecision:

    def test_stores_decision(self):
        state = ExecutionState()
        state.record_decision("router", "branch_a", 1)
        assert state.speculation_decisions["router"] == "branch_a"

    def test_stores_iteration(self):
        state = ExecutionState()
        state.record_decision("router", "branch_a", 3)
        assert state.last_decision_iteration["router"] == 3

    def test_stores_in_completed_nodes(self):
        state = ExecutionState()
        state.record_decision("router", "branch_a", 1)
        assert state.completed_nodes["router"] == {"chosen": "branch_a"}


class TestClearForReexecution:

    def test_removes_from_completed(self):
        state = ExecutionState()
        state.mark_node_completed("a", {"result": True})
        state.clear_for_reexecution("a")
        assert "a" not in state.completed_nodes

    def test_removes_speculation_decision(self):
        state = ExecutionState()
        state.record_decision("router", "left", 1)
        state.clear_for_reexecution("router")
        assert "router" not in state.speculation_decisions

    def test_no_error_if_not_present(self):
        state = ExecutionState()
        state.clear_for_reexecution("nonexistent")


class TestRecordTimelineEvent:

    def test_appends_event(self):
        state = ExecutionState()
        state.execution_start_time = 100.0
        state.record_timeline_event("a", 100.5, 101.0)
        assert len(state.execution_timeline) == 1

    def test_event_shape(self):
        state = ExecutionState()
        state.execution_start_time = 100.0
        state.record_timeline_event("a", 100.5, 101.0, status="completed")

        event = state.execution_timeline[0]
        assert event["node"] == "a"
        assert event["start"] == pytest.approx(0.5)
        assert event["end"] == pytest.approx(1.0)
        assert event["duration"] == pytest.approx(0.5)
        assert event["status"] == "completed"

    def test_cancelled_status(self):
        state = ExecutionState()
        state.execution_start_time = 0.0
        state.record_timeline_event("b", 0.1, 0.2, status="cancelled")
        assert state.execution_timeline[0]["status"] == "cancelled"


class TestRecordNodeDuration:

    def test_appends_duration(self):
        state = ExecutionState()
        state.record_node_duration("a", 0.5)
        assert state.node_durations["a"] == [0.5]

    def test_multiple_durations(self):
        state = ExecutionState()
        state.record_node_duration("a", 0.5)
        state.record_node_duration("a", 0.3)
        assert state.node_durations["a"] == [0.5, 0.3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
