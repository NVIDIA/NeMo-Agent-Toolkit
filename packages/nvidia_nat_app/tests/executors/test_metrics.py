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
"""Tests for ExecutionMetrics: properties, to_dict, from_execution_state."""

import pytest

from nat_app.executors.execution_state import ExecutionState
from nat_app.executors.metrics import ExecutionMetrics


class TestSpeedupProperties:

    def test_speedup_ratio(self):
        m = ExecutionMetrics(
            total_time_ms=500.0,
            sequential_time_ms=1000.0,
            tools_launched=3,
            tools_completed=2,
            tools_cancelled=1,
        )
        assert m.speedup_ratio == pytest.approx(2.0)

    def test_speedup_pct(self):
        m = ExecutionMetrics(
            total_time_ms=500.0,
            sequential_time_ms=1000.0,
            tools_launched=3,
            tools_completed=2,
            tools_cancelled=1,
        )
        assert m.speedup_pct == pytest.approx(100.0)

    def test_speedup_ratio_when_total_is_zero(self):
        m = ExecutionMetrics(
            total_time_ms=0.0,
            sequential_time_ms=1000.0,
            tools_launched=0,
            tools_completed=0,
            tools_cancelled=0,
        )
        assert m.speedup_ratio == 1.0

    def test_no_speedup(self):
        m = ExecutionMetrics(
            total_time_ms=1000.0,
            sequential_time_ms=1000.0,
            tools_launched=1,
            tools_completed=1,
            tools_cancelled=0,
        )
        assert m.speedup_ratio == pytest.approx(1.0)
        assert m.speedup_pct == pytest.approx(0.0)


class TestToDict:

    def test_includes_required_keys(self):
        m = ExecutionMetrics(
            total_time_ms=500.0,
            sequential_time_ms=1000.0,
            tools_launched=3,
            tools_completed=2,
            tools_cancelled=1,
        )
        d = m.to_dict()
        assert d["total_time_ms"] == 500.0
        assert d["sequential_time_ms"] == 1000.0
        assert d["tools_launched"] == 3
        assert d["tools_completed"] == 2
        assert d["tools_cancelled"] == 1

    def test_includes_speedup_when_sequential_positive(self):
        m = ExecutionMetrics(
            total_time_ms=500.0,
            sequential_time_ms=1000.0,
            tools_launched=3,
            tools_completed=2,
            tools_cancelled=1,
        )
        d = m.to_dict()
        assert "speedup_ratio" in d
        assert "speedup_pct" in d

    def test_omits_speedup_when_sequential_zero(self):
        m = ExecutionMetrics(
            total_time_ms=500.0,
            sequential_time_ms=0.0,
            tools_launched=1,
            tools_completed=1,
            tools_cancelled=0,
        )
        d = m.to_dict()
        assert "speedup_ratio" not in d

    def test_omits_iterations_when_zero(self):
        m = ExecutionMetrics(
            total_time_ms=100.0,
            sequential_time_ms=200.0,
            tools_launched=1,
            tools_completed=1,
            tools_cancelled=0,
            iterations=0,
        )
        d = m.to_dict()
        assert "iterations" not in d

    def test_includes_iterations_when_nonzero(self):
        m = ExecutionMetrics(
            total_time_ms=100.0,
            sequential_time_ms=200.0,
            tools_launched=1,
            tools_completed=1,
            tools_cancelled=0,
            iterations=5,
        )
        d = m.to_dict()
        assert d["iterations"] == 5

    def test_omits_profiling_when_empty(self):
        m = ExecutionMetrics(
            total_time_ms=100.0,
            sequential_time_ms=200.0,
            tools_launched=1,
            tools_completed=1,
            tools_cancelled=0,
        )
        d = m.to_dict()
        assert "profiling" not in d

    def test_includes_profiling_when_present(self):
        m = ExecutionMetrics(
            total_time_ms=100.0,
            sequential_time_ms=200.0,
            tools_launched=1,
            tools_completed=1,
            tools_cancelled=0,
            profiling={"deepcopy_ms": 5.0},
        )
        d = m.to_dict()
        assert d["profiling"] == {"deepcopy_ms": 5.0}


class TestFromExecutionState:

    @pytest.fixture(name="populated_state")
    def fixture_populated_state(self):
        state = ExecutionState()
        state.execution_start_time = 0.0
        state.tools_launched = 4
        state.tools_completed = 3
        state.tools_cancelled = 1
        state.speculation_decisions["router_1"] = "left"
        state.record_timeline_event("a", 0.0, 0.1)
        state.record_timeline_event("b", 0.0, 0.15)
        state.record_timeline_event("c", 0.15, 0.25)
        return state

    def test_basic_fields(self, populated_state):
        m = ExecutionMetrics.from_execution_state(populated_state, elapsed_s=0.25)
        assert m.total_time_ms == pytest.approx(250.0)
        assert m.tools_launched == 4
        assert m.tools_completed == 3
        assert m.tools_cancelled == 1

    def test_speculation_decisions_copied(self, populated_state):
        m = ExecutionMetrics.from_execution_state(populated_state, elapsed_s=0.25)
        assert m.speculation_decisions == {"router_1": "left"}

    def test_timeline_copied(self, populated_state):
        m = ExecutionMetrics.from_execution_state(populated_state, elapsed_s=0.25)
        assert len(m.execution_timeline) == 3

    def test_sequential_time_estimated(self, populated_state):
        m = ExecutionMetrics.from_execution_state(populated_state, elapsed_s=0.25)
        assert m.sequential_time_ms > 0

    def test_iterations_stored(self, populated_state):
        m = ExecutionMetrics.from_execution_state(populated_state, elapsed_s=0.25, iterations=7)
        assert m.iterations == 7

    def test_from_empty_state(self):
        state = ExecutionState()
        m = ExecutionMetrics.from_execution_state(state, elapsed_s=0.0)
        assert m.total_time_ms == 0.0
        assert m.tools_launched == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
