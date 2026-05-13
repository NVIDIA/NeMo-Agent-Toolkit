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
"""
Standardized execution metrics.

Provides a shared ``ExecutionMetrics`` dataclass that all framework
executors use to report execution results.  Works for any execution strategy
(speculative, parallel, or sequential).  Absorbs the sequential-time
estimation logic previously duplicated across framework packages.

No framework imports -- uses only Python stdlib + ``ExecutionState``.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from nat_app.executors.execution_state import ExecutionState


def _estimate_sequential_time_ms(execution_state: ExecutionState) -> float:
    """Estimate what sequential execution would have taken.

    Walks the timeline events in order, stacking durations as if each
    waited for the previous to finish.  Falls back to summing raw node
    durations when no timeline events are recorded.

    Args:
        execution_state: The completed execution state with timing data.

    Returns:
        Estimated sequential execution time in milliseconds.
    """
    timeline = execution_state.execution_timeline
    if timeline:
        sorted_events = sorted(timeline, key=lambda x: x["start"])
        last_end = 0.0
        for event in sorted_events:
            event_start = max(event["start"], last_end)
            last_end = event_start + event["duration"]
        return last_end * 1000
    return sum(sum(d) for d in execution_state.node_durations.values()) * 1000


@dataclass
class ExecutionMetrics:
    """Standardized metrics from an execution run.

    Strategy-agnostic: works for speculative, parallel, or any future
    execution strategy.  Fields like ``tools_cancelled`` and
    ``speculation_decisions`` default to safe empty values when unused.

    Framework executors build this via ``from_execution_state`` rather
    than hand-assembling a metrics dict.
    """

    total_time_ms: float
    sequential_time_ms: float
    tools_launched: int
    tools_completed: int
    tools_cancelled: int
    iterations: int = 0
    speculation_decisions: dict[str, str] = field(default_factory=dict)
    execution_timeline: list[dict[str, Any]] = field(default_factory=list)
    profiling: dict[str, Any] = field(default_factory=dict)

    @property
    def speedup_ratio(self) -> float:
        """Estimated speedup vs sequential execution."""
        return self.sequential_time_ms / self.total_time_ms if self.total_time_ms > 0 else 1.0

    @property
    def speedup_pct(self) -> float:
        """Speedup as a percentage improvement (0 = no change)."""
        return (self.speedup_ratio - 1.0) * 100

    @classmethod
    def from_execution_state(
        cls,
        execution_state: ExecutionState,
        elapsed_s: float,
        iterations: int = 0,
    ) -> ExecutionMetrics:
        """Build metrics from an ``ExecutionState`` after execution completes.

        Args:
            execution_state: The execution state used during the run.
            elapsed_s: Wall-clock elapsed time in **seconds**.
            iterations: Number of execution loop iterations (0 for stage-based).

        Returns:
            ExecutionMetrics populated from the execution state.
        """
        return cls(
            total_time_ms=elapsed_s * 1000,
            sequential_time_ms=_estimate_sequential_time_ms(execution_state),
            tools_launched=execution_state.tools_launched,
            tools_completed=execution_state.tools_completed,
            tools_cancelled=execution_state.tools_cancelled,
            iterations=iterations,
            speculation_decisions=dict(execution_state.speculation_decisions),
            execution_timeline=list(execution_state.execution_timeline),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to the dict format expected by callers.

        Always includes ``speedup_ratio`` when sequential time is positive.

        Returns:
            Metrics as a plain dict for serialization and logging.
        """
        d: dict[str, Any] = {
            "total_time_ms": self.total_time_ms,
            "sequential_time_ms": self.sequential_time_ms,
            "tools_launched": self.tools_launched,
            "tools_completed": self.tools_completed,
            "tools_cancelled": self.tools_cancelled,
            "speculation_decisions": self.speculation_decisions,
            "execution_timeline": self.execution_timeline,
        }
        if self.iterations:
            d["iterations"] = self.iterations
        if self.profiling:
            d["profiling"] = self.profiling
        if self.sequential_time_ms > 0:
            d["speedup_ratio"] = self.speedup_ratio
            d["speedup_pct"] = self.speedup_pct
        return d
