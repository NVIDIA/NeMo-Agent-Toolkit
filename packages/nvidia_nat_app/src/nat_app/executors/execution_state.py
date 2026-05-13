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
Execution state tracking.

Framework-agnostic state machine for graph execution. Tracks node
lifecycle (ready -> running -> completed/cancelled), speculation decisions,
cycle re-execution, and execution timeline for profiling.

No framework imports -- uses only Python stdlib (asyncio, collections, dataclasses).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass
class ExecutionState:
    """
    Mutable state tracking for graph execution.

    Centralizes all mutable state during graph execution. Framework-specific
    executors use this to track node progress, speculation decisions, and
    profiling metrics.
    """

    # Node tracking
    ready_nodes: set[str] = field(default_factory=set)
    """Nodes ready to launch (all deps satisfied)."""

    running_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    """Currently executing tasks (node_name -> task)."""

    completed_nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Completed nodes with their results (node_name -> result)."""

    speculation_decisions: dict[str, str] = field(default_factory=dict)
    """Speculation decisions made (decision_node -> chosen_target)."""

    cancelled_nodes: set[str] = field(default_factory=set)
    """Nodes that were cancelled (unchosen paths)."""

    node_execution_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """How many times each node has executed (for cycle detection)."""

    last_decision_iteration: dict[str, int] = field(default_factory=dict)
    """Which iteration each decision node last decided."""

    channels: dict[str, Any] = field(default_factory=dict)
    """Main execution channels for state management (framework-specific)."""

    # Counters
    tools_launched: int = 0
    tools_cancelled: int = 0
    tools_completed: int = 0

    # Timing
    node_start_times: dict[str, float] = field(default_factory=dict)
    """When each node started (node_name -> start_time)."""

    node_durations: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    """Duration of each node execution (node_name -> [durations])."""

    # Profiling
    deepcopy_times: list[float] = field(default_factory=list)
    task_creation_times: list[float] = field(default_factory=list)
    state_merge_times: list[float] = field(default_factory=list)

    # Execution timeline for visualization
    execution_timeline: list[dict[str, Any]] = field(default_factory=list)
    execution_start_time: float = 0.0

    prerecorded_end_times: dict[str, float] = field(default_factory=dict)
    """Pre-recorded end times for speculative tools (accurate timing measurement)."""

    # -- State mutation helpers ---------------------------------------------

    def mark_node_ready(self, node_name: str) -> None:
        """Add a node to the ready set.

        Args:
            node_name: Name of the node that is ready to execute.
        """
        self.ready_nodes.add(node_name)

    def mark_node_completed(self, node_name: str, result: dict[str, Any] | None = None) -> None:
        """Record a node as completed with its result.

        Args:
            node_name: Name of the completed node.
            result: The node's output dict, or None.
        """
        self.completed_nodes[node_name] = result or {}
        self.tools_completed += 1
        self.node_execution_count[node_name] += 1

    def mark_node_cancelled(self, node_name: str) -> None:
        """Record a node as cancelled.

        Args:
            node_name: Name of the cancelled node.
        """
        self.cancelled_nodes.add(node_name)
        self.tools_cancelled += 1

    def record_decision(self, decision_node: str, chosen_target: str, iteration: int) -> None:
        """Record a decision node's choice for a given iteration.

        Args:
            decision_node: Name of the decision node.
            chosen_target: The target node chosen.
            iteration: The execution loop iteration number.
        """
        self.speculation_decisions[decision_node] = chosen_target
        self.completed_nodes[decision_node] = {"chosen": chosen_target}
        self.last_decision_iteration[decision_node] = iteration

    def clear_for_reexecution(self, node_name: str) -> None:
        """Clear a node's completion status to allow re-execution in cycles.

        Args:
            node_name: Name of the node to reset.
        """
        if node_name in self.completed_nodes:
            del self.completed_nodes[node_name]
        if node_name in self.speculation_decisions:
            del self.speculation_decisions[node_name]

    def record_timeline_event(
        self,
        node_name: str,
        start_time: float,
        end_time: float,
        status: str = "completed",
    ) -> None:
        """Record a node execution in the timeline for visualization.

        Args:
            node_name: Name of the node.
            start_time: Absolute wall-clock start time.
            end_time: Absolute wall-clock end time.
            status: Outcome label (e.g. ``"completed"``, ``"cancelled"``).
        """
        duration = end_time - start_time
        self.execution_timeline.append({
            "node": node_name,
            "start": start_time - self.execution_start_time,
            "end": end_time - self.execution_start_time,
            "duration": duration,
            "iteration": self.node_execution_count[node_name],
            "status": status,
        })

    def record_node_duration(self, node_name: str, duration: float) -> None:
        """Append a duration measurement for a node execution.

        Args:
            node_name: Name of the node.
            duration: Elapsed time in seconds for this execution.
        """
        self.node_durations[node_name].append(duration)
