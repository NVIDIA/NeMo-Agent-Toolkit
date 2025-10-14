# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import statistics
from collections import defaultdict
from typing import Any

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType

logger = logging.getLogger(__name__)


def compute_node_metrics(all_steps: list[list[IntermediateStep]]) -> dict[str, Any]:
    """
    Compute node-level execution metrics from profiler traces. This function analyzes NODE_START and
    NODE_END events to calculate:
    1. Total execution time per node
    2. Average execution time per node
    3. Min/max execution time per node
    4. Percentile statistics (P50, P90, P95, P99)
    5. Call count per node

    Args:
        all_steps: List of intermediate steps for each example/request

    Returns:
        Dictionary with node metrics structured as:
        {
            "node_metrics": {
                "agent": {
                    "call_count": 10,
                    "total_time_seconds": 45.2,
                    "avg_time_seconds": 4.52,
                    "min_time_seconds": 2.1,
                    "max_time_seconds": 8.3,
                    "p50_time_seconds": 4.1,
                    "p90_time_seconds": 7.2,
                    "p95_time_seconds": 7.8,
                    "p99_time_seconds": 8.2
                },
                ...
            },
            "total_nodes_tracked": 3,
            "total_node_executions": 30
        }
    """
    # Collect all node execution durations by node name
    node_durations = defaultdict(list)
    node_call_counts = defaultdict(int)
    total_node_executions = 0

    # Process all examples
    for steps in all_steps:
        for step in steps:
            # Only process NODE_END events (which contain duration info)
            if step.event_type == IntermediateStepType.NODE_END:
                node_name = step.name

                if node_name:
                    # Extract duration from metadata
                    duration = None
                    if step.metadata and isinstance(step.metadata, dict):
                        node_info = step.metadata.get("node_info", {})
                        if isinstance(node_info, dict):
                            duration = node_info.get("duration_seconds")

                    # Fallback: calculate from timestamps if metadata missing
                    if duration is None and step.span_event_timestamp:
                        duration = step.event_timestamp - step.span_event_timestamp

                    if duration is not None and duration >= 0:
                        node_durations[node_name].append(duration)
                        node_call_counts[node_name] += 1
                        total_node_executions += 1

    # Calculate metrics for each node
    node_metrics = {}

    for node_name, durations in node_durations.items():
        if not durations:
            continue

        # Sort durations for percentile calculation
        sorted_durations = sorted(durations)

        # Calculate percentiles
        def percentile(data, pct):
            if len(data) == 0:
                return 0.0
            if len(data) == 1:
                return data[0]
            k = (len(data) - 1) * pct / 100.0
            f = int(k)
            c = f + 1
            if c >= len(data):
                return data[-1]
            d0 = data[f]
            d1 = data[c]
            return d0 + (d1 - d0) * (k - f)

        node_metrics[node_name] = {
            "call_count": node_call_counts[node_name],
            "total_time_seconds": round(sum(durations), 3),
            "avg_time_seconds": round(statistics.mean(durations), 3),
            "min_time_seconds": round(min(durations), 3),
            "max_time_seconds": round(max(durations), 3),
            "p50_time_seconds": round(percentile(sorted_durations, 50), 3),
            "p90_time_seconds": round(percentile(sorted_durations, 90), 3),
            "p95_time_seconds": round(percentile(sorted_durations, 95), 3),
            "p99_time_seconds": round(percentile(sorted_durations, 99), 3),
        }

        # Add median and standard deviation for additional insight
        if len(durations) > 1:
            node_metrics[node_name]["median_time_seconds"] = round(statistics.median(durations), 3)
            node_metrics[node_name]["std_dev_seconds"] = round(statistics.stdev(durations), 3)
        else:
            node_metrics[node_name]["median_time_seconds"] = node_metrics[node_name]["avg_time_seconds"]
            node_metrics[node_name]["std_dev_seconds"] = 0.0

    # Sort nodes by total time (descending) for easier identification of bottlenecks
    sorted_node_metrics = dict(sorted(node_metrics.items(), key=lambda x: x[1]["total_time_seconds"], reverse=True))

    result = {
        "node_metrics": sorted_node_metrics,
        "total_nodes_tracked": len(node_metrics),
        "total_node_executions": total_node_executions
    }

    logger.info("Computed node metrics: %d unique nodes, %d total executions", len(node_metrics), total_node_executions)

    return result


def generate_node_metrics_summary(node_metrics: dict[str, Any]) -> str:
    """
    Generate a human-readable text summary of node metrics.

    Args:
        node_metrics: Output from compute_node_metrics()

    Returns:
        Formatted string summary of node performance
    """
    if not node_metrics or "node_metrics" not in node_metrics:
        return "No node metrics available."

    metrics = node_metrics["node_metrics"]
    total_nodes = node_metrics.get("total_nodes_tracked", 0)
    total_executions = node_metrics.get("total_node_executions", 0)

    if not metrics:
        return "No node metrics available."

    summary = []
    summary.append("=" * 80)
    summary.append("NODE EXECUTION METRICS SUMMARY")
    summary.append("=" * 80)
    summary.append(f"\nTotal Unique Nodes: {total_nodes}")
    summary.append(f"Total Node Executions: {total_executions}\n")

    summary.append("-" * 80)
    summary.append("Node Performance Breakdown (sorted by total time)")
    summary.append("-" * 80)

    for node_name, stats in metrics.items():
        summary.append(f"\nNode: {node_name}")
        summary.append(f"  Call Count:        {stats['call_count']}")
        summary.append(f"  Total Time:        {stats['total_time_seconds']:.3f}s")
        summary.append(f"  Average Time:      {stats['avg_time_seconds']:.3f}s")
        summary.append(f"  Min Time:          {stats['min_time_seconds']:.3f}s")
        summary.append(f"  Max Time:          {stats['max_time_seconds']:.3f}s")
        summary.append(f"  Median Time:       {stats.get('median_time_seconds', 0):.3f}s")
        summary.append(f"  P50 (Median):      {stats['p50_time_seconds']:.3f}s")
        summary.append(f"  P90:               {stats['p90_time_seconds']:.3f}s")
        summary.append(f"  P95:               {stats['p95_time_seconds']:.3f}s")
        summary.append(f"  P99:               {stats['p99_time_seconds']:.3f}s")

        if "std_dev_seconds" in stats:
            summary.append(f"  Std Deviation:     {stats['std_dev_seconds']:.3f}s")

    summary.append("\n" + "=" * 80)

    # Add bottleneck identification
    if metrics:
        slowest_node = next(iter(metrics.items()))
        summary.append("\nBottleneck Analysis:")
        summary.append(f"  Slowest Node (by total time): {slowest_node[0]}")
        summary.append(f"    Total Time: {slowest_node[1]['total_time_seconds']:.3f}s")
        summary.append(f"    Avg Time:   {slowest_node[1]['avg_time_seconds']:.3f}s")

        # Find node with highest average time
        slowest_avg = max(metrics.items(), key=lambda x: x[1]['avg_time_seconds'])
        if slowest_avg[0] != slowest_node[0]:
            summary.append(f"\n  Slowest Node (by avg time): {slowest_avg[0]}")
            summary.append(f"    Avg Time: {slowest_avg[1]['avg_time_seconds']:.3f}s")

    summary.append("=" * 80)

    return "\n".join(summary)
