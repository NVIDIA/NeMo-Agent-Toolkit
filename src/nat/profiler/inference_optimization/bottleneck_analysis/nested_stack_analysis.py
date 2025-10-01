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
"""
An enhanced script that:

1. Groups events by example_number.
2. Builds a nested call tree (stack-based) for each example_number, so calls from different examples never nest.
3. Combines all calls into one global list for concurrency analysis.
4. Computes:

  - self_time, subtree_time for each call
  - concurrency distribution (p50, p90, p95, p99) across all examples
  - each node's midpoint concurrency
  - a custom 'bottleneck_score' (here = subtree_time)

5. Optionally saves a Gantt chart.
6. Returns a Pydantic object with concurrency stats, node metrics, top bottlenecks, and a textual report.
"""

import logging
import os

import pandas as pd

from nat.data_models.intermediate_step import IntermediateStep
from nat.profiler.inference_optimization.data_models import CallNode
from nat.profiler.inference_optimization.data_models import ConcurrencyDistribution
from nat.profiler.inference_optimization.data_models import NestedCallProfilingResult
from nat.profiler.inference_optimization.data_models import NodeMetrics
from nat.profiler.utils import create_standardized_dataframe

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# 1) Build the Nested Call Tree PER EXAMPLE
# --------------------------------------------------------------------------------


def build_call_tree_for_example(example_df: pd.DataFrame) -> list[CallNode]:
    """
    Stack-based approach for a single example:

    1. Sort events by timestamp ascending.
    2. On `*_START` => push a new node, attach to parent's children if stack not empty.
    3. On `*_END` => pop from stack if matches the top's UUID, finalize end_time/duration.

    Returns:
      A list of top-level calls for this example.
    """
    stack: list[CallNode] = []
    top_level_dict: dict[str, CallNode] = {}
    partial_map: dict[str, CallNode] = {}

    def parse_op_type(evt: str) -> str | None:
        evt = evt.upper()
        if evt.startswith("LLM_"):
            return "LLM"
        if evt.startswith("TOOL_"):
            return "TOOL"
        if evt.startswith("FUNCTION_"):
            return "FUNCTION"
        if evt.startswith("SPAN_"):
            return "FUNCTION"
        return None

    def get_op_name(row: pd.Series, op_type: str) -> str:
        if op_type == "LLM":
            return row.get("llm_name") or "unknown_llm"
        if op_type == "FUNCTION":
            return row.get("function_name") or "unknown_function"
        if op_type == "TOOL":
            return row.get("tool_name") or "unknown_tool"

        return "unknown_op"

    for _, row in example_df.iterrows():
        et = row["event_type"].value.upper()
        uuid = str(row["UUID"])
        ts = float(row["event_timestamp"])

        op_type = parse_op_type(et)
        if not op_type:
            # not an LLM_/TOOL_ event => skip
            continue

        if et.endswith("_START"):
            name = get_op_name(row, op_type)
            node = CallNode(uuid=uuid,
                            operation_type=op_type,
                            operation_name=name,
                            start_time=ts,
                            end_time=ts,
                            duration=0.0,
                            children=[],
                            parent=None)
            if stack:
                parent = stack[-1]
                node.parent = parent
                parent.children.append(node)
            else:
                # top-level
                top_level_dict[uuid] = node

            stack.append(node)
            partial_map[uuid] = node

        elif et.endswith("_END"):
            if uuid not in partial_map:
                # no known start => skip
                continue
            node = partial_map[uuid]
            if stack and stack[-1].uuid == uuid:
                stack.pop()

            node.end_time = ts
            node.duration = max(0.0, ts - node.start_time)
            del partial_map[uuid]

    # partial calls remain in stack => they have no final end_time
    # we won't forcibly remove them

    # collect top-level nodes
    roots = []
    for _, node in top_level_dict.items():
        if node.parent is None:
            roots.append(node)

    return roots


def build_call_tree_per_example(all_steps: list[list[IntermediateStep]]) -> list[CallNode]:
    """
    1) Group the DataFrame by example_number.
    2) For each example, build a separate stack-based call tree.
    3) Return a combined list of all top-level calls from all examples.

    This ensures no cross-example nesting.
    """
    df = create_standardized_dataframe(all_steps)
    required = {"example_number", "event_type", "UUID", "event_timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Sort globally first (so each example is also in ascending time)
    dfc = df.copy()
    dfc.sort_values(["example_number", "event_timestamp"], inplace=True)

    # We'll collect top-level calls for each example
    all_roots: list[CallNode] = []

    for _, group_df in dfc.groupby("example_number"):
        # Build the call tree for this single example
        # group_df is already sorted within this example
        roots_for_example = build_call_tree_for_example(group_df)
        all_roots.extend(roots_for_example)

    return all_roots


# --------------------------------------------------------------------------------
# 2) Concurrency Computation
# --------------------------------------------------------------------------------


def compute_time_based_concurrency(roots: list[CallNode]) -> ConcurrencyDistribution:
    """
    Build a timeline of (start, +1), (end, -1) from all calls, then:
      - Sort events by time
      - Create segments [ (t_i, t_{i+1}, concurrency) ]
      - Compute concurrency percentiles (p50, p90, p95, p99) based on total time spent at each concurrency.
      - This concurrency is across ALL calls from ALL examples.

    Returns:
    --------
    ConcurrencyDistribution
        with the piecewise segments + concurrency percentiles.
    """
    # Flatten
    all_nodes = []

    def dfs(n: CallNode):
        all_nodes.append(n)
        for c in n.children:
            dfs(c)

    for r in roots:
        dfs(r)

    if not all_nodes:
        return ConcurrencyDistribution(timeline_segments=[], p50=0, p90=0, p95=0, p99=0)

    events = []
    for n in all_nodes:
        st = n.start_time
        et = n.end_time
        if st > et:
            # partial or invalid => skip
            continue
        events.append((st, +1))
        events.append((et, -1))

    events.sort(key=lambda x: x[0])
    timeline_segments: list[tuple[float, float, int]] = []
    curr_concurrency = 0
    prev_time = events[0][0]

    for _, (t, delta) in enumerate(events):
        if t > prev_time:
            # segment is [prev_time, t) at concurrency=curr_concurrency
            timeline_segments.append((prev_time, t, curr_concurrency))
        curr_concurrency += delta
        prev_time = t

    # Summaries
    total_time = 0.0
    concurrency_durations: dict[int, float] = {}

    for (seg_start, seg_end, c_val) in timeline_segments:
        length = seg_end - seg_start
        if length <= 0:
            continue
        total_time += length
        concurrency_durations[c_val] = concurrency_durations.get(c_val, 0) + length

    if total_time <= 0:
        return ConcurrencyDistribution(timeline_segments=timeline_segments, p50=0, p90=0, p95=0, p99=0)

    # Build concurrency-level distribution
    sorted_levels = sorted(concurrency_durations.items(), key=lambda x: x[0])  # ascending concurrency

    def concurrency_at_percentile(p: float) -> float:
        threshold = total_time * (p / 100.0)
        accum = 0.0
        last_c = 0
        for c_val, c_dur in sorted_levels:
            accum += c_dur
            if accum >= threshold:
                return float(c_val)
            last_c = c_val
        return float(last_c)

    p50_val = concurrency_at_percentile(50)
    p90_val = concurrency_at_percentile(90)
    p95_val = concurrency_at_percentile(95)
    p99_val = concurrency_at_percentile(99)

    return ConcurrencyDistribution(timeline_segments=timeline_segments,
                                   p50=p50_val,
                                   p90=p90_val,
                                   p95=p95_val,
                                   p99=p99_val)


def find_midpoint_concurrency(node: CallNode, segments: list[tuple[float, float, int]]) -> float:
    """
    Approximate concurrency for a node by finding the concurrency in timeline_segments
    at the node's midpoint (or start if zero-length).
    """
    if node.start_time >= node.end_time:
        mid = node.start_time
    else:
        mid = 0.5 * (node.start_time + node.end_time)

    # Binary search in segments
    left, right = 0, len(segments) - 1
    while left <= right:
        mid_idx = (left + right) // 2
        seg_start, seg_end, seg_conc = segments[mid_idx]
        if seg_start <= mid < seg_end:
            return float(seg_conc)
        if mid < seg_start:
            right = mid_idx - 1
        else:
            left = mid_idx + 1
    return 0.0


# --------------------------------------------------------------------------------
# 3) Gantt Chart
# --------------------------------------------------------------------------------


def save_gantt_chart(all_nodes: list[CallNode], output_path: str) -> None:
    """
    Save a Gantt chart as a PNG, color-coded by operation_type.
    Each node is displayed as a horizontal bar from start_time to end_time.
    The y-axis is the node index (sorted by start_time).
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is not installed. Please install matplotlib to use generate plots for the profiler "
                     "or install \"nvidia-nat[profiler]\" to install all necessary profiling packages.")

        raise

    # Sort calls by start_time
    sorted_nodes = sorted(all_nodes, key=lambda x: x.start_time)
    min_start = sorted_nodes[0].start_time
    max_end = max(node.end_time for node in sorted_nodes)

    color_map = {
        "LLM": "tab:blue",
        "TOOL": "tab:green",
        "FUNCTION": "tab:orange",
    }
    default_color = "tab:gray"

    fig, ax = plt.subplots(figsize=(20, 15))

    y_positions = range(len(sorted_nodes))
    labels = []
    for i, node in enumerate(sorted_nodes):
        start = node.start_time
        width = node.end_time - node.start_time
        c = color_map.get(node.operation_type, default_color)
        ax.barh(y=i, width=width, left=start - min_start, height=0.6, color=c, edgecolor="black")
        labels.append(f"{node.operation_type}:{node.operation_name}")

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max_end - min_start)
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart of Nested Calls (All Examples)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------
# 4) Analysis & Final Pydantic Result
# --------------------------------------------------------------------------------


def analyze_calls_and_build_result(roots: list[CallNode], output_dir: str | None = None) -> NestedCallProfilingResult:
    """
    1. Compute concurrency distribution (p50, p90, p95, p99) across ALL calls in all examples.
    2. For each node, compute self_time, subtree_time, concurrency at midpoint, bottleneck_score.
    3. Identify top 5 bottlenecks (by subtree_time).
    4. Build a textual report.
    5. Optionally save a Gantt chart to 'output_dir'.

    Returns NestedCallProfilingResult.
    """
    if not roots:
        empty_concurrency = ConcurrencyDistribution(timeline_segments=[], p50=0, p90=0, p95=0, p99=0)
        return NestedCallProfilingResult(concurrency=empty_concurrency,
                                         node_metrics={},
                                         top_bottlenecks=[],
                                         textual_report="No calls found.")

    # Flatten all calls
    all_nodes: list[CallNode] = []

    def dfs(n: CallNode):
        all_nodes.append(n)
        for c in n.children:
            dfs(c)

    for r in roots:
        dfs(r)

    # 1) concurrency across all calls
    concurrency_info = compute_time_based_concurrency(roots)

    # 2) build NodeMetrics
    node_metrics_map: dict[str, NodeMetrics] = {}
    for node in all_nodes:
        self_t = node.compute_self_time()
        subtree_t = node.compute_subtree_time()
        bscore = subtree_t
        mid_conc = find_midpoint_concurrency(node, concurrency_info.timeline_segments)

        m = NodeMetrics(uuid=node.uuid,
                        operation_type=node.operation_type,
                        operation_name=node.operation_name,
                        start_time=node.start_time,
                        end_time=node.end_time,
                        duration=node.duration,
                        self_time=self_t,
                        subtree_time=subtree_t,
                        concurrency_midpoint=mid_conc,
                        bottleneck_score=bscore)
        node_metrics_map[node.uuid] = m

    # 3) top 5
    all_metrics = list(node_metrics_map.values())
    sorted_metrics = sorted(all_metrics, key=lambda x: x.bottleneck_score, reverse=True)
    top_5 = sorted_metrics[:5]

    # 4) textual report
    lines = []
    lines.append("=== Multi-Example Nested Call Profiling Report ===")
    lines.append(f"Total calls (across all examples): {len(all_nodes)}")

    lines.append("\n-- Concurrency Distribution (all examples) --")
    lines.append(f"p50={concurrency_info.p50:.1f}, p90={concurrency_info.p90:.1f}, "
                 f"p95={concurrency_info.p95:.1f}, p99={concurrency_info.p99:.1f}")

    lines.append("\n-- Top 5 Calls by Bottleneck Score (subtree_time) --")
    for i, tm in enumerate(top_5, start=1):
        lines.append(f"{i}) UUID={tm.uuid}, {tm.operation_type} '{tm.operation_name}', "
                     f"dur={tm.duration:.2f}, self_time={tm.self_time:.2f}, "
                     f"subtree_time={tm.subtree_time:.2f}, concurrency={tm.concurrency_midpoint:.1f}, "
                     f"score={tm.bottleneck_score:.2f}")

    lines.append("\n-- Full Tree(s) (All Examples) --")

    for root in roots:
        lines.append(str(root))

    report_text = "\n".join(lines)

    # 5) optional Gantt chart
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        chart_path = os.path.join(output_dir, "gantt_chart.png")
        save_gantt_chart(all_nodes, chart_path)

    # Return the final Pydantic result
    return NestedCallProfilingResult(concurrency=concurrency_info,
                                     node_metrics=node_metrics_map,
                                     top_bottlenecks=top_5,
                                     textual_report=report_text)


def multi_example_call_profiling(all_steps: list[list[IntermediateStep]],
                                 output_dir: str | None = None) -> NestedCallProfilingResult:
    """
    The high-level function:

    1. Build a forest of calls by grouping by example_number (so no cross-example nesting).
    2. Analyze concurrency across all calls in all examples.
    3. Return a NestedCallProfilingResult with concurrency distribution, node metrics, top bottlenecks, and textual
       report. Optionally saves a Gantt chart.

    :param all_steps: Intermediate steps for each example.
    :param output_dir: Directory path to save gantt_chart.png (if provided)
    :return: NestedCallProfilingResult (pydantic)
    """
    # Build the forest (all examples combined)
    roots = build_call_tree_per_example(all_steps)
    # Analyze calls
    result = analyze_calls_and_build_result(roots, output_dir=output_dir)
    return result
