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

1. Groups workflow events by example_number to build a per-example call tree (no cross-example nesting).
2. Tracks concurrency globally across *all* examples.
3. Identifies concurrency "spikes" (concurrency >= a threshold).
4. Correlates concurrency spikes with token usage and call metadata.
5. Computes average call latency by concurrency level, using midpoint concurrency as an approximation.
6. Returns a Pydantic result containing concurrency distribution, spike intervals, correlation stats, etc.,
   along with a textual report containing the real call count, active calls in spikes, etc.

Changes from previous version:

- Now shows the actual total calls in the dataset.
- Displays the real number of active calls for each spike interval.
- Computes and reports average latency by concurrency (no visualization).

"""

import numpy as np
import pandas as pd

from nat.data_models.intermediate_step import IntermediateStep
from nat.profiler.inference_optimization.data_models import ConcurrencyAnalysisResult
from nat.profiler.inference_optimization.data_models import ConcurrencyCallNode
from nat.profiler.inference_optimization.data_models import ConcurrencyCorrelationStats
from nat.profiler.inference_optimization.data_models import ConcurrencySpikeInfo
from nat.profiler.utils import create_standardized_dataframe

# --------------------------------------------------------------------------------
# 1) Building the Per-Example Call Trees
# --------------------------------------------------------------------------------


def build_call_tree_for_example(example_df: pd.DataFrame) -> list[ConcurrencyCallNode]:
    """
    Sort events by time, push on `*_START`, pop on `*_END`, build stack-based calls for a single example.
    """
    stack: list[ConcurrencyCallNode] = []
    top_level: dict[str, ConcurrencyCallNode] = {}
    partial_map: dict[str, ConcurrencyCallNode] = {}

    def parse_op_type(et: str) -> str | None:
        et = et.upper()
        if et.startswith("LLM_"):
            return "LLM"
        if et.startswith("TOOL_"):
            return "TOOL"
        return None

    def get_op_name(row: pd.Series, op_type: str) -> str:
        if op_type == "LLM":
            return row.get("llm_name") or "unknown_llm"
        if op_type == "TOOL":
            return row.get("tool_name") or "unknown_tool"
        return "unknown_op"

    example_num = int(example_df["example_number"].iloc[0])

    for _, row in example_df.iterrows():
        et = row["event_type"].value.upper()
        uuid = str(row["UUID"])
        ts = float(row["event_timestamp"])
        op_type = parse_op_type(et)
        if not op_type:
            continue

        if et.endswith("_START"):
            op_name = get_op_name(row, op_type)
            node = ConcurrencyCallNode(
                uuid=uuid,
                example_number=example_num,
                operation_type=op_type,
                operation_name=op_name,
                start_time=ts,
                end_time=ts,  # updated on END
                duration=0.0)
            if stack:
                parent = stack[-1]
                node.parent = parent
                parent.children.append(node)
            else:
                top_level[uuid] = node

            stack.append(node)
            partial_map[uuid] = node

        elif et.endswith("_END"):
            if uuid not in partial_map:
                continue
            node = partial_map[uuid]
            node.end_time = ts
            node.duration = max(0.0, node.end_time - node.start_time)
            node.prompt_tokens = row.get("prompt_tokens")
            node.completion_tokens = row.get("completion_tokens")
            node.total_tokens = row.get("total_tokens")
            node.tool_outputs = row.get("metadata").get("tool_outputs") if (
                row.get("metadata") and row.get("metadata").get("tool_outputs")) else None
            node.llm_text_output = row.get("llm_text_output")

            if stack and stack[-1].uuid == uuid:
                stack.pop()
            del partial_map[uuid]

    # gather top-level
    roots = []
    for _, nd in top_level.items():
        if nd.parent is None:
            roots.append(nd)
    return roots


def build_call_tree_per_example(df: pd.DataFrame) -> list[ConcurrencyCallNode]:
    """
    Groups by example_number, builds separate call trees, returns combined list of top-level calls.
    """
    req_cols = {"example_number", "event_type", "UUID", "event_timestamp"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    dfc = df.copy()
    dfc.sort_values(["example_number", "event_timestamp"], inplace=True)

    all_roots: list[ConcurrencyCallNode] = []
    for _, grp in dfc.groupby("example_number"):
        r = build_call_tree_for_example(grp)
        all_roots.extend(r)
    return all_roots


def flatten_calls(roots: list[ConcurrencyCallNode]) -> list[ConcurrencyCallNode]:
    """
    DFS to produce a flat list of all calls (including nested).
    """
    all_nodes = []

    def dfs(n: ConcurrencyCallNode):
        all_nodes.append(n)
        for c in n.children:
            dfs(c)

    for rt in roots:
        dfs(rt)
    return all_nodes


# --------------------------------------------------------------------------------
# 2) Global Concurrency Distribution & Segments
# --------------------------------------------------------------------------------


def compute_concurrency_distribution(roots: list[ConcurrencyCallNode]) -> dict[int, float]:
    """
    Flatten calls, produce (start, +1)/(end, -1), accumulate total time at each concurrency level.
    """
    all_nodes = flatten_calls(roots)
    if not all_nodes:
        return {}

    events = []
    for n in all_nodes:
        if n.start_time <= n.end_time:
            events.append((n.start_time, +1))
            events.append((n.end_time, -1))

    events.sort(key=lambda x: x[0])
    dist_map: dict[int, float] = {}
    curr_conc = 0
    prev_time = events[0][0]

    for (time_val, delta) in events:
        if time_val > prev_time:
            length = time_val - prev_time
            dist_map[curr_conc] = dist_map.get(curr_conc, 0.0) + length
        curr_conc += delta
        prev_time = time_val

    return dist_map


def build_concurrency_segments(roots: list[ConcurrencyCallNode]) -> list[tuple[float, float, int]]:
    """
    Return piecewise segments of (start, end, concurrency) across all calls.
    """
    all_nodes = flatten_calls(roots)
    if not all_nodes:
        return []

    events = []
    for n in all_nodes:
        if n.start_time <= n.end_time:
            events.append((n.start_time, +1))
            events.append((n.end_time, -1))

    events.sort(key=lambda x: x[0])
    segments: list[tuple[float, float, int]] = []
    curr_conc = 0
    prev_time = events[0][0]

    for (t, delta) in events:
        if t > prev_time:
            segments.append((prev_time, t, curr_conc))
        curr_conc += delta
        prev_time = t

    return segments


def find_percentile_concurrency(dist_map: dict[int, float], percentile: float) -> float:
    """
    concurrency => total_time -> find concurrency level at given percentile of total time.
    """
    total_time = sum(dist_map.values())
    if total_time <= 0:
        return 0.0

    items = sorted(dist_map.items(), key=lambda x: x[0])  # ascending concurrency
    threshold = percentile * 0.01 * total_time
    accum = 0.0
    last_c = 0

    for c_val, dur in items:
        accum += dur
        if accum >= threshold:
            return float(c_val)
        last_c = c_val
    return float(last_c)


# --------------------------------------------------------------------------------
# 3) Spike Detection & Active Calls
# --------------------------------------------------------------------------------


def detect_concurrency_spikes(segments: list[tuple[float, float, int]], threshold: int) -> list[ConcurrencySpikeInfo]:
    """
    If concurrency >= threshold, label that segment a 'spike'.
    """
    spikes = []
    for (s, e, c_val) in segments:
        if c_val >= threshold and e > s:
            sp = ConcurrencySpikeInfo(start_time=s, end_time=e, concurrency=c_val)
            spikes.append(sp)
    return spikes


def find_calls_active_in_interval(roots: list[ConcurrencyCallNode], start_t: float,
                                  end_t: float) -> list[ConcurrencyCallNode]:
    """
    Return all calls overlapping [start_t, end_t).
    Overlap => not (call.end_time <= start_t or call.start_time >= end_t).
    """
    results = []
    all_nodes = flatten_calls(roots)
    for n in all_nodes:
        if not (n.end_time <= start_t or n.start_time >= end_t):
            results.append(n)
    return results


# --------------------------------------------------------------------------------
# 4) Correlations & Average Latency by Concurrency
# --------------------------------------------------------------------------------

def correlate_spike_calls(spikes: list[ConcurrencySpikeInfo], roots: list[ConcurrencyCallNode]) \
        -> ConcurrencyCorrelationStats:
    """
    For each spike, gather calls that overlap, compute average prompt_tokens, total_tokens across them.
    """
    p_tokens = []
    t_tokens = []

    for sp in spikes:
        active = find_calls_active_in_interval(roots, sp.start_time, sp.end_time)
        # record the active call uuids for each spike
        sp.active_uuids = list({c.uuid for c in active})

        for c in active:
            if c.prompt_tokens and c.prompt_tokens > 0:
                p_tokens.append(c.prompt_tokens)
            if c.total_tokens and c.total_tokens > 0:
                t_tokens.append(c.total_tokens)

    def safe_avg(lst):
        return float(np.mean(lst)) if lst else 0.0

    return ConcurrencyCorrelationStats(
        avg_prompt_tokens=safe_avg(p_tokens),
        avg_total_tokens=safe_avg(t_tokens),
    )


def compute_midpoint_concurrency(n: ConcurrencyCallNode, segments: list[tuple[float, float, int]]) -> float:
    """
    Approx concurrency at the midpoint of this call.
    """
    if n.start_time >= n.end_time:
        return 0.0
    mid = 0.5 * (n.start_time + n.end_time)

    # binary or linear search
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


def average_latency_by_midpoint_concurrency(roots: list[ConcurrencyCallNode]) -> dict[int, float]:
    """
    For each call, find concurrency at midpoint, then bucket durations by concurrency, compute avg.
    """
    segs = build_concurrency_segments(roots)
    all_nodes = flatten_calls(roots)

    # concurrency => list of durations
    from collections import defaultdict
    calls_by_conc = defaultdict(list)

    for c in all_nodes:
        mc = compute_midpoint_concurrency(c, segs)
        # round or cast to int
        c_level = int(mc)
        calls_by_conc[c_level].append(c.duration)

    result = {}
    for c_level, durations in calls_by_conc.items():
        if durations:
            result[c_level] = float(np.mean(durations))
        else:
            result[c_level] = 0.0
    return result


# --------------------------------------------------------------------------------
# 5) Main Analysis Function
# --------------------------------------------------------------------------------


def concurrency_spike_analysis(
    all_steps: list[list[IntermediateStep]],
    concurrency_spike_threshold: int | None = None,
) -> ConcurrencyAnalysisResult:
    """
    1) Build per-example call trees (no cross-example nesting).
    2) Compute concurrency distribution & concurrency segments across *all* calls.
    3) Derive concurrency percentiles (p50, p90, p95, p99).
    4) If threshold not provided, pick e.g. ceil of p90 concurrency.
    5) Detect spikes, gather calls in those intervals => correlation stats.
    6) Also compute average latency by concurrency and add to report.
    7) Return a Pydantic object with everything, plus a textual report.
    """
    df = create_standardized_dataframe(all_steps)
    required_cols = {
        "framework",
        "llm_name",
        "llm_text_input",
        "llm_text_output",
        "event_timestamp",
        "event_type",
        "UUID",
        "example_number",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Build global forest
    roots = build_call_tree_per_example(df)
    all_calls = flatten_calls(roots)
    num_calls = len(all_calls)

    # Concurrency distribution
    dist_map = compute_concurrency_distribution(roots)
    total_time = sum(dist_map.values())

    p50_c = find_percentile_concurrency(dist_map, 50)
    p90_c = find_percentile_concurrency(dist_map, 90)
    p95_c = find_percentile_concurrency(dist_map, 95)
    p99_c = find_percentile_concurrency(dist_map, 99)

    # Threshold
    if concurrency_spike_threshold is None:
        concurrency_spike_threshold = max(1, int(np.ceil(p90_c)))

    # Build concurrency segments, detect spikes
    segments = build_concurrency_segments(roots)
    spike_intervals = detect_concurrency_spikes(segments, concurrency_spike_threshold)

    # Correlate
    corr_stats = correlate_spike_calls(spike_intervals, roots)

    # Average latency by concurrency
    avg_lat_by_conc = average_latency_by_midpoint_concurrency(roots)

    # Build textual report
    lines = []
    lines.append("=== Concurrency Spike Analysis ===")
    lines.append(f"Total calls in dataset: {num_calls}")
    lines.append(f"Total time observed: {total_time:.2f} units (sum of concurrency timeline)")

    lines.append("\n-- Concurrency Distribution --")
    for c_val in sorted(dist_map.keys()):
        dur = dist_map[c_val]
        lines.append(f"  concurrency={c_val}: {dur:.2f} time")

    lines.append(f"\nPercentiles => p50={p50_c:.1f}, p90={p90_c:.1f}, p95={p95_c:.1f}, p99={p99_c:.1f}")
    lines.append(f"Spike threshold chosen: {concurrency_spike_threshold}")

    lines.append("\n-- Detected Spike Intervals --")
    if not spike_intervals:
        lines.append("No intervals exceed concurrency spike threshold.")
    else:
        for i, sp in enumerate(spike_intervals, start=1):
            length = sp.end_time - sp.start_time
            active_count = len(sp.active_uuids)
            lines.append(f"{i}) {sp.start_time:.2f}-{sp.end_time:.2f}, concurrency={sp.concurrency}, "
                         f"length={length:.2f}, #active_calls={active_count}")

    lines.append("\n-- Correlation Stats for Spiked Calls --")
    lines.append(f"Avg prompt_tokens in spike calls: {corr_stats.avg_prompt_tokens:.1f}")
    lines.append(f"Avg total_tokens in spike calls : {corr_stats.avg_total_tokens:.1f}")

    lines.append("\n-- Average Latency by Midpoint Concurrency --")
    if not avg_lat_by_conc:
        lines.append("No calls or no concurrency data.")
    else:
        for c_level in sorted(avg_lat_by_conc.keys()):
            lat = avg_lat_by_conc[c_level]
            lines.append(f"  concurrency={c_level} => avg_latency={lat:.2f}")

    final_report = "\n".join(lines)

    # Build result object
    return ConcurrencyAnalysisResult(concurrency_distribution=dist_map,
                                     p50_concurrency=p50_c,
                                     p90_concurrency=p90_c,
                                     p95_concurrency=p95_c,
                                     p99_concurrency=p99_c,
                                     spike_threshold=concurrency_spike_threshold,
                                     spike_intervals=spike_intervals,
                                     correlation_stats=corr_stats,
                                     textual_report=final_report,
                                     average_latency_by_concurrency=avg_lat_by_conc)
