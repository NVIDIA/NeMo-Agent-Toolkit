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
Advanced bottleneck analysis for a dataframe that contains:
- event_type in {LLM_START, LLM_END, TOOL_START, TOOL_END, ...}
- llm_name
- tool_name
- UUID
- event_timestamp (float or datetime)
- other metadata...

We pair start/end events by UUID, compute operation durations,
then analyze concurrency and produce a summary report.
"""

import numpy as np
import pandas as pd

from nat.data_models.intermediate_step import IntermediateStep
from nat.profiler.inference_optimization.data_models import SimpleBottleneckReport
from nat.profiler.inference_optimization.data_models import SimpleOperationStats
from nat.profiler.utils import create_standardized_dataframe


# ----------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------
def profile_workflow_bottlenecks(all_steps: list[list[IntermediateStep]]) -> SimpleBottleneckReport:
    """
    Perform advanced bottleneck profiling on a workflow dataframe.

    1) Pair LLM_START/LLM_END and TOOL_START/TOOL_END by UUID.
    2) Compute operation durations.
    3) Analyze concurrency (max concurrent usage).
    4) Summarize as SimpleOperationStats and produce a final SimpleBottleneckReport.

    Parameters
    ----------
    all_steps : Intermediate Steps

    Returns
    -------
    SimpleBottleneckReport
        Contains detailed stats per operation and a textual summary of top bottlenecks.
    """
    df = create_standardized_dataframe(all_steps)
    # -------------------------------------------------------------
    # 1) Separate events by operation type and match start/end
    # -------------------------------------------------------------
    required_cols = {"event_type", "UUID", "event_timestamp"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    # We'll unify LLM and TOOL operations into a single set, with:
    #   operation_type = 'LLM' or 'TOOL'
    #   operation_name = llm_name/tool_name
    #   start_time
    #   end_time
    #   duration = end_time - start_time
    # We'll store them in a list of dicts, then convert to DataFrame.
    operations_records = []

    # We'll create a copy to avoid mutating user data
    dfc = df.copy()

    # We define a small helper to map event_type -> (operation_type, which_name_field)
    def get_operation_info(event_type: str) -> str | None:
        """
        Return 'LLM' if event_type starts with 'LLM_', 'TOOL' if event_type starts with 'TOOL_',
        else None (unknown).
        """
        if event_type.startswith("LLM_"):
            return "LLM"
        if event_type.startswith("TOOL_"):
            return "TOOL"
        return None

    # Group by UUID so we can pair each START with the corresponding END
    grouped = dfc.groupby("UUID", as_index=False, group_keys=True)

    for uuid_val, group_df in grouped:
        if len(group_df) < 2:
            # Possibly incomplete or single event, skip
            continue

        # We might have multiple events with the same UUID, but typically we expect:
        #   LLM_START, LLM_END (or TOOL_START, TOOL_END).
        # Sort by timestamp
        group_df = group_df.sort_values("event_timestamp")

        # Identify operation_type from the first row's event_type
        first_event_type = group_df["event_type"].iloc[0]
        operation_type = get_operation_info(first_event_type)
        if not operation_type:
            # unknown or not LLM_/TOOL_
            continue

        # We'll attempt to find the start row and the end row
        # Usually there's exactly 1 start, 1 end
        start_rows = group_df[group_df["event_type"] == f"{operation_type}_START"]
        end_rows = group_df[group_df["event_type"] == f"{operation_type}_END"]

        if len(start_rows) == 0 or len(end_rows) == 0:
            # No matching start/end
            continue

        # We'll just take the earliest start and the latest end for the entire group.
        start_time = start_rows["event_timestamp"].min()
        end_time = end_rows["event_timestamp"].max()
        duration = end_time - start_time

        # For the name, we pick 'llm_name' or 'tool_name' depending on operation_type
        if operation_type == "LLM":
            # Among the rows, pick a non-null llm_name if present
            op_names = group_df["llm_name"].dropna().unique()
            # fallback to a default if none
            operation_name = op_names[0] if len(op_names) else "unknown_llm"
        else:
            op_names = group_df["tool_name"].dropna().unique()
            operation_name = op_names[0] if len(op_names) else "unknown_tool"

        operations_records.append({
            "operation_type": operation_type,
            "operation_name": operation_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "UUID": uuid_val
        })

    if not operations_records:
        # No valid operations found
        return SimpleBottleneckReport(stats={}, summary="No operations found to profile.")

    operations_df = pd.DataFrame(operations_records)

    # -------------------------------------------------------------
    # 2) Concurrency Analysis
    # -------------------------------------------------------------
    # We want to find the maximum concurrency for each operation_name.
    # We'll do a timeline-based approach: for each operation we have a start_time, end_time
    # We'll create +1 event at start_time, -1 event at end_time, then do a running sum.
    # Then we can measure concurrency across the entire timeline. However, we want concurrency
    # specifically *by operation_name* as well as overall.
    #
    # We'll do it in two passes:
    #   A) Overall concurrency ignoring operation_name
    #   B) concurrency per (operation_type, operation_name)
    # Then we can combine them for a "peak concurrency" measure.

    # A) Overall concurrency (not always essential, but might be interesting)
    timeline_events = []
    for row in operations_df.itertuples(index=False):
        timeline_events.append((row.start_time, +1))
        timeline_events.append((row.end_time, -1))

    timeline_events.sort(key=lambda x: x[0])  # sort by time
    current_concurrency = 0
    concurrency_trace = []
    for ts, delta in timeline_events:
        current_concurrency += delta
        concurrency_trace.append((ts, current_concurrency))
    overall_max_concurrency = max(c[1] for c in concurrency_trace) if concurrency_trace else 0

    # B) concurrency by operation_name
    # We'll generate timeline events per operation_name
    # Then compute the max concurrency for that subset
    operation_names = operations_df["operation_name"].unique()
    max_concurrency_by_name = {}

    for op_name in operation_names:
        sub = operations_df[operations_df["operation_name"] == op_name]
        events_sub = []
        for row in sub.itertuples(index=False):
            events_sub.append((row.start_time, +1))
            events_sub.append((row.end_time, -1))
        if not events_sub:
            max_concurrency_by_name[op_name] = 0
            continue
        events_sub.sort(key=lambda x: x[0])
        c_curr = 0
        c_max = 0
        for ts, delta in events_sub:
            c_curr += delta
            if c_curr > c_max:  # noqa: PLR1730 - don't use max built-in
                c_max = c_curr
        max_concurrency_by_name[op_name] = c_max

    # -------------------------------------------------------------
    # 3) Compute summary stats per (operation_type, operation_name)
    # -------------------------------------------------------------
    # We'll gather durations in a list, compute average, p95, p99, etc.

    stats_dict = {}
    grouped_ops = operations_df.groupby(["operation_type", "operation_name"])
    for (op_type, op_name), grp in grouped_ops:
        durations = grp["duration"].values
        usage_count = len(durations)
        avg_duration = durations.mean()
        p95_duration = np.percentile(durations, 95)
        p99_duration = np.percentile(durations, 99)

        # concurrency
        max_concur = max_concurrency_by_name.get(op_name, 0)

        # define a custom "bottleneck_score":
        # We say score = avg_duration * max_concurrency,
        bottleneck_score = float(avg_duration * max_concur)

        # store in dictionary
        key = f"{op_type}:{op_name}"
        stats_dict[key] = SimpleOperationStats(operation_type=op_type,
                                               operation_name=op_name,
                                               usage_count=usage_count,
                                               avg_duration=float(avg_duration),
                                               p95_duration=float(p95_duration),
                                               p99_duration=float(p99_duration),
                                               max_concurrency=int(max_concur),
                                               bottleneck_score=bottleneck_score)

    # -------------------------------------------------------------
    # 4) Produce a textual summary highlighting top bottlenecks
    # -------------------------------------------------------------
    # We'll rank by bottleneck_score descending and show top 3.
    if not stats_dict:
        return SimpleBottleneckReport(stats={}, summary="No stats to report.")

    top_items = sorted(stats_dict.values(), key=lambda x: x.bottleneck_score, reverse=True)
    top_3 = top_items[:3]

    # Build a simple textual summary
    lines = []
    lines.append("---- BOTTLENECK REPORT ----")
    lines.append(f"Total distinct operations found: {len(stats_dict)}")
    lines.append(f"Overall max concurrency (all ops): {overall_max_concurrency}")
    lines.append("Top 3 Bottlenecks by bottleneck_score (avg_duration * max_concurrency):")
    for i, item in enumerate(top_3, start=1):
        lines.append(f"{i}) {item.operation_type} '{item.operation_name}': "
                     f"score={item.bottleneck_score:.2f}, "
                     f"avg_dur={item.avg_duration:.2f}, "
                     f"max_concurrency={item.max_concurrency}")
    summary_report = "\n".join(lines)

    # Construct a final Pydantic model
    return SimpleBottleneckReport(stats=stats_dict, summary=summary_report)
