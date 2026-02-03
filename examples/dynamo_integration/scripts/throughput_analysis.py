#!/usr/bin/env python3
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
Calculate completion tokens per second and inter-token latency from NAT profiler CSV output.

This script works around the issue where LangChain's ChatNVIDIA integration
doesn't populate usage_metadata, by counting LLM_NEW_TOKEN events instead.


Sample console output from a previous run:

================================================================================
LLM Performance Analysis Summary
================================================================================

Dataset Overview:
  Total LLM Calls:        874
  Total Tokens Generated: 107,905
  Sum of LLM Durations:   2632.03s
  Wall-Clock Time:        88.99s
  Concurrent Examples:    100

---------------------------Time To First Token (TTFT)---------------------------
  Mean:     133.78 ms
  Median:   105.58 ms
  P90:      206.35 ms
  P95:      340.47 ms
  P99:      614.93 ms
  Min:       55.01 ms
  Max:      984.04 ms

------------Inter-Token Latency (ITL) / Time Per Output Token (TPOT)------------
  Global Statistics (across all 107,031 token intervals):
    Mean:      23.43 ms
    Median:    16.96 ms
    P90:       46.06 ms
    P95:       58.35 ms
    P99:      103.72 ms
    Min:        0.21 ms
    Max:     1070.66 ms

  Per-Call Average ITL:
    Mean:      23.47 ms
    Median:    23.93 ms
    P90:       26.27 ms
    P95:       27.44 ms

-------------------Per-Request Throughput (Tokens Per Second)-------------------
  Mean:      41.78 tok/s
  Median:    40.55 tok/s
  P90:       46.76 tok/s
  P95:       60.99 tok/s
  P99:       75.84 tok/s
  Min:       29.67 tok/s
  Max:       87.26 tok/s

-----------------Aggregate Throughput (All Concurrent Requests)-----------------
  Wall-Clock Time:           88.99 s
  Total Tokens:            107,905
  Aggregate Throughput:    1212.57 tok/s
  Speedup vs Per-Request:    29.02x
================================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_tokens_per_second(
    csv_path: str, ) -> tuple[pd.DataFrame | None, np.ndarray | None, dict[str, float | int] | None]:
    """Calculate tokens/sec and inter-token latency for each LLM call from NEW_TOKEN events.

    Returns:
        tuple: (results_df, all_itls_array, aggregate_stats)
            - results_df: DataFrame with per-call metrics
            - all_itls_array: Array of all inter-token latencies
            - aggregate_stats: Dict with aggregate throughput metrics
    """

    df = pd.read_csv(csv_path)

    results = []
    all_itls = []  # Collect all inter-token latencies across all calls

    # Group by example_number to process each workflow run separately
    for example_num in df['example_number'].unique():
        example_df = df[df['example_number'] == example_num].copy()

        # Sort by timestamp to ensure correct ordering
        example_df = example_df.sort_values('event_timestamp')

        # Track LLM call boundaries
        llm_call_id = 0
        current_start = None

        for _, row in example_df.iterrows():
            if row['event_type'] == 'LLM_START':
                current_start = row['event_timestamp']
                llm_call_id += 1

            elif row['event_type'] == 'LLM_END' and current_start is not None:
                # Find all NEW_TOKEN events for this LLM call
                tokens = example_df[(example_df['event_type'] == 'LLM_NEW_TOKEN')
                                    & (example_df['event_timestamp'] > current_start) &
                                    (example_df['event_timestamp']
                                     <= row['event_timestamp'])].sort_values('event_timestamp')

                num_tokens = len(tokens)
                duration = row['event_timestamp'] - current_start

                if duration > 0 and num_tokens > 0:
                    tokens_per_sec = num_tokens / duration

                    # Get time to first token (TTFT)
                    token_times = tokens['event_timestamp'].values
                    ttft = token_times[0] - current_start

                    # Calculate inter-token latency (ITL)
                    mean_itl = None
                    median_itl = None
                    if num_tokens > 1:
                        itls = np.diff(token_times)  # Time between consecutive tokens
                        mean_itl = np.mean(itls)
                        median_itl = np.median(itls)
                        all_itls.extend(itls)  # Collect for global stats

                    results.append({
                        'example_num': example_num,
                        'llm_call_id': llm_call_id,
                        'llm_name': row['llm_name'],
                        'start_time': current_start,
                        'end_time': row['event_timestamp'],
                        'duration_sec': duration,
                        'num_tokens': num_tokens,
                        'tokens_per_sec': tokens_per_sec,
                        'time_to_first_token_sec': ttft,
                        'mean_itl_sec': mean_itl,
                        'median_itl_sec': median_itl
                    })

                current_start = None

    if not results:
        print("No LLM calls with token data found!")
        return None, None, None

    results_df = pd.DataFrame(results)
    all_itls_array = np.array(all_itls) if all_itls else None

    # Calculate aggregate throughput across all concurrent requests
    # Wall-clock time = time from first LLM start to last LLM end
    wall_clock_start = results_df['start_time'].min()
    wall_clock_end = results_df['end_time'].max()
    wall_clock_time = wall_clock_end - wall_clock_start
    total_tokens = results_df['num_tokens'].sum()

    # Aggregate throughput = total tokens / wall-clock time
    aggregate_throughput = total_tokens / wall_clock_time if wall_clock_time > 0 else 0

    # Store aggregate statistics
    aggregate_stats = {
        'wall_clock_time_sec': wall_clock_time,
        'total_tokens': total_tokens,
        'aggregate_throughput_toks': aggregate_throughput,
        'num_concurrent_examples': len(results_df['example_num'].unique()),
        'total_llm_calls': len(results_df),
        'sum_of_durations_sec': results_df['duration_sec'].sum()
    }

    # Print summary statistics
    print(f"\n{'='*80}")
    print("LLM Performance Analysis Summary")
    print(f"{'='*80}")
    print("\nDataset Overview:")
    print(f"  Total LLM Calls:        {len(results_df)}")
    print(f"  Total Tokens Generated: {total_tokens:,}")
    print(f"  Sum of LLM Durations:   {results_df['duration_sec'].sum():.2f}s")
    print(f"  Wall-Clock Time:        {wall_clock_time:.2f}s")
    print(f"  Concurrent Examples:    {aggregate_stats['num_concurrent_examples']}")

    print(f"\n{'Time To First Token (TTFT)':-^80}")
    print(f"  Mean:   {results_df['time_to_first_token_sec'].mean()*1000:>8.2f} ms")
    print(f"  Median: {results_df['time_to_first_token_sec'].median()*1000:>8.2f} ms")
    print(f"  P90:    {results_df['time_to_first_token_sec'].quantile(0.90)*1000:>8.2f} ms")
    print(f"  P95:    {results_df['time_to_first_token_sec'].quantile(0.95)*1000:>8.2f} ms")
    print(f"  P99:    {results_df['time_to_first_token_sec'].quantile(0.99)*1000:>8.2f} ms")
    print(f"  Min:    {results_df['time_to_first_token_sec'].min()*1000:>8.2f} ms")
    print(f"  Max:    {results_df['time_to_first_token_sec'].max()*1000:>8.2f} ms")

    if all_itls_array is not None and len(all_itls_array) > 0:
        print(f"\n{'Inter-Token Latency (ITL) / Time Per Output Token (TPOT)':-^80}")
        print(f"  Global Statistics (across all {len(all_itls_array):,} token intervals):")
        print(f"    Mean:   {np.mean(all_itls_array)*1000:>8.2f} ms")
        print(f"    Median: {np.median(all_itls_array)*1000:>8.2f} ms")
        print(f"    P90:    {np.percentile(all_itls_array, 90)*1000:>8.2f} ms")
        print(f"    P95:    {np.percentile(all_itls_array, 95)*1000:>8.2f} ms")
        print(f"    P99:    {np.percentile(all_itls_array, 99)*1000:>8.2f} ms")
        print(f"    Min:    {np.min(all_itls_array)*1000:>8.2f} ms")
        print(f"    Max:    {np.max(all_itls_array)*1000:>8.2f} ms")

        # Filter out None values for per-call ITL stats
        call_mean_itls = results_df['mean_itl_sec'].dropna()
        if len(call_mean_itls) > 0:
            print("\n  Per-Call Average ITL:")
            print(f"    Mean:   {call_mean_itls.mean()*1000:>8.2f} ms")
            print(f"    Median: {call_mean_itls.median()*1000:>8.2f} ms")
            print(f"    P90:    {call_mean_itls.quantile(0.90)*1000:>8.2f} ms")
            print(f"    P95:    {call_mean_itls.quantile(0.95)*1000:>8.2f} ms")

    print(f"\n{'Per-Request Throughput (Tokens Per Second)':-^80}")
    print(f"  Mean:   {results_df['tokens_per_sec'].mean():>8.2f} tok/s")
    print(f"  Median: {results_df['tokens_per_sec'].median():>8.2f} tok/s")
    print(f"  P90:    {results_df['tokens_per_sec'].quantile(0.90):>8.2f} tok/s")
    print(f"  P95:    {results_df['tokens_per_sec'].quantile(0.95):>8.2f} tok/s")
    print(f"  P99:    {results_df['tokens_per_sec'].quantile(0.99):>8.2f} tok/s")
    print(f"  Min:    {results_df['tokens_per_sec'].min():>8.2f} tok/s")
    print(f"  Max:    {results_df['tokens_per_sec'].max():>8.2f} tok/s")

    print(f"\n{'Aggregate Throughput (All Concurrent Requests)':-^80}")
    print(f"  Wall-Clock Time:        {wall_clock_time:>8.2f} s")
    print(f"  Total Tokens:           {total_tokens:>8,}")
    print(f"  Aggregate Throughput:   {aggregate_throughput:>8.2f} tok/s")
    print(f"  Speedup vs Per-Request: {aggregate_throughput / results_df['tokens_per_sec'].mean():>8.2f}x")

    print(f"{'='*80}\n")

    return results_df, all_itls_array, aggregate_stats


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python throughput_analysis.py <path_to_standardized_data_all.csv>")
        print("\nExample:")
        print("  python throughput_analysis.py outputs/dynamo_evals/jobs/job_*/standardized_data_all.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    results_df, all_itls, aggregate_stats = calculate_tokens_per_second(csv_path)

    if results_df is not None:
        # Save per-LLM-call results with throughput metrics
        output_path = Path(csv_path).parent / "tokens_per_second_analysis.csv"
        results_df.to_csv(output_path, index=False)
        print(f"Per-LLM-call analysis saved to: {output_path}")

        # Save global ITL distribution
        if all_itls is not None and len(all_itls) > 0:
            itl_output_path = Path(csv_path).parent / "inter_token_latency_distribution.csv"
            pd.DataFrame({'itl_sec': all_itls}).to_csv(itl_output_path, index=False)
            print(f"ITL distribution saved to: {itl_output_path}")
