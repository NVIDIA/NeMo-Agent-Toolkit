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
Histogram plotting script for throughput metrics distribution analysis.

This script creates histograms showing the distribution of throughput metrics,
similar to plot_throughput_vs_tsq_per_request.py but with count on Y-axis instead of TSQ.

Features:
- Per-LLM-call histograms (TTFT, ITL, Throughput) showing distribution of every individual LLM call
- Per-request aggregate histograms (Total Tokens, LLM Calls, Duration)
- Statistical annotations: median lines (dotted), P10/P90 percentiles in stats box
- Each job plotted as a separate histogram with its own color
- Legend with per-job statistics (n, mean, median, std, P10, P90)

Usage:
    # Single job (or multiple jobs in jobs/ directory)
    python plot_throughput_histograms_per_request.py ./outputs/dynamo_evals/experiment1

    # Custom output directory
    python plot_throughput_histograms_per_request.py ./outputs/exp1 --output ./comparison

Example:
    python plot_throughput_histograms_per_request.py ./outputs/dynamo_evals/banking_data_eval_full_test/jobs/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

# Maximum TTFT value to display in histograms (milliseconds)
MAX_TTFT_MS = 500
# Bin width for TTFT histograms (milliseconds) - ensures good resolution in visible range
TTFT_BIN_WIDTH_MS = 5


def get_job_label(job_dir_name: str) -> str:
    """Extract short label from job directory name (first 7 chars)."""
    return job_dir_name[:7]


def get_experiment_label(dir_path: Path) -> str:
    """Extract a short label from the experiment directory name."""
    return dir_path.name


def extract_per_request_tsq_scores(job_dir: Path) -> dict[int, dict] | None:
    """Extract individual TSQ scores from tool_selection_quality_output.json.

    Returns a dict mapping example_number to {score, id, reasoning}.
    """
    tsq_file = job_dir / "tool_selection_quality_output.json"
    if not tsq_file.exists():
        print(f"    Warning: No TSQ output found in {job_dir.name}")
        return None

    try:
        with open(tsq_file) as f:
            data = json.load(f)

        # Parse eval_output_items to get per-request scores
        eval_items = data.get("eval_output_items", [])
        if not eval_items:
            print(f"    Warning: No eval_output_items in TSQ file for {job_dir.name}")
            return None

        scores_by_example = {}
        for idx, item in enumerate(eval_items):
            # Extract example number from id like "banking_scenario_000"
            item_id = item.get("id", f"example_{idx}")
            score = item.get("score", 0.0)
            reasoning = item.get("reasoning", {})

            # Try to parse example number from id
            example_num = idx  # Default to index if can't parse
            if "_" in item_id:
                try:
                    # Handle formats like "banking_scenario_000"
                    num_str = item_id.split("_")[-1]
                    example_num = int(num_str)
                except (ValueError, IndexError):
                    pass

            scores_by_example[example_num] = {"id": item_id, "score": score, "reasoning": reasoning}

        return scores_by_example

    except (json.JSONDecodeError, KeyError) as e:
        print(f"    Warning: Error reading TSQ file in {job_dir.name}: {e}")
        return None


def calculate_per_request_throughput_metrics(csv_path: Path) -> tuple[dict[int, dict] | None, list[dict] | None]:
    """
    Calculate throughput metrics from standardized_data_all.csv on a per-request basis.

    Returns tuple of:
        1. dict mapping example_number to aggregated metrics dict with:
            - median_ttft_ms: Median Time To First Token (milliseconds)
            - median_itl_ms: Median Inter-Token Latency (milliseconds)
            - median_tps: Median tokens per second (per LLM call)
            - total_tokens: Total tokens generated for this request
            - num_llm_calls: Number of LLM calls for this request
            - total_duration_sec: Total duration for all LLM calls
        2. list of per-LLM-call dicts with:
            - example_number: Which request this LLM call belongs to
            - llm_call_idx: Index of this LLM call within the request
            - ttft_ms: Time To First Token for this specific call
            - tps: Tokens per second for this specific call
            - itl_ms: Median inter-token latency for this specific call
            - num_tokens: Number of tokens generated in this call
    """
    if not csv_path.exists():
        return None, None

    try:
        df = pd.read_csv(csv_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"    Warning: Error reading CSV {csv_path}: {e}")
        return None, None

    metrics_by_example = {}
    all_llm_call_data = []  # Per-LLM-call data for granular plotting

    # Group by example_number to process each request separately
    for example_num in df['example_number'].unique():
        example_df = df[df['example_number'] == example_num].copy()
        example_df = example_df.sort_values('event_timestamp')

        llm_calls = []
        all_itls = []
        current_start = None
        llm_call_idx = 0

        for _, row in example_df.iterrows():
            if row['event_type'] == 'LLM_START':
                current_start = row['event_timestamp']

            elif row['event_type'] == 'LLM_END' and current_start is not None:
                tokens = example_df[(example_df['event_type'] == 'LLM_NEW_TOKEN')
                                    & (example_df['event_timestamp'] > current_start) &
                                    (example_df['event_timestamp']
                                     <= row['event_timestamp'])].sort_values('event_timestamp')

                num_tokens = len(tokens)
                duration = row['event_timestamp'] - current_start

                if duration > 0 and num_tokens > 0:
                    tokens_per_sec = num_tokens / duration
                    token_times = tokens['event_timestamp'].values
                    ttft = token_times[0] - current_start

                    call_itls = []
                    if num_tokens > 1:
                        call_itls = np.diff(token_times).tolist()
                        all_itls.extend(call_itls)

                    llm_calls.append({
                        'tokens_per_sec': tokens_per_sec,
                        'ttft': ttft,
                        'num_tokens': num_tokens,
                        'duration': duration,
                        'start_time': current_start,
                        'end_time': row['event_timestamp'],
                    })

                    # Store per-LLM-call data for granular plotting
                    all_llm_call_data.append({
                        'example_number': int(example_num),
                        'llm_call_idx': llm_call_idx,
                        'ttft_ms': ttft * 1000,
                        'tps': tokens_per_sec,
                        'itl_ms': np.median(call_itls) * 1000 if call_itls else 0,
                        'num_tokens': num_tokens,
                        'duration_sec': duration,
                    })
                    llm_call_idx += 1

                current_start = None

        if not llm_calls:
            continue

        calls_df = pd.DataFrame(llm_calls)
        all_itls_array = np.array(all_itls) if all_itls else np.array([0])

        # Calculate aggregate metrics using MEDIAN for latency/throughput
        total_tokens = calls_df['num_tokens'].sum()
        total_duration = calls_df['duration'].sum()

        metrics_by_example[int(example_num)] = {
            'median_ttft_ms': calls_df['ttft'].median() * 1000,
            'median_itl_ms': np.median(all_itls_array) * 1000 if len(all_itls_array) > 0 else 0,
            'median_tps': calls_df['tokens_per_sec'].median(),
            'total_tokens': int(total_tokens),
            'num_llm_calls': len(calls_df),
            'total_duration_sec': total_duration,  # Also include p95 values for reference
            'p95_ttft_ms': calls_df['ttft'].quantile(0.95) * 1000,
            'p95_itl_ms': np.percentile(all_itls_array, 95) * 1000 if len(all_itls_array) > 0 else 0,
        }

    return metrics_by_example, all_llm_call_data


def collect_job_data_from_dir(jobs_dir: Path, experiment_label: str | None = None) -> tuple[list[dict], list[dict]]:
    """Collect per-request TSQ scores and throughput metrics from all job directories.

    Args:
        jobs_dir: Path to the jobs/ directory containing job subdirectories
        experiment_label: Label for this experiment (used in plots)

    Returns:
        Tuple of (per_request_data, per_llm_call_data):
            - per_request_data: List of dicts with aggregated metrics per request
            - per_llm_call_data: List of dicts with metrics for each individual LLM call
    """
    data = []
    llm_call_data = []

    job_dirs = sorted([d for d in jobs_dir.iterdir() if d.is_dir() and d.name.startswith('job_')])

    if not job_dirs:
        print(f"  No job directories found in {jobs_dir}")
        return data, llm_call_data

    print(f"  Found {len(job_dirs)} job directories")

    for job_dir in job_dirs:
        print(f"    Processing {job_dir.name}...")

        # Get per-request TSQ scores
        tsq_scores = extract_per_request_tsq_scores(job_dir)
        if tsq_scores is None:
            continue

        # Get per-request throughput metrics and per-LLM-call data
        csv_path = job_dir / "standardized_data_all.csv"
        throughput_metrics, job_llm_call_data = calculate_per_request_throughput_metrics(csv_path)
        if throughput_metrics is None:
            print(f"    Warning: No throughput data found in {job_dir.name}")
            continue

        # Match TSQ scores with throughput metrics by example_number
        matched_count = 0
        for example_num, tsq_data in tsq_scores.items():
            if example_num not in throughput_metrics:
                continue

            metrics = throughput_metrics[example_num]
            matched_count += 1

            row = {
                'job_name': job_dir.name,
                'job_label': get_job_label(job_dir.name),
                'experiment': experiment_label or jobs_dir.parent.name,
                'example_number': example_num,
                'sample_id': tsq_data['id'],
                'tsq_score': tsq_data['score'],
                **metrics
            }

            data.append(row)

        # Add per-LLM-call data with experiment/job metadata and TSQ scores
        if job_llm_call_data:
            for call_data in job_llm_call_data:
                example_num = call_data['example_number']
                if example_num in tsq_scores:
                    call_row = {
                        'job_name': job_dir.name,
                        'experiment': experiment_label or jobs_dir.parent.name,
                        'tsq_score': tsq_scores[example_num]['score'],
                        **call_data
                    }
                    llm_call_data.append(call_row)

        print(f"      Matched {matched_count} samples")

    return data, llm_call_data


def collect_job_data(input_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect per-request TSQ scores and throughput metrics from multiple input directories.

    Returns:
        Tuple of (per_request_df, per_llm_call_df):
            - per_request_df: DataFrame with aggregated metrics per request
            - per_llm_call_df: DataFrame with metrics for each individual LLM call
    """
    all_data = []
    all_llm_call_data = []

    for input_dir in input_dirs:
        experiment_label = get_experiment_label(input_dir)
        print(f"Collecting from: {input_dir} (label: {experiment_label})")

        # Check for jobs subdirectory
        jobs_dir = input_dir / "jobs"
        if not jobs_dir.exists():
            jobs_dir = input_dir

        if not jobs_dir.exists():
            print(f"Warning: Directory not found: {input_dir}")
            continue

        data, llm_call_data = collect_job_data_from_dir(jobs_dir, experiment_label)
        all_data.extend(data)
        all_llm_call_data.extend(llm_call_data)

    return pd.DataFrame(all_data), pd.DataFrame(all_llm_call_data)


def _add_job_stats_table(ax, job_stats: dict, job_labels: dict, job_colors: dict):
    """Add a table-style legend showing per-job statistics.

    Creates a formatted table with:
    - Row 1: Color squares and job IDs
    - Rows 2-7: Statistics (n, mean, median, σ, P10, P90)
    """
    jobs = list(job_stats.keys())
    n_jobs = len(jobs)

    if n_jobs == 0:
        return

    # Build table text with Unicode color blocks
    lines = []

    # Header row: job IDs with color indicator (using █ character)
    id_parts = []
    for job in jobs:
        id_parts.append(f"█ {job_labels[job]}")
    id_row = " │ ".join(id_parts)
    lines.append(id_row)
    lines.append("─" * len(id_row))

    # Stats rows
    stat_labels = ['n', 'mean', 'med', 'std', 'P10', 'P90']
    stat_keys = ['n', 'mean', 'median', 'std', 'p10', 'p90']

    for label, key in zip(stat_labels, stat_keys, strict=True):
        values = []
        for job in jobs:
            val = job_stats[job][key]
            if key == 'n':
                values.append(f"{int(val):>6}")
            elif abs(val) >= 100:
                values.append(f"{val:>6.1f}")
            else:
                values.append(f"{val:>6.2f}")
        row = " │ ".join(values)
        lines.append(f"{label:>3}: {row}")

    table_text = "\n".join(lines)

    # Position the text box
    ax.text(0.98,
            0.98,
            table_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            horizontalalignment='right',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))

    # We can't easily color individual characters, so add a legend strip above
    # using small colored patches via legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=job_colors[job], edgecolor='none', label=job_labels[job]) for job in jobs]
    ax.legend(handles=legend_handles,
              loc='upper right',
              bbox_to_anchor=(1.0, 1.15),
              ncol=n_jobs,
              fontsize=7,
              frameon=False,
              handlelength=1,
              handleheight=1,
              columnspacing=0.5)


def _add_job_stats_table_compact(ax, job_stats: dict, job_labels: dict, job_colors: dict):
    """Add a compact table-style legend for summary plots (smaller font)."""
    jobs = list(job_stats.keys())
    n_jobs = len(jobs)

    if n_jobs == 0:
        return

    # Build compact table text
    lines = []

    # Header row: job IDs
    id_parts = [f"{job_labels[job]}" for job in jobs]
    id_row = " │ ".join(id_parts)
    lines.append(id_row)
    lines.append("─" * len(id_row))

    # Stats rows (compact but with all stats including P10/P90)
    stat_labels = ['n', 'mean', 'med', 'std', 'P10', 'P90']
    stat_keys = ['n', 'mean', 'median', 'std', 'p10', 'p90']

    for label, key in zip(stat_labels, stat_keys, strict=True):
        values = []
        for job in jobs:
            val = job_stats[job][key]
            if key == 'n':
                values.append(f"{int(val):>5}")
            elif abs(val) >= 100:
                values.append(f"{val:>5.0f}")
            else:
                values.append(f"{val:>5.1f}")
        row = " │ ".join(values)
        lines.append(f"{label:>3}: {row}")

    table_text = "\n".join(lines)

    ax.text(0.98,
            0.98,
            table_text,
            transform=ax.transAxes,
            fontsize=6,
            verticalalignment='top',
            horizontalalignment='right',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='gray'))

    # Add colored legend strip
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=job_colors[job], edgecolor='none', label=job_labels[job]) for job in jobs]
    ax.legend(handles=legend_handles,
              loc='upper right',
              bbox_to_anchor=(1.0, 1.12),
              ncol=n_jobs,
              fontsize=5,
              frameon=False,
              handlelength=0.8,
              handleheight=0.8,
              columnspacing=0.3)


def create_histogram_plots(df: pd.DataFrame, output_dir: Path, llm_call_df: pd.DataFrame | None = None):
    """Create histogram plots of throughput metrics distribution.

    Uses per-metric bin counts:
    - 100 bins for TTFT, ITL, Throughput (per-LLM-call)
    - 50 bins for Total Tokens
    - 25 bins for LLM Calls
    - 25 bins for Total Duration

    Each job is plotted as a separate histogram with its own color.

    Args:
        df: DataFrame with throughput and TSQ data (per-request aggregates)
        output_dir: Directory to save plots
        llm_call_df: Optional DataFrame with per-LLM-call metrics for granular plotting
    """

    if df.empty:
        print("No data to plot!")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by job_name for separate histograms per job
    jobs = df['job_name'].unique() if 'job_name' in df.columns else ['default']
    multi_job = len(jobs) > 1

    # Create color map for jobs
    colors = cm.tab10(np.linspace(0, 1, min(len(jobs), 10)))
    job_colors = {job: colors[i % 10] for i, job in enumerate(jobs)}

    # Create short labels for legend (first 8 chars of job UUID)
    job_labels = {job: job.replace('job_', '')[:8] for job in jobs}

    # Define plots to create: (metric_column, x_label, filename, data_source, num_bins)
    # Top row (per-LLM-call): 100 bins
    # Bottom row (per-request): 50 for tokens, 25 for calls, 25 for duration
    plots = [
        ('ttft_ms', 'Time To First Token (ms)', 'ttft_histogram.png', 'llm_call', 100),
        ('itl_ms', 'Inter-Token Latency (ms)', 'itl_histogram.png', 'llm_call', 100),
        ('tps', 'Throughput (tok/s)', 'tps_histogram.png', 'llm_call', 100),
        ('total_tokens', 'Total Tokens Generated', 'total_tokens_histogram.png', 'request', 50),
        ('num_llm_calls', 'Number of LLM Calls', 'llm_calls_histogram.png', 'request', 25),
        ('total_duration_sec', 'Total Duration (s)', 'total_duration_histogram.png', 'request', 25),
    ]

    for metric_col, x_label, filename, data_source, num_bins in plots:
        # Select appropriate data source
        if data_source == 'llm_call' and llm_call_df is not None and not llm_call_df.empty:
            data_df = llm_call_df
        else:
            data_df = df
            # Use median values for per-request data when llm_call data not available
            if data_source == 'llm_call':
                metric_col = f'median_{metric_col}'

        if metric_col not in data_df.columns:
            continue

        _, ax = plt.subplots(figsize=(10, 7))

        metric_data = data_df[metric_col].dropna()

        if len(metric_data) == 0:
            plt.close()
            continue

        # For TTFT metrics, use fixed bin width to ensure good resolution in visible range
        if metric_col in ('ttft_ms', 'median_ttft_ms'):
            max_val = metric_data.max()
            bins_to_use = np.arange(0, max_val + TTFT_BIN_WIDTH_MS, TTFT_BIN_WIDTH_MS)
        else:
            bins_to_use = num_bins

        # Collect per-job statistics for the legend table
        job_stats = {}

        if multi_job:
            # Overlay histograms for each job
            for job in jobs:
                job_df = data_df[data_df['job_name'] == job]
                job_data = job_df[metric_col].dropna()
                if len(job_data) > 0:
                    ax.hist(job_data, bins=bins_to_use, alpha=0.5, color=job_colors[job], edgecolor='none')
                    # Add median line for this job (dotted, same color)
                    median_j = job_data.median()
                    ax.axvline(x=median_j, color=job_colors[job], linestyle=':', linewidth=2, alpha=0.9)
                    # Store stats for legend table
                    job_stats[job] = {
                        'n': len(job_data),
                        'mean': job_data.mean(),
                        'median': median_j,
                        'std': job_data.std(),
                        'p10': job_data.quantile(0.10),
                        'p90': job_data.quantile(0.90),
                    }
        else:
            ax.hist(metric_data, bins=bins_to_use, alpha=0.7, color='steelblue', edgecolor='darkblue', linewidth=0.5)
            # Add median line for single job
            median_val = metric_data.median()
            ax.axvline(x=median_val, color='steelblue', linestyle=':', linewidth=2, alpha=0.9)
            job_stats['default'] = {
                'n': len(metric_data),
                'mean': metric_data.mean(),
                'median': median_val,
                'std': metric_data.std(),
                'p10': metric_data.quantile(0.10),
                'p90': metric_data.quantile(0.90),
            }

        # Build table-style legend text
        if multi_job and job_stats:
            _add_job_stats_table(ax, job_stats, job_labels, job_colors)

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        data_type = 'Per-LLM-Call' if data_source == 'llm_call' else 'Per-Request'
        if metric_col in ('ttft_ms', 'median_ttft_ms'):
            bins_info = f'bin_width={TTFT_BIN_WIDTH_MS}ms, xlim={MAX_TTFT_MS}ms'
        else:
            bins_info = f'bins={num_bins}'
        ax.set_title(f'{x_label} Distribution\n({data_type}, n={len(metric_data)}, {bins_info})', fontsize=14)

        ax.grid(True, alpha=0.3, axis='y')

        # Apply x-axis limit for TTFT metrics
        if metric_col in ('ttft_ms', 'median_ttft_ms'):
            ax.set_xlim(0, MAX_TTFT_MS)

        # Add stats box for single job only (multi-job uses table legend)
        if not multi_job and job_stats:
            stats = list(job_stats.values())[0]
            stats_text = (f'n={stats["n"]}\n'
                          f'mean={stats["mean"]:.2f}\n'
                          f'median={stats["median"]:.2f}\n'
                          f'std={stats["std"]:.2f}\n'
                          f'P10={stats["p10"]:.2f}\n'
                          f'P90={stats["p90"]:.2f}')
            ax.text(0.98,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='right',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))

        plt.tight_layout()

        output_path = output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")

    # Create a combined summary plot
    create_summary_histogram_plot(df, output_dir, llm_call_df)

    # Save the collected data as CSV
    csv_path = output_dir / 'throughput_histogram_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved data: {csv_path}")

    # Save per-LLM-call data if available
    if llm_call_df is not None and not llm_call_df.empty:
        llm_call_csv_path = output_dir / 'throughput_histogram_per_llm_call_data.csv'
        llm_call_df.to_csv(llm_call_csv_path, index=False)
        print(f"  Saved data: {llm_call_csv_path}")


def create_summary_histogram_plot(df: pd.DataFrame, output_dir: Path, llm_call_df: pd.DataFrame | None = None):
    """Create a multi-panel summary histogram plot.

    Top row shows per-LLM-call metrics (TTFT, ITL, Throughput) if llm_call_df is provided,
    otherwise falls back to per-request medians.
    Bottom row shows per-request aggregate metrics (Total Tokens, LLM Calls, Duration).

    Each job is plotted as a separate histogram with its own color.

    Args:
        df: DataFrame with per-request throughput and TSQ data
        output_dir: Directory to save plots
        llm_call_df: Optional DataFrame with per-LLM-call metrics for granular plotting
    """

    _, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Group by job_name for separate histograms per job
    jobs = df['job_name'].unique() if 'job_name' in df.columns else ['default']
    multi_job = len(jobs) > 1

    # Create color map for jobs
    colors = cm.tab10(np.linspace(0, 1, min(len(jobs), 10)))
    job_colors = {job: colors[i % 10] for i, job in enumerate(jobs)}

    # Create short labels for legend (first 8 chars of job UUID)
    job_labels = {job: job.replace('job_', '')[:8] for job in jobs}

    # Check if we have per-LLM-call data for granular plotting
    has_llm_call_data = llm_call_df is not None and not llm_call_df.empty

    # Top row: Per-LLM-call metrics (or per-request medians if no granular data)
    # Bottom row: Per-request aggregate metrics
    # Include per-metric bin counts: 100 for top row, 50/25/25 for bottom row
    if has_llm_call_data:
        # Top row uses per-LLM-call data
        top_row_metrics = [
            ('ttft_ms', 'TTFT per LLM Call (ms)', llm_call_df, 100),
            ('itl_ms', 'ITL per LLM Call (ms)', llm_call_df, 100),
            ('tps', 'Throughput per LLM Call (tok/s)', llm_call_df, 100),
        ]
    else:
        # Fall back to per-request medians
        top_row_metrics = [
            ('median_ttft_ms', 'Median TTFT (ms)', df, 100),
            ('median_itl_ms', 'Median ITL (ms)', df, 100),
            ('median_tps', 'Median Throughput (tok/s)', df, 100),
        ]

    # Bottom row always uses per-request data
    bottom_row_metrics = [
        ('total_tokens', 'Total Tokens', df, 50),
        ('num_llm_calls', 'LLM Calls', df, 25),
        ('total_duration_sec', 'Total Duration (s)', df, 25),
    ]

    all_metrics = top_row_metrics + bottom_row_metrics

    for ax, (metric_col, label, data_df, num_bins) in zip(axes, all_metrics, strict=True):
        if metric_col not in data_df.columns:
            ax.set_visible(False)
            continue

        metric_data = data_df[metric_col].dropna()

        if len(metric_data) == 0:
            ax.set_visible(False)
            continue

        # For TTFT metrics, use fixed bin width to ensure good resolution in visible range
        if metric_col in ('ttft_ms', 'median_ttft_ms'):
            max_val = metric_data.max()
            bins_to_use = np.arange(0, max_val + TTFT_BIN_WIDTH_MS, TTFT_BIN_WIDTH_MS)
        else:
            bins_to_use = num_bins

        # Collect per-job statistics
        job_stats_summary = {}

        if multi_job:
            # Overlay histograms for each job
            for job in jobs:
                job_df = data_df[data_df['job_name'] == job]
                job_data = job_df[metric_col].dropna()
                if len(job_data) > 0:
                    ax.hist(job_data, bins=bins_to_use, alpha=0.5, color=job_colors[job], edgecolor='none')
                    # Add median line for this job (dotted, same color)
                    median_j = job_data.median()
                    ax.axvline(x=median_j, color=job_colors[job], linestyle=':', linewidth=1.5, alpha=0.9)
                    # Store stats
                    job_stats_summary[job] = {
                        'n': len(job_data),
                        'mean': job_data.mean(),
                        'median': median_j,
                        'std': job_data.std(),
                        'p10': job_data.quantile(0.10),
                        'p90': job_data.quantile(0.90),
                    }
        else:
            ax.hist(metric_data, bins=bins_to_use, alpha=0.7, color='steelblue', edgecolor='none')
            median_val = metric_data.median()
            ax.axvline(x=median_val, color='steelblue', linestyle=':', linewidth=1.5, alpha=0.9)

        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Apply x-axis limit for TTFT metrics
        if metric_col in ('ttft_ms', 'median_ttft_ms'):
            ax.set_xlim(0, MAX_TTFT_MS)

        # Add table-style legend for multi-job, or simple stats box for single job
        if multi_job and job_stats_summary:
            _add_job_stats_table_compact(ax, job_stats_summary, job_labels, job_colors)
        elif not multi_job:
            median_val = metric_data.median()
            mean_val = metric_data.mean()
            std_val = metric_data.std()
            p10_val = metric_data.quantile(0.10)
            p90_val = metric_data.quantile(0.90)
            if abs(median_val) >= 100:
                stats_text = (f'n={len(metric_data)}\n'
                              f'mean={mean_val:.1f}\n'
                              f'med={median_val:.1f}\n'
                              f'std={std_val:.1f}\n'
                              f'P10={p10_val:.1f}\n'
                              f'P90={p90_val:.1f}')
            else:
                stats_text = (f'n={len(metric_data)}\n'
                              f'mean={mean_val:.2f}\n'
                              f'med={median_val:.2f}\n'
                              f'std={std_val:.2f}\n'
                              f'P10={p10_val:.2f}\n'
                              f'P90={p90_val:.2f}')

            ax.text(0.98,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    horizontalalignment='right',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))

        ax.set_title(label, fontsize=10, fontweight='bold')

    n_requests = len(df)
    n_llm_calls = len(llm_call_df) if has_llm_call_data else 'N/A'
    n_jobs = len(jobs)
    plt.suptitle(
        f'Throughput Metrics Distribution ({n_jobs} job{"s" if n_jobs > 1 else ""})\n'
        f'Top row: Per-LLM-Call ({n_llm_calls} calls, 100 bins), '
        f'Bottom row: Per-Request ({n_requests} requests, 50/25/25 bins)',
        fontsize=14,
        y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'summary_throughput_histograms.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Histogram plotting for throughput metrics distribution analysis.',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
Examples:
  # Analyze jobs from an experiment
  python plot_throughput_histograms_per_request.py ./outputs/dynamo_evals/experiment1/jobs/

  # Custom output directory
  python plot_throughput_histograms_per_request.py ./outputs/exp1 --output ./comparison

Features:
  - Per-LLM-call histograms (top row): TTFT, ITL, Throughput (100 bins each)
  - Per-request aggregates (bottom row): Total Tokens (50 bins), LLM Calls (25 bins), Duration (25 bins)
  - Each job plotted as a separate histogram with its own color
  - Legend with per-job statistics (n, mean, median, std, P10, P90)
  - Statistical annotations: median lines (dotted, per-job color)
""")
    parser.add_argument('directories',
                        type=str,
                        nargs='+',
                        help='Path(s) to directories containing jobs/ subdirectories')
    parser.add_argument('--output',
                        '-o',
                        type=str,
                        default=None,
                        help='Output directory for plots (default: auto-determined based on input)')

    args = parser.parse_args()

    # Parse input directories
    input_dirs = []
    for dir_path in args.directories:
        path = Path(dir_path)
        if not path.exists():
            print(f"Warning: Directory not found: {path}")
            continue
        input_dirs.append(path)

    if not input_dirs:
        print("Error: No valid input directories found!")
        sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    elif len(input_dirs) == 1:
        jobs_subdir = input_dirs[0] / "jobs"
        if jobs_subdir.exists():
            output_dir = jobs_subdir / 'throughput_histogram_plots'
        else:
            output_dir = input_dirs[0] / 'throughput_histogram_plots'
    else:
        output_dir = Path('./throughput_histogram_plots')

    print(f"Input directories: {len(input_dirs)}")
    for d in input_dirs:
        print(f"  - {d}")
    print(f"Output directory: {output_dir}")
    print("Histogram bins: TTFT/ITL/TPS=100, Tokens=50, Calls=25, Duration=25")
    print()

    # Collect all job data (per-request and per-LLM-call)
    df, llm_call_df = collect_job_data(input_dirs)

    if df.empty:
        print("\nNo valid job data found!")
        sys.exit(1)

    # Print summary
    jobs = df['job_name'].unique() if 'job_name' in df.columns else ['default']
    print(f"\nCollected {len(df)} per-request data points across {len(jobs)} job(s)")
    if not llm_call_df.empty:
        print(f"Collected {len(llm_call_df)} per-LLM-call data points")

    if len(jobs) > 1:
        print("\nSamples per job:")
        for job in jobs:
            job_df = df[df['job_name'] == job]
            count = len(job_df)
            short_label = job.replace('job_', '')[:8]
            print(f"  - {short_label}...: {count} samples")
    print()

    # Create plots
    print("Creating histogram plots...")
    create_histogram_plots(df, output_dir, llm_call_df=llm_call_df if not llm_call_df.empty else None)

    print(f"\nDone! Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
