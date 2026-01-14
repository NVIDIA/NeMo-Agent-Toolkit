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
Unified plotting script for throughput metrics vs TSQ (Tool Selection Quality) scores.

This script combines per-request and per-LLM-call analysis with multi-experiment comparison:

Features:
- Per-LLM-call scatter plots (TTFT, ITL, Throughput) showing every individual LLM call
- Per-request aggregate metrics (Total Tokens, LLM Calls, Duration)
- Statistical annotations: median lines, +/-2 std bounds, correlation, mean, std
- Lines of best fit for aggregate metrics
- Multi-experiment comparison with color coding
- Optimizer trial parameter matching (e.g., color by temperature)

Usage:
    # Single experiment
    python plot_throughput_vs_tsq_per_request.py ./outputs/dynamo_evals/experiment1

    # Compare multiple experiments
    python plot_throughput_vs_tsq_per_request.py ./outputs/exp1 ./outputs/exp2 ./outputs/exp3

    # Color by optimizer hyperparameter
    python plot_throughput_vs_tsq_per_request.py ./outputs/exp1 --color-by temperature

    # Custom output directory
    python plot_throughput_vs_tsq_per_request.py ./outputs/exp1 ./outputs/exp2 --output ./comparison

Example:
    python plot_throughput_vs_tsq_per_request.py ./outputs/dynamo_evals/unified_default \
        ./outputs/dynamo_evals/unified_thompson
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy import stats


def get_job_label(job_dir_name: str) -> str:
    """Extract short label from job directory name (first 7 chars)."""
    return job_dir_name[:7]


def get_experiment_label(dir_path: Path) -> str:
    """Extract a short label from the experiment directory name."""
    return dir_path.name


def load_optimizer_trials(experiment_dir: Path) -> pd.DataFrame | None:
    """Load optimizer trial parameters if available.

    Looks for optimizer_results/trials_dataframe_params.csv in the experiment directory.

    Returns:
        DataFrame with columns: trial_number, value (TSQ score), and parameter columns.
        Returns None if no optimizer results found.
    """
    trials_file = experiment_dir / "optimizer_results" / "trials_dataframe_params.csv"
    if not trials_file.exists():
        return None

    try:
        df = pd.read_csv(trials_file)
        # Rename 'number' to 'trial_number' for clarity
        df = df.rename(columns={'number': 'trial_number'})

        # Extract parameter columns (those starting with 'params_')
        param_cols = [c for c in df.columns if c.startswith('params_')]

        # Create simplified param names (e.g., 'params_llms.dynamo_llm.temperature' -> 'temperature')
        rename_map = {}
        for col in param_cols:
            # Extract the last part after the last '.'
            simple_name = col.split('.')[-1]
            rename_map[col] = simple_name

        df = df.rename(columns=rename_map)

        # Keep trial_number, value, and simplified param columns
        keep_cols = ['trial_number', 'value'] + list(rename_map.values())
        df = df[[c for c in keep_cols if c in df.columns]]

        param_names = list(rename_map.values())
        print(f"    Loaded optimizer trials: {len(df)} trials with params: {param_names}")
        return df

    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, KeyError) as e:
        print(f"    Warning: Error loading optimizer trials: {e}")
        return None


def match_job_to_trial(job_avg_score: float, trials_df: pd.DataFrame, tolerance: float = 1e-6) -> dict | None:
    """Match a job's average TSQ score to a trial's value column.

    Args:
        job_avg_score: The job's average TSQ score from tool_selection_quality_output.json
        trials_df: DataFrame from load_optimizer_trials()
        tolerance: Floating-point comparison tolerance

    Returns:
        Dict with trial_number and parameter values if exactly one match found.
        None if no match or multiple trials have the same score (ambiguous).
    """
    if trials_df is None or job_avg_score is None:
        return None

    # Find trials with matching value (within floating-point tolerance)
    matches = trials_df[abs(trials_df['value'] - job_avg_score) < tolerance]

    if len(matches) == 1:
        # Unique match - return trial info
        return matches.iloc[0].to_dict()
    elif len(matches) > 1:
        # Ambiguous - multiple trials have same score, skip matching
        print(f"      Warning: {len(matches)} trials have identical TSQ score "
              f"{job_avg_score:.10f}, skipping trial assignment")
        return None
    else:
        # No match found
        return None


def get_job_average_tsq(job_dir: Path) -> float | None:
    """Get the average TSQ score from a job's tool_selection_quality_output.json."""
    tsq_file = job_dir / "tool_selection_quality_output.json"
    if not tsq_file.exists():
        return None

    try:
        with open(tsq_file) as f:
            data = json.load(f)
        return data.get("average_score")
    except (json.JSONDecodeError, KeyError):
        return None


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


def collect_job_data_from_dir(jobs_dir: Path,
                              experiment_label: str | None = None,
                              trials_df: pd.DataFrame | None = None) -> tuple[list[dict], list[dict]]:
    """Collect per-request TSQ scores and throughput metrics from all job directories.

    Args:
        jobs_dir: Path to the jobs/ directory containing job subdirectories
        experiment_label: Label for this experiment (used in plots)
        trials_df: Optional DataFrame from load_optimizer_trials() for matching
                   jobs to optimizer trial parameters

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

        # Try to match this job to an optimizer trial
        trial_params = None
        if trials_df is not None:
            job_avg_score = get_job_average_tsq(job_dir)
            trial_params = match_job_to_trial(job_avg_score, trials_df)
            if trial_params:
                trial_num = trial_params.get('trial_number')
                # Get param values for display (exclude trial_number and value)
                param_display = {k: v for k, v in trial_params.items() if k not in ['trial_number', 'value']}
                print(f"      Matched to trial {trial_num}: {param_display}")

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

            # Add trial parameters if matched
            if trial_params:
                row['trial_number'] = trial_params.get('trial_number')
                # Add all parameter columns (exclude 'value' which is the TSQ score)
                for key, val in trial_params.items():
                    if key not in ['trial_number', 'value']:
                        row[key] = val

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

    If optimizer_results/trials_dataframe_params.csv exists in an experiment directory,
    jobs will be matched to their corresponding optimizer trial parameters.

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

        # Try to load optimizer trials for this experiment
        trials_df = load_optimizer_trials(input_dir)

        # Check for jobs subdirectory
        jobs_dir = input_dir / "jobs"
        if not jobs_dir.exists():
            jobs_dir = input_dir

        if not jobs_dir.exists():
            print(f"Warning: Directory not found: {input_dir}")
            continue

        data, llm_call_data = collect_job_data_from_dir(jobs_dir, experiment_label, trials_df)
        all_data.extend(data)
        all_llm_call_data.extend(llm_call_data)

    return pd.DataFrame(all_data), pd.DataFrame(all_llm_call_data)


def create_scatter_plots(df: pd.DataFrame,
                         output_dir: Path,
                         color_by: str | None = None,
                         llm_call_df: pd.DataFrame | None = None):
    """Create scatter plots of per-request throughput metrics vs TSQ score.

    Args:
        df: DataFrame with throughput and TSQ data (per-request aggregates)
        output_dir: Directory to save plots
        color_by: Optional column name to use for coloring points (e.g., 'temperature')
        llm_call_df: Optional DataFrame with per-LLM-call metrics for granular plotting
    """

    if df.empty:
        print("No data to plot!")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine coloring strategy
    use_color_by = color_by and color_by in df.columns
    if color_by and not use_color_by:
        print(f"  Warning: --color-by column '{color_by}' not found in data. "
              f"Available columns: {list(df.columns)}")

    # Check if we have multiple experiments (fallback if no color_by)
    experiments = df['experiment'].unique() if 'experiment' in df.columns else ['default']
    multi_experiment = len(experiments) > 1 and not use_color_by

    # Create color map
    if use_color_by:
        # Color by the specified column (e.g., temperature)
        unique_values = sorted(df[color_by].dropna().unique())
        colors = cm.viridis(np.linspace(0, 1, len(unique_values)))
        value_colors = {val: colors[i] for i, val in enumerate(unique_values)}
        print(f"  Coloring by '{color_by}': {unique_values}")
    elif multi_experiment:
        colors = cm.tab10(np.linspace(0, 1, len(experiments)))
        exp_colors = {exp: colors[i] for i, exp in enumerate(experiments)}

    # Define plots to create: (metric_column, x_label, filename)
    # Using MEDIAN values for latency/throughput metrics
    plots = [
        ('median_ttft_ms', 'Median Time To First Token (ms)', 'ttft_vs_tsq.png'),
        ('median_itl_ms', 'Median Inter-Token Latency (ms)', 'itl_vs_tsq.png'),
        ('median_tps', 'Median Per-Request Throughput (tok/s)', 'tps_vs_tsq.png'),
        ('total_tokens', 'Total Tokens Generated', 'total_tokens_vs_tsq.png'),
        ('num_llm_calls', 'Number of LLM Calls', 'llm_calls_vs_tsq.png'),
    ]

    for metric_col, x_label, filename in plots:
        if metric_col not in df.columns:
            continue

        _, ax = plt.subplots(figsize=(10, 7))

        if use_color_by:
            # Color by the specified column (e.g., temperature)
            for val in unique_values:
                val_df = df[df[color_by] == val]
                ax.scatter(val_df[metric_col],
                           val_df['tsq_score'],
                           s=50,
                           alpha=0.6,
                           c=[value_colors[val]],
                           edgecolors='darkgray',
                           linewidths=0.5,
                           label=f'{color_by}={val}')
            ax.legend(title=color_by.capitalize(), loc='best')
        elif multi_experiment:
            for exp in experiments:
                exp_df = df[df['experiment'] == exp]
                ax.scatter(exp_df[metric_col],
                           exp_df['tsq_score'],
                           s=50,
                           alpha=0.6,
                           c=[exp_colors[exp]],
                           edgecolors='darkgray',
                           linewidths=0.5,
                           label=exp)
            ax.legend(title='Experiment', loc='best')
        else:
            ax.scatter(df[metric_col],
                       df['tsq_score'],
                       s=50,
                       alpha=0.6,
                       c='steelblue',
                       edgecolors='darkblue',
                       linewidths=0.5)

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('TSQ Score (per request)', fontsize=12)
        title_suffix = f' (colored by {color_by})' if use_color_by else ''
        ax.set_title(f'{x_label} vs Tool Selection Quality\n(Per-Request Analysis{title_suffix})', fontsize=14)

        ax.grid(True, alpha=0.3)

        # Add correlation info
        if len(df) > 2:
            corr = df[metric_col].corr(df['tsq_score'])
            ax.text(0.02,
                    0.98,
                    f'Correlation: {corr:.3f}\nN={len(df)} samples',
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        output_path = output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")

    # Create a combined summary plot
    create_summary_plot(df, output_dir, color_by, llm_call_df)

    # Save the collected data as CSV
    csv_path = output_dir / 'throughput_vs_tsq_per_request_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved data: {csv_path}")

    # Save per-LLM-call data if available
    if llm_call_df is not None and not llm_call_df.empty:
        llm_call_csv_path = output_dir / 'throughput_vs_tsq_per_llm_call_data.csv'
        llm_call_df.to_csv(llm_call_csv_path, index=False)
        print(f"  Saved data: {llm_call_csv_path}")


def create_summary_plot(df: pd.DataFrame,
                        output_dir: Path,
                        color_by: str | None = None,
                        llm_call_df: pd.DataFrame | None = None):
    """Create a multi-panel summary plot.

    Top row shows per-LLM-call metrics (TTFT, ITL, Throughput) if llm_call_df is provided,
    otherwise falls back to per-request medians.
    Bottom row shows per-request aggregate metrics (Total Tokens, LLM Calls, Duration).

    Args:
        df: DataFrame with per-request throughput and TSQ data
        output_dir: Directory to save plots
        color_by: Optional column name to use for coloring points (e.g., 'temperature')
        llm_call_df: Optional DataFrame with per-LLM-call metrics for granular plotting
    """

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Determine coloring strategy
    use_color_by = color_by and color_by in df.columns

    experiments = df['experiment'].unique() if 'experiment' in df.columns else ['default']
    multi_experiment = len(experiments) > 1 and not use_color_by

    # Create color map
    if use_color_by:
        unique_values = sorted(df[color_by].dropna().unique())
        colors = cm.viridis(np.linspace(0, 1, len(unique_values)))
        value_colors = {val: colors[i] for i, val in enumerate(unique_values)}
    elif multi_experiment:
        colors = cm.tab10(np.linspace(0, 1, len(experiments)))
        exp_colors = {exp: colors[i] for i, exp in enumerate(experiments)}

    # Check if we have per-LLM-call data for granular plotting
    has_llm_call_data = llm_call_df is not None and not llm_call_df.empty

    # Top row: Per-LLM-call metrics (or per-request medians if no granular data)
    # Bottom row: Per-request aggregate metrics
    if has_llm_call_data:
        # Top row uses per-LLM-call data
        top_row_metrics = [
            ('ttft_ms', 'TTFT per LLM Call (ms)', llm_call_df),
            ('itl_ms', 'ITL per LLM Call (ms)', llm_call_df),
            ('tps', 'Throughput per LLM Call (tok/s)', llm_call_df),
        ]
    else:
        # Fall back to per-request medians
        top_row_metrics = [
            ('median_ttft_ms', 'Median TTFT (ms)', df),
            ('median_itl_ms', 'Median ITL (ms)', df),
            ('median_tps', 'Median Throughput (tok/s)', df),
        ]

    # Bottom row always uses per-request data
    bottom_row_metrics = [
        ('total_tokens', 'Total Tokens', df),
        ('num_llm_calls', 'LLM Calls', df),
        ('total_duration_sec', 'Total Duration (s)', df),
    ]

    all_metrics = top_row_metrics + bottom_row_metrics
    bottom_row_cols = [m[0] for m in bottom_row_metrics]  # Track which are bottom row

    for ax, (metric_col, label, data_df) in zip(axes, all_metrics, strict=True):
        if metric_col not in data_df.columns:
            ax.set_visible(False)
            continue

        # Determine if this is per-LLM-call data (different sample size)
        is_llm_call_data = data_df is llm_call_df if has_llm_call_data else False
        is_bottom_row = metric_col in bottom_row_cols

        if use_color_by and color_by in data_df.columns:
            for val in unique_values:
                val_df = data_df[data_df[color_by] == val]
                ax.scatter(val_df[metric_col],
                           val_df['tsq_score'],
                           s=15 if is_llm_call_data else 30,
                           alpha=0.3 if is_llm_call_data else 0.5,
                           c=[value_colors[val]],
                           edgecolors='none' if is_llm_call_data else 'darkgray',
                           linewidths=0.3,
                           label=f'{val}')
        elif multi_experiment:
            for exp in experiments:
                exp_df = data_df[data_df['experiment'] == exp]
                ax.scatter(exp_df[metric_col],
                           exp_df['tsq_score'],
                           s=15 if is_llm_call_data else 30,
                           alpha=0.3 if is_llm_call_data else 0.5,
                           c=[exp_colors[exp]],
                           edgecolors='none' if is_llm_call_data else 'darkgray',
                           linewidths=0.3,
                           label=exp)
        else:
            ax.scatter(data_df[metric_col],
                       data_df['tsq_score'],
                       s=15 if is_llm_call_data else 30,
                       alpha=0.3 if is_llm_call_data else 0.5,
                       c='steelblue',
                       edgecolors='none' if is_llm_call_data else 'darkblue',
                       linewidths=0.3)

        # Add median and ±2 std dev vertical lines
        metric_data = data_df[metric_col].dropna()
        if len(metric_data) > 0:
            median_val = metric_data.median()
            std_val = metric_data.std()
            low_2std = median_val - 2 * std_val
            high_2std = median_val + 2 * std_val

            # Draw vertical lines
            ax.axvline(x=median_val, color='red', linestyle='-', linewidth=2, alpha=0.8, zorder=5)
            ax.axvline(x=low_2std, color='red', linestyle='--', linewidth=1.5, alpha=0.6, zorder=5)
            ax.axvline(x=high_2std, color='red', linestyle='--', linewidth=1.5, alpha=0.6, zorder=5)

        # Add line of best fit for bottom row (per-request aggregate metrics)
        if is_bottom_row and len(data_df) > 2:
            x_data = data_df[metric_col].values
            y_data = data_df['tsq_score'].values
            mask = ~(np.isnan(x_data) | np.isnan(y_data))
            if mask.sum() >= 2:
                slope, intercept, _r_value, _p_value, _std_err = stats.linregress(x_data[mask], y_data[mask])
                x_line = np.linspace(x_data[mask].min(), x_data[mask].max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color='darkgreen', linestyle='-', linewidth=2, alpha=0.8, zorder=4)

        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('TSQ Score', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Build statistics text box with numerical labels
        if len(data_df) > 2:
            metric_data = data_df[metric_col].dropna()
            corr = data_df[metric_col].corr(data_df['tsq_score'])
            n_samples = len(data_df)
            median_val = metric_data.median()
            std_val = metric_data.std()
            mean_val = metric_data.mean()

            # For bottom row, also compute R² from line of best fit
            if is_bottom_row:
                x_data = data_df[metric_col].values
                y_data = data_df['tsq_score'].values
                mask = ~(np.isnan(x_data) | np.isnan(y_data))
                if mask.sum() >= 2:
                    slope, intercept, r_value, _, _ = stats.linregress(x_data[mask], y_data[mask])
                    r_squared = r_value**2
                else:
                    r_squared = 0
                    slope = 0

            # Format numbers appropriately based on magnitude
            if abs(median_val) >= 100:
                stats_text = (f'n={n_samples}\n'
                              f'r={corr:.2f}\n'
                              f'mean={mean_val:.1f}\n'
                              f'med={median_val:.1f}\n'
                              f'std={std_val:.1f}')
            else:
                stats_text = (f'n={n_samples}\n'
                              f'r={corr:.2f}\n'
                              f'mean={mean_val:.2f}\n'
                              f'med={median_val:.2f}\n'
                              f'std={std_val:.2f}')

            # Add R² and slope for bottom row (line of best fit info)
            if is_bottom_row:
                if abs(slope) < 0.001:
                    stats_text += f'\nR²={r_squared:.3f}\nslope={slope:.2e}'
                else:
                    stats_text += f'\nR²={r_squared:.3f}\nslope={slope:.4f}'

            ax.text(0.02,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))

            ax.set_title(label, fontsize=10, fontweight='bold')

    # Add legend
    if use_color_by or multi_experiment:
        handles, labels = axes[0].get_legend_handles_labels()
        legend_title = color_by.capitalize() if use_color_by else 'Experiment'
        fig.legend(handles, labels, loc='upper right', title=legend_title, bbox_to_anchor=(0.99, 0.99))

    title_suffix = f' (by {color_by})' if use_color_by else ''
    n_requests = len(df)
    n_llm_calls = len(llm_call_df) if has_llm_call_data else 'N/A'
    plt.suptitle(
        f'Throughput Metrics vs TSQ{title_suffix}\n'
        f'Top row: Per-LLM-Call ({n_llm_calls} calls), '
        f'Bottom row: Per-Request ({n_requests} requests)',
        fontsize=14,
        y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'summary_throughput_vs_tsq.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified plotting for throughput metrics vs TSQ scores with multi-experiment comparison.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single experiment analysis
  python plot_throughput_vs_tsq_per_request.py ./outputs/dynamo_evals/experiment1

  # Compare multiple experiments (replaces plot_comparison.py)
  python plot_throughput_vs_tsq_per_request.py ./outputs/unified_default ./outputs/unified_thompson

  # Compare 3+ experiments
  python plot_throughput_vs_tsq_per_request.py ./outputs/exp1 ./outputs/exp2 ./outputs/exp3

  # Custom output directory
  python plot_throughput_vs_tsq_per_request.py ./outputs/exp1 ./outputs/exp2 --output ./comparison

  # Color by optimizer hyperparameter (e.g., temperature sweep)
  python plot_throughput_vs_tsq_per_request.py ./outputs/exp1 --color-by temperature

Features:
  - Per-LLM-call scatter plots (top row): TTFT, ITL, Throughput for every individual LLM call
  - Per-request aggregates (bottom row): Total Tokens, LLM Calls, Duration with line of best fit
  - Statistical annotations: median (red solid), +/-2 std (red dashed), correlation, mean, std
  - Multi-experiment comparison with automatic color coding
  - Optimizer trial parameter matching and coloring
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
    parser.add_argument('--color-by',
                        '-c',
                        type=str,
                        default=None,
                        help='Column to use for coloring points (e.g., "temperature" from optimizer trials)')

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
            output_dir = jobs_subdir / 'throughput_analysis_plots_per_request'
        else:
            output_dir = input_dirs[0] / 'throughput_analysis_plots_per_request'
    else:
        output_dir = Path('./throughput_analysis_plots_per_request')

    print(f"Input directories: {len(input_dirs)}")
    for d in input_dirs:
        print(f"  - {d}")
    print(f"Output directory: {output_dir}")
    print()

    # Collect all job data (per-request and per-LLM-call)
    df, llm_call_df = collect_job_data(input_dirs)

    if df.empty:
        print("\nNo valid job data found!")
        sys.exit(1)

    # Print summary
    experiments = df['experiment'].unique() if 'experiment' in df.columns else ['default']
    print(f"\nCollected {len(df)} per-request data points across {len(experiments)} experiment(s)")
    if not llm_call_df.empty:
        print(f"Collected {len(llm_call_df)} per-LLM-call data points")
    print(f"TSQ Score range: {df['tsq_score'].min():.3f} - {df['tsq_score'].max():.3f}")
    print(f"TSQ Score median: {df['tsq_score'].median():.3f}")
    print(f"TSQ Score std dev: {df['tsq_score'].std():.3f}")

    if len(experiments) > 1:
        print("\nSamples per experiment:")
        for exp in experiments:
            exp_df = df[df['experiment'] == exp]
            count = len(exp_df)
            median = exp_df['tsq_score'].median()
            std = exp_df['tsq_score'].std()
            print(f"  - {exp}: {count} samples, TSQ median={median:.3f}, std={std:.3f}")

    # Report on trial matching if trial_number column exists
    if 'trial_number' in df.columns:
        matched = df['trial_number'].notna().sum()
        total = len(df)
        print(f"\nOptimizer trial matching: {matched}/{total} samples matched to trials")

        # Show unique trials and their parameters
        trial_cols = [
            c for c in df.columns if c not in [
                'job_name', 'job_label', 'experiment', 'example_number', 'sample_id', 'tsq_score', 'median_ttft_ms',
                'median_itl_ms', 'median_tps', 'total_tokens', 'num_llm_calls', 'total_duration_sec', 'p95_ttft_ms',
                'p95_itl_ms', 'trial_number'
            ]
        ]
        if trial_cols:
            print(f"  Hyperparameter columns available: {trial_cols}")
    print()

    # Create plots
    print("Creating plots...")
    create_scatter_plots(df,
                         output_dir,
                         color_by=args.color_by,
                         llm_call_df=llm_call_df if not llm_call_df.empty else None)

    print(f"\nDone! Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
