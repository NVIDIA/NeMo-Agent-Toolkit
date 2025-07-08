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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aiq.profiler.calc.data_models import GPUEstimates
from aiq.profiler.calc.data_models import LinearFitResult
from aiq.profiler.calc.data_models import SizingMetrics

logger = logging.getLogger(__name__)


def compute_slope(concurrencies: list[float],
                  time_metrics: list[float],
                  remove_outliers: bool = True,
                  min_r_squared: float = 0.7) -> LinearFitResult:
    """
    Concurrency is the independent variable (x-axis) and time metric (which can be runtime or latency)
    is the dependent variable (y-axis).

    Compute the slope and intercept of the time metric vs concurrency
    with optional outlier detection and linearity validation.

    Args:
        concurrencies: List of concurrency values (x-axis)
        time_metrics: List of time metric values (y-axis)
        remove_outliers: Whether to remove outliers using IQR method
        min_r_squared: Minimum R-squared value to consider the relationship linear

    Returns:
        LinearFitResult with slope, intercept, R-squared, and number of outliers removed

    Raises:
        ValueError: If insufficient data or poor linear fit
    """
    if len(concurrencies) != len(time_metrics) or len(concurrencies) < 2:
        raise ValueError("Need at least two data points with matching lengths.")

    x = np.array(concurrencies)
    y = np.array(time_metrics)

    outliers_removed = 0

    # Remove outliers if requested
    if remove_outliers and len(x) > 4:  # Need at least 4 points for outlier detection
        x_clean, y_clean, removed = _remove_outliers(x, y)
        x, y = x_clean, y_clean
        outliers_removed = removed

        if len(x) < 2:
            raise ValueError("After outlier removal, insufficient data points remain.")

    # Use the least squares formula for slope and intercept
    n = len(x)
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_x2 = (x**2).sum()

    # Calculate slope
    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x2 - sum_x**2

    if denominator == 0:
        raise ValueError("Cannot compute slope: denominator is zero (possibly all x values are identical)")

    slope = numerator / denominator

    # Calculate y-intercept
    intercept = (sum_y - slope * sum_x) / n

    # Calculate R-squared for linearity validation
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Validate linearity
    if r_squared < min_r_squared:
        raise ValueError(f"Poor linear fit detected (R² = {r_squared:.3f} < {min_r_squared}). "
                         f"The relationship may not be linear. Consider using non-linear regression.")

    return LinearFitResult(slope=slope, intercept=intercept, r_squared=r_squared, outliers_removed=outliers_removed)


def _remove_outliers(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Remove outliers using the Interquartile Range (IQR) method.
    For small concurrency range (≤ 10 points), also checks raw y-values for extreme outliers.

    Args:
        x: Input x values
        y: Input y values

    Returns:
        Tuple of (cleaned_x, cleaned_y, number_of_outliers_removed)
    """
    # if the number of concurrency points is less removing outliers can be challenging
    # as exteme outliers can skew the results.
    # We use a threshold of 10 points to check for extreme outliers in raw y-values first.
    small_concurrency_range_threshold = 10
    # Extreme outlier threshold is 2.0 times the IQR, extreme outliers are removed.
    extreme_outlier_threshold = 2.0
    # Conservative outlier threshold is 1.5 times the IQR, conservative outliers are removed
    conservative_outlier_threshold = 1.5

    n = len(x)

    # For smaller concurrency ranges, check for extreme outliers in raw y-values first
    if n <= small_concurrency_range_threshold:
        # Calculate IQR on raw y-values
        y_q1 = np.percentile(y, 25)
        y_q3 = np.percentile(y, 75)
        y_iqr = y_q3 - y_q1

        # Use a more aggressive threshold for small datasets
        y_lower_bound = y_q1 - extreme_outlier_threshold * y_iqr  # More aggressive than 1.5
        y_upper_bound = y_q3 + extreme_outlier_threshold * y_iqr

        # Find extreme outliers in raw values
        extreme_outlier_mask = (y >= y_lower_bound) & (y <= y_upper_bound)
        extreme_outliers_removed = np.sum(~extreme_outlier_mask)

        if extreme_outliers_removed > 0:
            logger.info("Removed %d extreme outliers from raw values", extreme_outliers_removed)
            return x[extreme_outlier_mask], y[extreme_outlier_mask], extreme_outliers_removed

    # Standard residual-based outlier detection
    # Calculate residuals from a simple linear fit
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_x2 = (x**2).sum()

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    intercept = (sum_y - slope * sum_x) / n

    # Calculate residuals
    y_pred = slope * x + intercept
    residuals = y - y_pred

    # Use IQR method to detect outliers
    q1 = np.percentile(residuals, 25)
    q3 = np.percentile(residuals, 75)
    iqr = q3 - q1

    # Define outlier bounds (1.5 * IQR rule)
    lower_bound = q1 - conservative_outlier_threshold * iqr
    upper_bound = q3 + conservative_outlier_threshold * iqr

    # Find non-outlier indices
    non_outlier_mask = (residuals >= lower_bound) & (residuals <= upper_bound)

    outliers_removed = np.sum(~non_outlier_mask)

    # Add debugging for small datasets
    if n <= small_concurrency_range_threshold:
        logger.info("Outlier detection for small dataset (n=%d):", n)
        logger.info("  Data points: %s", list(zip(x, y)))
        logger.info("  Residuals: %s", residuals.tolist())
        logger.info("  Q1=%.3f, Q3=%.3f, IQR=%.3f", q1, q3, iqr)
        logger.info("  Bounds: [%.3f, %.3f]", lower_bound, upper_bound)
        logger.info("  Outliers removed: %d", outliers_removed)

    return x[non_outlier_mask], y[non_outlier_mask], outliers_removed


def calc_gpu_estimate_based_on_slope(target_time_metric: float,
                                     target_users: int,
                                     test_gpu_count: int,
                                     observed_slope: float,
                                     observed_intercept: float = 0.0) -> float:
    """
    Calculate the GPU estimate based on the slope of the time metric.

    This function uses the linear relationship between concurrency and time metrics
    to estimate the required GPU count for a target user load.

    Args:
        target_time_metric: Target time metric (latency or runtime) in seconds
        observed_slope: Slope from linear regression of time vs concurrency
        target_users: Target number of concurrent users
        test_gpu_count: Number of GPUs used in the test
        observed_intercept: Y-intercept from linear regression (default: 0.0)

    Returns:
        Estimated number of GPUs required

    Raises:
        ValueError: If target_time_metric is less than or equal to intercept
    """
    if target_time_metric <= observed_intercept:
        raise ValueError(f"Target time metric ({target_time_metric}) must be greater than "
                         f"the intercept ({observed_intercept}) for valid GPU estimation.")

    # Calculate the concurrency that would achieve the target time metric
    # Using the linear equation: time = slope * concurrency + intercept
    # Solving for concurrency: concurrency = (time - intercept) / slope
    calculated_concurrency = (target_time_metric - observed_intercept) / observed_slope
    logger.info("Calculated concurrency: %f for target time metric: %f, observed intercept: %f, observed slope: %f",
                calculated_concurrency,
                target_time_metric,
                observed_intercept,
                observed_slope)

    if calculated_concurrency <= 0:
        raise ValueError(f"Calculated target concurrency ({calculated_concurrency}) is not positive. "
                         f"This suggests the slope or intercept values may be invalid.")

    # Estimate GPUs using the ratio of target users to target concurrency
    # scaled by the test GPU count
    gpu_estimate = (target_users / calculated_concurrency) * test_gpu_count

    return gpu_estimate


def calc_gpu_estimate_for_single_concurrency(target_llm_latency: float,
                                             target_workflow_runtime: float,
                                             target_users: int,
                                             test_concurrency: int,
                                             test_gpu_count: int,
                                             observed_latency: float,
                                             observed_runtime: float) -> GPUEstimates:
    """
    ROUGH ESTIMATE: Calculate GPU count estimate for a single concurrency level.

    This is a simplified estimate that assumes linear scaling and should be used
    as a baseline only. For more accurate estimates, use slope-based estimation
    with multiple concurrency levels.

    Formula based on the target latency:
        G_required = (U_target / C_test) * (L_obs / L_target) * G_test

    Formula based on the target runtime:
        G_required = (U_target / C_test) * (R_obs / R_target) * G_test

    where:
        - U_target: Target number of users
        - C_test: Test concurrency level
        - L_obs: Observed LLM latency
        - L_target: Target LLM latency
        - R_obs: Observed workflow runtime
        - R_target: Target workflow runtime
        - G_test: Test GPU count

    WARNING: This is a rough estimate that:
    - Assumes perfect linear scaling (rarely true in practice)
    - Doesn't account for GPU utilization inefficiencies
    - May underestimate GPU requirements for high concurrency
    - Should be validated against slope-based estimates
    """
    use_latency = target_llm_latency > 0
    use_runtime = target_workflow_runtime > 0

    # If observed latency or runtime exceeds the target, return empty estimates
    if use_latency and observed_latency > target_llm_latency:
        return GPUEstimates()

    if use_runtime and observed_runtime > target_workflow_runtime:
        return GPUEstimates()

    # Calculate multipliers (how much faster we need to be)
    llm_latency_multiplier = observed_latency / target_llm_latency if use_latency else 1.0
    wf_runtime_multiplier = observed_runtime / target_workflow_runtime if use_runtime else 1.0

    # Calculate GPU estimates using the corrected formula
    gpu_estimate_by_wf_runtime = (target_users /
                                  test_concurrency) * wf_runtime_multiplier * test_gpu_count if use_runtime else None
    gpu_estimate_by_llm_latency = (target_users /
                                   test_concurrency) * llm_latency_multiplier * test_gpu_count if use_latency else None

    return GPUEstimates(gpu_estimate_by_wf_runtime=gpu_estimate_by_wf_runtime,
                        gpu_estimate_by_llm_latency=gpu_estimate_by_llm_latency)


def plot_concurrency_vs_time_metrics(metrics_per_concurrency: dict[int, SizingMetrics],
                                     output_dir: Path,
                                     target_latency: float = 0.0,
                                     target_runtime: float = 0.0) -> None:
    """
    Plot concurrency vs. p95 latency and workflow runtime using metrics_per_concurrency.
    Enhanced with better styling, trend analysis, and annotations.

    Args:
        metrics_per_concurrency: Dictionary mapping concurrency to metrics objects
        output_dir: Directory to save the plots
        target_latency: Target p95 LLM latency (seconds). If 0, no SLA line is drawn
        target_runtime: Target p95 workflow runtime (seconds). If 0, no SLA line is drawn
    """
    rows = []

    for concurrency, metrics in metrics_per_concurrency.items():
        if not metrics or not metrics.llm_latency_p95 or not metrics.workflow_runtime_p95:
            continue

        latency = metrics.llm_latency_p95
        workflow_runtime = metrics.workflow_runtime_p95

        rows.append({"concurrency": concurrency, "llm_latency_p95": latency, "workflow_runtime_p95": workflow_runtime})

    if not rows:
        logger.warning("No metrics data available to plot.")
        return

    df = pd.DataFrame(rows).sort_values("concurrency")

    # Create enhanced plot with better styling
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: LLM Latency
    ax1.scatter(df["concurrency"],
                df["llm_latency_p95"],
                alpha=0.7,
                s=120,
                c='steelblue',
                edgecolors='white',
                linewidth=1,
                label='Data Points')

    # Add trend line for latency
    if len(df) > 1:
        z_latency = np.polyfit(df["concurrency"], df["llm_latency_p95"], 1)
        p_latency = np.poly1d(z_latency)
        slope_latency = z_latency[0]

        # Calculate R-squared for latency
        y_pred_latency = p_latency(df["concurrency"])
        ss_res_latency = np.sum((df["llm_latency_p95"] - y_pred_latency)**2)
        ss_tot_latency = np.sum((df["llm_latency_p95"] - df["llm_latency_p95"].mean())**2)
        r_squared_latency = 1 - (ss_res_latency / ss_tot_latency) if ss_tot_latency != 0 else 0

        ax1.plot(df["concurrency"],
                 p_latency(df["concurrency"]),
                 "r--",
                 alpha=0.8,
                 linewidth=2,
                 label=f'Trend (slope={slope_latency:.4f}, R²={r_squared_latency:.3f})')

    # Add SLA reference line for latency (only if target is positive)
    if target_latency > 0:
        sla_latency = target_latency
        ax1.axhline(y=sla_latency,
                    color='red',
                    linestyle=':',
                    alpha=0.7,
                    linewidth=2,
                    label=f'SLA Threshold ({sla_latency}s)')

    ax1.set_xlabel('Concurrency', fontsize=12, fontweight='bold')
    ax1.set_ylabel('P95 LLM Latency (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Concurrency vs P95 LLM Latency', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot 2: Workflow Runtime
    ax2.scatter(df["concurrency"],
                df["workflow_runtime_p95"],
                alpha=0.7,
                s=120,
                c='darkgreen',
                edgecolors='white',
                linewidth=1,
                label='Data Points')

    # Add trend line for runtime
    if len(df) > 1:
        z_runtime = np.polyfit(df["concurrency"], df["workflow_runtime_p95"], 1)
        p_runtime = np.poly1d(z_runtime)
        slope_runtime = z_runtime[0]

        # Calculate R-squared for runtime
        y_pred_runtime = p_runtime(df["concurrency"])
        ss_res_runtime = np.sum((df["workflow_runtime_p95"] - y_pred_runtime)**2)
        ss_tot_runtime = np.sum((df["workflow_runtime_p95"] - df["workflow_runtime_p95"].mean())**2)
        r_squared_runtime = 1 - (ss_res_runtime / ss_tot_runtime) if ss_tot_runtime != 0 else 0

        ax2.plot(df["concurrency"],
                 p_runtime(df["concurrency"]),
                 "r--",
                 alpha=0.8,
                 linewidth=2,
                 label=f'Trend (slope={slope_runtime:.4f}, R²={r_squared_runtime:.3f})')

    # Add SLA reference line for runtime (only if target is positive)
    if target_runtime > 0:
        sla_runtime = target_runtime
        ax2.axhline(y=sla_runtime,
                    color='red',
                    linestyle=':',
                    alpha=0.7,
                    linewidth=2,
                    label=f'SLA Threshold ({sla_runtime}s)')

    ax2.set_xlabel('Concurrency', fontsize=12, fontweight='bold')
    ax2.set_ylabel('P95 Workflow Runtime (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Concurrency vs P95 Workflow Runtime', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Add summary statistics
    stats_text = f'Data Points: {len(df)}\n'
    stats_text += f'Concurrency Range: {df["concurrency"].min()}-{df["concurrency"].max()}\n'
    stats_text += f'Latency Range: {df["llm_latency_p95"].min():.3f}-{df["llm_latency_p95"].max():.3f}s\n'
    stats_text += f'Runtime Range: {df["workflow_runtime_p95"].min():.3f}-{df["workflow_runtime_p95"].max():.3f}s'

    fig.text(0.02, 0.02, stats_text, fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    # Adjust layout and save with high quality
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save enhanced plot
    enhanced_plot_path = output_dir / "concurrency_vs_p95_analysis.png"
    plt.savefig(enhanced_plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    logger.info("Enhanced plot saved to %s", enhanced_plot_path)

    # Also save a simpler version for quick viewing
    plt.figure(figsize=(12, 6))
    plt.plot(df["concurrency"], df["llm_latency_p95"], label="p95 LLM Latency (s)", marker="o", linewidth=2)
    plt.plot(df["concurrency"], df["workflow_runtime_p95"], label="p95 Workflow Runtime (s)", marker="s", linewidth=2)
    plt.xlabel("Concurrency")
    plt.ylabel("Time (seconds)")
    plt.title("Concurrency vs. p95 LLM Latency and Workflow Runtime")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    simple_plot_path = output_dir / "concurrency_vs_p95_simple.png"
    plt.savefig(simple_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("Simple plot saved to %s", simple_plot_path)
