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

import numpy as np

from aiq.profiler.calc.data_models import LinearFitResult
from aiq.profiler.data_models import GPUEstimatesPerConcurrency


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

    Args:
        x: Input x values
        y: Input y values

    Returns:
        Tuple of (cleaned_x, cleaned_y, number_of_outliers_removed)
    """
    # Calculate residuals from a simple linear fit
    n = len(x)
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
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Find non-outlier indices
    non_outlier_mask = (residuals >= lower_bound) & (residuals <= upper_bound)

    outliers_removed = np.sum(~non_outlier_mask)

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
                                             observed_runtime: float) -> GPUEstimatesPerConcurrency:
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
        return GPUEstimatesPerConcurrency()

    if use_runtime and observed_runtime > target_workflow_runtime:
        return GPUEstimatesPerConcurrency()

    # Calculate multipliers (how much faster we need to be)
    llm_latency_multiplier = observed_latency / target_llm_latency if use_latency else 1.0
    wf_runtime_multiplier = observed_runtime / target_workflow_runtime if use_runtime else 1.0

    # Calculate GPU estimates using the corrected formula
    gpu_estimate_by_wf_runtime = (target_users /
                                  test_concurrency) * wf_runtime_multiplier * test_gpu_count if use_runtime else None
    gpu_estimate_by_llm_latency = (target_users /
                                   test_concurrency) * llm_latency_multiplier * test_gpu_count if use_latency else None

    return GPUEstimatesPerConcurrency(gpu_estimate_by_wf_runtime=gpu_estimate_by_wf_runtime,
                                      gpu_estimate_by_llm_latency=gpu_estimate_by_llm_latency)
