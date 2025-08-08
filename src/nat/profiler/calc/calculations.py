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

import numpy as np

from aiq.profiler.calc.data_models import FitConfig
from aiq.profiler.calc.data_models import GPUEstimates
from aiq.profiler.calc.data_models import LinearFitResult

logger = logging.getLogger(__name__)


def compute_slope(concurrencies: list[float],
                  time_metrics: list[float],
                  fit_config: FitConfig | None = None) -> LinearFitResult:
    """
    Concurrency is the independent variable (x-axis) and time metric (which can be runtime or latency)
    is the dependent variable (y-axis). This function computes the slope of the linear relationship
    between concurrency and time metric.

    Args:
        concurrencies: List of concurrency values (x-axis)
        time_metrics: List of time metric values (y-axis)
        fit_config: Configuration for outlier detection and fit validation

    Returns:
        LinearFitResult containing slope, intercept, R-squared, and outliers removed

    Raises:
        ValueError: If the relationship is not linear (R² < min_r_squared)
    """
    # Use default config if none provided
    if fit_config is None:
        fit_config = FitConfig()

    # Convert to numpy arrays for calculations
    x = np.array(concurrencies)
    y = np.array(time_metrics)

    # Validate input
    if len(x) != len(y):
        raise ValueError("Concurrencies and time_metrics must have the same length")
    if len(x) < 2:
        raise ValueError("Need at least 2 points for linear regression")

    outliers_removed = []

    # Remove outliers if requested
    if fit_config.remove_outliers and len(x) > 4:  # Need at least 4 points for outlier detection
        x_clean, y_clean, removed_concurrencies = _remove_outliers(x, y, fit_config)
        x, y = x_clean, y_clean
        outliers_removed = removed_concurrencies

    # Calculate linear regression using least squares
    n = len(x)
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_x2 = (x**2).sum()

    # Calculate slope and intercept
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    intercept = (sum_y - slope * sum_x) / n

    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_res = ((y - y_pred)**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Validate linearity
    if r_squared < fit_config.min_r_squared:
        raise ValueError(f"Poor linear fit detected (R² = {r_squared:.3f} < {fit_config.min_r_squared}). "
                         f"The relationship may not be linear. Consider using non-linear regression.")

    return LinearFitResult(slope=slope, intercept=intercept, r_squared=r_squared, outliers_removed=outliers_removed)


def _remove_outliers(x: np.ndarray, y: np.ndarray, fit_config: FitConfig) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Remove outliers using the Interquartile Range (IQR) method.
    For small concurrency range (≤ threshold points), also checks raw y-values for extreme outliers.

    Args:
        x: Input x values (concurrencies)
        y: Input y values (time metrics)
        fit_config: Configuration for outlier detection

    Returns:
        Tuple of (cleaned_x, cleaned_y, list_of_removed_concurrencies)
    """
    # if the number of concurrency points is less removing outliers can be challenging
    # as extreme outliers can skew the results.
    # We use a threshold to check for extreme outliers in raw y-values first.
    n = len(x)
    all_removed_concurrencies = []

    # For smaller concurrency ranges, check for extreme outliers in raw y-values first
    if n <= fit_config.small_concurrency_range_threshold:
        # Calculate IQR on raw y-values
        y_q1 = np.percentile(y, 25)
        y_q3 = np.percentile(y, 75)
        y_iqr = y_q3 - y_q1

        # Use a more aggressive threshold for small datasets
        y_lower_bound = y_q1 - fit_config.extreme_outlier_threshold * y_iqr  # More aggressive than 1.5
        y_upper_bound = y_q3 + fit_config.extreme_outlier_threshold * y_iqr

        # Find extreme outliers in raw values
        extreme_outlier_mask = (y >= y_lower_bound) & (y <= y_upper_bound)
        extreme_outliers_removed = np.sum(~extreme_outlier_mask)

        if extreme_outliers_removed > 0:
            extreme_removed_concurrencies = x[~extreme_outlier_mask].tolist()
            all_removed_concurrencies.extend(extreme_removed_concurrencies)
            logger.info("Removed %d extreme outliers from raw values: concurrencies %s",
                        extreme_outliers_removed,
                        extreme_removed_concurrencies)
            # Continue with residual-based detection on the cleaned data
            x = x[extreme_outlier_mask]
            y = y[extreme_outlier_mask]
            n = len(x)

    # Standard residual-based outlier detection
    # Calculate residuals from a simple linear fit
    if n == 0:
        raise ValueError("No data points remaining after outlier removal. Cannot compute linear fit.")

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
    lower_bound = q1 - fit_config.conservative_outlier_threshold * iqr
    upper_bound = q3 + fit_config.conservative_outlier_threshold * iqr

    # Find non-outlier indices
    non_outlier_mask = (residuals >= lower_bound) & (residuals <= upper_bound)

    outliers_removed = np.sum(~non_outlier_mask)
    residual_removed_concurrencies = x[~non_outlier_mask].tolist()
    all_removed_concurrencies.extend(residual_removed_concurrencies)

    # Add debugging for small datasets
    if len(x) <= fit_config.small_concurrency_range_threshold:
        logger.debug("Outlier detection for small dataset (n=%d):", len(x))
        logger.debug("  Data points: %s", list(zip(x, y)))
        logger.debug("  Residuals: %s", residuals.tolist())
        logger.debug("  Q1=%.3f, Q3=%.3f, IQR=%.3f", q1, q3, iqr)
        logger.debug("  Bounds: [%.3f, %.3f]", lower_bound, upper_bound)
        logger.info("  Outliers removed: %d (concurrencies: %s)", outliers_removed, residual_removed_concurrencies)

    return x[non_outlier_mask], y[non_outlier_mask], all_removed_concurrencies


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
