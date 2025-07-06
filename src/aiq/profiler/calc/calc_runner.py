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
import shutil
import time
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel
from pydantic import ValidationError

from aiq.eval.config import EvaluationRunConfig
from aiq.eval.config import MultiEvaluationRunConfig
from aiq.eval.runners.multi_eval_runner import MultiEvaluationRunner
from aiq.eval.usage_stats import UsageStats
from aiq.profiler.calc.data_models import CalcRunnerConfig
from aiq.profiler.calc.data_models import CalcRunnerOutput
from aiq.profiler.calc.data_models import CalcRunnerOutputPerConcurrency
from aiq.profiler.calc.data_models import GPUEstimates
from aiq.profiler.calc.data_models import OutOfRangeRunsPerConcurrency
from aiq.profiler.calc.data_models import SizingMetricPerItem
from aiq.profiler.calc.data_models import SizingMetricsPerConcurrency
from aiq.profiler.calc.utils import calc_gpu_estimate_based_on_slope
from aiq.profiler.calc.utils import calc_gpu_estimate_for_single_concurrency
from aiq.profiler.calc.utils import compute_slope
from aiq.profiler.data_models import GPUEstimatesPerConcurrency
from aiq.profiler.data_models import ProfilerResults

logger = logging.getLogger(__name__)


class EvalRunnerOutputForSizingMetrics(BaseModel):
    """
    Output of a single evaluation run needed for sizing metrics.
    """
    profiler_results: ProfilerResults
    usage_stats: UsageStats
    workflow_interrupted: bool


class CalcRunner:
    """
    Runs MultiEvaluationRunner for a list of concurrencies.
    """

    def __init__(self, config: CalcRunnerConfig):
        """
        Initialize CalcRunner with a config file and a list of concurrencies.
        """
        self.config = config

        # Evaluation outputs per concurrency
        self.eval_outputs: dict[int, EvalRunnerOutputForSizingMetrics] = {}

        self.metrics_per_concurrency: dict[int, SizingMetricsPerConcurrency] = {}

        # Validate configuration
        self.validate_config()

    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        Raises ValueError if configuration is invalid.
        """
        # atleast two concurrencies are needed to estimate the GPU count
        if len(self.config.concurrencies) < 2:
            raise ValueError("Atleast two concurrencies are needed to estimate the GPU count.")

        if self.config.offline_mode:
            # In offline mode target test parameters are needed to estimate the GPU count
            if self.target_latency <= 0 and self.target_runtime <= 0:
                raise ValueError("Both target_llm_latency and target_workflow_runtime are 0. "
                                 "Cannot estimate the GPU count in offline mode.")
            if self.test_gpu_count <= 0:
                raise ValueError("Test GPU count is 0. Cannot estimate the GPU count in offline mode.")
            if self.target_users <= 0:
                raise ValueError("Target users is 0. Cannot estimate the GPU count in offline mode.")
            if self.append_job:
                raise ValueError("Appending jobs is not supported in offline mode.")
            if not self.config.output_dir:
                raise ValueError("Output directory is required in offline mode.")
        else:
            # Online mode validation
            if not self.config.config_file:
                raise ValueError("Config file is required in online mode.")
            if self.target_latency <= 0 and self.target_runtime <= 0:
                logger.warning("Both target_llm_latency and target_workflow_runtime are 0. "
                               "No SLA will be enforced.")
            if self.test_gpu_count <= 0:
                logger.warning("Test GPU count is 0. Tests will be run but the GPU count will not be estimated.")
            if self.target_users <= 0:
                logger.warning("Target users is 0. Tests will be run but the GPU count will not be estimated.")

    @property
    def target_latency(self) -> float:
        return self.config.target_llm_latency_p95

    @property
    def target_runtime(self) -> float:
        return self.config.target_workflow_runtime_p95

    @property
    def target_users(self) -> int:
        return self.config.target_users

    @property
    def test_gpu_count(self) -> int:
        return self.config.test_gpu_count

    @property
    def append_job(self) -> bool:
        return self.config.append_job

    @property
    def output_dir(self) -> Path:
        return self.config.output_dir

    def calc_slope_based_gpu_estimates(self,
                                       valid_runs: list[tuple[int, SizingMetricsPerConcurrency]],
                                       use_latency: bool,
                                       use_runtime: bool) -> GPUEstimates:
        """
        Calculate GPU estimates based on linear regression slope of time metrics vs concurrency.
        """
        if len(valid_runs) < 2:
            logger.warning("Need at least 2 valid runs for slope-based estimation.")
            return GPUEstimates()

        # Extract concurrencies and metrics for slope calculation
        concurrencies = [run[0] for run in valid_runs]
        latencies = [run[1].llm_latency_p95 for run in valid_runs]
        runtimes = [run[1].workflow_runtime_p95 for run in valid_runs]

        gpu_estimate_by_wf_runtime = None
        gpu_estimate_by_llm_latency = None

        # Calculate GPU estimate based on workflow runtime slope
        if use_runtime:
            try:
                # Compute slope for runtime vs concurrency
                runtime_fit = compute_slope(concurrencies, runtimes, remove_outliers=True, min_r_squared=0.7)

                # Calculate GPU estimate using slope-based method
                gpu_estimate_by_wf_runtime = calc_gpu_estimate_based_on_slope(target_time_metric=self.target_runtime,
                                                                              target_users=self.target_users,
                                                                              test_gpu_count=self.test_gpu_count,
                                                                              observed_slope=runtime_fit.slope,
                                                                              observed_intercept=runtime_fit.intercept)

                logger.info(
                    "[GPU Estimation %s] Runtime slope=%.4f, intercept=%.4f, R²=%.3f, "
                    "outliers_removed=%d, GPUs_by_runtime=%.2f",
                    "offline" if self.config.offline_mode else "online",
                    runtime_fit.slope,
                    runtime_fit.intercept,
                    runtime_fit.r_squared,
                    runtime_fit.outliers_removed,
                    gpu_estimate_by_wf_runtime)

            except ValueError as e:
                logger.warning("Failed to calculate runtime-based GPU estimate: %s", e)
                gpu_estimate_by_wf_runtime = None

        # Calculate GPU estimate based on LLM latency slope
        if use_latency:
            try:
                # Compute slope for latency vs concurrency
                latency_fit = compute_slope(concurrencies, latencies, remove_outliers=True, min_r_squared=0.7)

                # Calculate GPU estimate using slope-based method
                gpu_estimate_by_llm_latency = calc_gpu_estimate_based_on_slope(target_time_metric=self.target_latency,
                                                                               observed_slope=latency_fit.slope,
                                                                               target_users=self.target_users,
                                                                               test_gpu_count=self.test_gpu_count,
                                                                               observed_intercept=latency_fit.intercept)

                logger.info(
                    "[GPU Estimation %s] Latency slope=%.4f, intercept=%.4f, R²=%.3f, "
                    "outliers_removed=%d, GPUs_by_latency=%.2f",
                    "offline" if self.config.offline_mode else "online",
                    latency_fit.slope,
                    latency_fit.intercept,
                    latency_fit.r_squared,
                    latency_fit.outliers_removed,
                    gpu_estimate_by_llm_latency)

            except ValueError as e:
                logger.warning("Failed to calculate latency-based GPU estimate: %s", e)
                gpu_estimate_by_llm_latency = None

        return GPUEstimates(gpu_estimate_by_wf_runtime=gpu_estimate_by_wf_runtime,
                            gpu_estimate_by_llm_latency=gpu_estimate_by_llm_latency)

    def calc_per_concurrency_metrics(
        self, sizing_metrics_per_concurrency: dict[int, SizingMetricsPerConcurrency]
    ) -> tuple[dict[int, GPUEstimatesPerConcurrency], dict[int, OutOfRangeRunsPerConcurrency]]:
        """Calculate per-concurrency GPU estimates and out-of-range runs."""
        use_latency = self.target_latency > 0
        use_runtime = self.target_runtime > 0

        gpu_estimates_per_concurrency = {}
        out_of_range_runs_per_concurrency = {}

        logger.info("Calculating per-concurrency metrics for %d concurrencies", len(sizing_metrics_per_concurrency))
        logger.info("Target users: %d, Test GPU count: %d", self.target_users, self.test_gpu_count)
        logger.info("Using targets - Latency: %s, Runtime: %s",
                    "Yes" if use_latency else "No",
                    "Yes" if use_runtime else "No")

        for concurrency, metrics_per_concurrency in sizing_metrics_per_concurrency.items():
            if not metrics_per_concurrency or not metrics_per_concurrency.llm_latency_p95 or\
                    not metrics_per_concurrency.workflow_runtime_p95:
                logger.debug("Skipping concurrency %d: missing required metrics", concurrency)
                continue

            observed_latency = metrics_per_concurrency.llm_latency_p95
            observed_runtime = metrics_per_concurrency.workflow_runtime_p95

            # Get ROUGH GPU estimates per concurrency using the centralized function
            gpu_estimates = calc_gpu_estimate_for_single_concurrency(target_llm_latency=self.target_latency,
                                                                     target_workflow_runtime=self.target_runtime,
                                                                     target_users=self.target_users,
                                                                     test_concurrency=concurrency,
                                                                     test_gpu_count=self.test_gpu_count,
                                                                     observed_latency=observed_latency,
                                                                     observed_runtime=observed_runtime)

            # Store the GPU estimates directly (no need to reconstruct the same object)
            gpu_estimates_per_concurrency[concurrency] = gpu_estimates

            # Calculate out-of-range runs based on per-item metrics (only if targets are specified)
            num_runs_greater_than_target_latency = 0
            num_runs_greater_than_target_runtime = 0

            if (use_latency or use_runtime) and metrics_per_concurrency.per_item_metrics:
                for item_metrics in metrics_per_concurrency.per_item_metrics.values():
                    if use_latency and item_metrics.llm_latency > self.target_latency:
                        num_runs_greater_than_target_latency += 1
                    if use_runtime and item_metrics.workflow_runtime > self.target_runtime:
                        num_runs_greater_than_target_runtime += 1
            else:
                logger.debug("Skipping per-item processing for concurrency %d (no targets or no per-item data)",
                             concurrency)

            # Get workflow interrupted status
            workflow_interrupted = self.eval_outputs[concurrency].workflow_interrupted

            out_of_range_runs_per_concurrency[concurrency] = OutOfRangeRunsPerConcurrency(
                num_runs_greater_than_target_latency=num_runs_greater_than_target_latency,
                num_runs_greater_than_target_runtime=num_runs_greater_than_target_runtime,
                workflow_interrupted=workflow_interrupted)

            logger.debug("Concurrency %d: GPU estimate=%.2f, out-of-range runs=%d",
                         concurrency,
                         gpu_estimates.gpu_estimate_by_wf_runtime,
                         num_runs_greater_than_target_latency + num_runs_greater_than_target_runtime)

        logger.info("Completed per-concurrency calculations:")
        logger.info("  - GPU estimates calculated for %d concurrencies", len(gpu_estimates_per_concurrency))
        logger.info("  - Out-of-range runs calculated for %d concurrencies", len(out_of_range_runs_per_concurrency))

        return gpu_estimates_per_concurrency, out_of_range_runs_per_concurrency

    def _validate_metrics_data(self, sizing_metrics_per_concurrency: dict) -> dict:
        """Validate and filter metrics data."""
        valid_metrics = {}
        for concurrency, metrics in sizing_metrics_per_concurrency.items():
            if not metrics or not metrics.llm_latency_p95 or not metrics.workflow_runtime_p95:
                logger.warning("Invalid metrics for concurrency %d: missing required fields", concurrency)
                continue
            valid_metrics[concurrency] = metrics
        return valid_metrics

    def calc_gpu_estimate(
        self, sizing_metrics_per_concurrency: dict[int, SizingMetricsPerConcurrency]
    ) -> tuple[GPUEstimates, dict[int, GPUEstimatesPerConcurrency], dict[int, OutOfRangeRunsPerConcurrency]]:
        """
        Estimate GPU count to meet target latency and/or workflow runtime SLA
        for a given target user load.
        """
        # Validate required parameters
        if self.target_users <= 0:
            logger.warning("Target users must be positive for GPU estimation")
            return GPUEstimates(), {}, {}

        if self.test_gpu_count <= 0:
            logger.warning("Test GPU count must be positive for GPU estimation")
            return GPUEstimates(), {}, {}

        # Note: Per-concurrency metrics will be calculated even without targets (baseline scaling)
        # Only slope-based estimation requires targets
        if self.target_latency <= 0 and self.target_runtime <= 0:
            logger.info("No targets specified - will calculate baseline per-concurrency estimates only")

        # Validate that metrics contain required fields
        valid_metrics = self._validate_metrics_data(sizing_metrics_per_concurrency)

        if len(valid_metrics) < 2:
            logger.warning("Need at least 2 concurrencies with valid metrics for slope-based estimation")
            return GPUEstimates(), {}, {}

        logger.info("Starting GPU estimation with %d concurrencies", len(valid_metrics))
        logger.info("Target users: %d, Test GPU count: %d", self.target_users, self.test_gpu_count)
        logger.info("Target latency: %.3fs, Target runtime: %.3fs",
                    self.target_latency if self.target_latency > 0 else 0,
                    self.target_runtime if self.target_runtime > 0 else 0)

        # Cache the validation results
        use_latency = self.target_latency > 0
        use_runtime = self.target_runtime > 0

        # Per-concurrency metrics are always calculated (with or without targets)
        # Slope-based estimation requires at least one target
        if not use_latency and not use_runtime:
            logger.info("No valid targets specified - skipping slope-based estimation")

        # Calculate per-concurrency metrics and gpu estimates
        gpu_estimates_per_concurrency, out_of_range_runs_per_concurrency = \
            self.calc_per_concurrency_metrics(valid_metrics)

        # Use all concurrencies for slope-based GPU estimation (better data for linear regression)
        all_runs = [(concurrency, metrics) for concurrency, metrics in valid_metrics.items()
                    if metrics.eligible_for_slope_based_estimation]

        if len(all_runs) < 2:
            logger.warning("Need at least 2 concurrencies with valid metrics for slope-based estimation.")
            return GPUEstimates(), gpu_estimates_per_concurrency, out_of_range_runs_per_concurrency

        # Calculate overall gpu estimates using all available data (only if targets are specified)
        if use_latency or use_runtime:
            gpu_estimates = self.calc_slope_based_gpu_estimates(all_runs, use_latency, use_runtime)
        else:
            logger.info("No targets specified - returning empty slope-based estimates")
            gpu_estimates = GPUEstimates()

        return gpu_estimates, gpu_estimates_per_concurrency, out_of_range_runs_per_concurrency

    def _build_calc_runner_output(self) -> CalcRunnerOutput:
        """
        Build CalcRunnerOutput from sizing metrics per concurrency.
        """
        if not self.metrics_per_concurrency:
            logger.warning("No metrics per concurrency found. Skipping build of CalcRunnerOutput.")
            return CalcRunnerOutput()

        logger.info("Building CalcRunnerOutput from %d concurrency metrics", len(self.metrics_per_concurrency))

        # Calculate gpu estimates
        gpu_estimates, gpu_estimates_per_concurrency, out_of_range_runs_per_concurrency = \
            self.calc_gpu_estimate(self.metrics_per_concurrency)

        # Build per-concurrency data
        per_concurrency_data = {}
        total_out_of_range_concurrencies = 0

        for concurrency in self.metrics_per_concurrency.keys():
            gpu_estimates_for_concurrency = gpu_estimates_per_concurrency.get(concurrency, GPUEstimatesPerConcurrency())
            out_of_range_runs_for_concurrency = out_of_range_runs_per_concurrency.get(
                concurrency, OutOfRangeRunsPerConcurrency())
            sizing_metrics_for_concurrency = self.metrics_per_concurrency[concurrency]

            # Only include out-of-range runs if there are actual violations
            if (out_of_range_runs_for_concurrency.num_runs_greater_than_target_latency > 0
                    or out_of_range_runs_for_concurrency.num_runs_greater_than_target_runtime > 0):
                total_out_of_range_concurrencies += 1

            per_concurrency_data[concurrency] = CalcRunnerOutputPerConcurrency(
                gpu_estimates=gpu_estimates_for_concurrency,
                out_of_range_runs=out_of_range_runs_for_concurrency,
                sizing_metrics=sizing_metrics_for_concurrency)

        logger.info("Found %d concurrencies with out-of-range runs (from %d total)",
                    total_out_of_range_concurrencies,
                    len(per_concurrency_data))

        # Log summary of GPU estimates
        if gpu_estimates.gpu_estimate_by_wf_runtime is not None:
            logger.info("GPU estimate by workflow runtime: %.2f", gpu_estimates.gpu_estimate_by_wf_runtime)
        if gpu_estimates.gpu_estimate_by_llm_latency is not None:
            logger.info("GPU estimate by LLM latency: %.2f", gpu_estimates.gpu_estimate_by_llm_latency)

        return CalcRunnerOutput(gpu_estimates=gpu_estimates, per_concurrency_data=per_concurrency_data)

    def plot_concurrency_vs_time_metrics(self, output_dir: Path):
        """
        Plots concurrency vs. p95 latency and workflow runtime using metrics_per_concurrency.
        Enhanced with better styling, trend analysis, and annotations.
        """
        import numpy as np

        rows = []

        for concurrency, metrics in self.metrics_per_concurrency.items():
            if not metrics or not metrics.llm_latency_p95 or not metrics.workflow_runtime_p95:
                continue

            latency = metrics.llm_latency_p95
            workflow_runtime = metrics.workflow_runtime_p95

            rows.append({
                "concurrency": concurrency, "llm_latency_p95": latency, "workflow_runtime_p95": workflow_runtime
            })

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
        if self.target_latency > 0:
            sla_latency = self.target_latency
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
        if self.target_runtime > 0:
            sla_runtime = self.target_runtime
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

        fig.text(0.02,
                 0.02,
                 stats_text,
                 fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

        # Adjust layout and save with high quality
        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save enhanced plot
        enhanced_plot_path = output_dir / "concurrency_vs_p95_analysis.png"
        plt.savefig(enhanced_plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        logger.info(f"Enhanced plot saved to {enhanced_plot_path}")

        # Also save a simpler version for quick viewing
        plt.figure(figsize=(12, 6))
        plt.plot(df["concurrency"], df["llm_latency_p95"], label="p95 LLM Latency (s)", marker="o", linewidth=2)
        plt.plot(df["concurrency"],
                 df["workflow_runtime_p95"],
                 label="p95 Workflow Runtime (s)",
                 marker="x",
                 linewidth=2)
        plt.xlabel("Concurrency")
        plt.ylabel("Time (seconds)")
        plt.title("Concurrency vs. p95 LLM Latency and Workflow Runtime")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        simple_plot_path = output_dir / "concurrency_vs_p95_simple.png"
        plt.savefig(simple_plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Simple plot saved to {simple_plot_path}")

    def write_output(self, output_dir: Path, calc_runner_output: CalcRunnerOutput):
        """
        Write the output to the output directory.
        """
        if not output_dir:
            logger.warning("Output directory is not set. Skipping write.")
            return

        mode = "offline" if self.config.offline_mode else "online"
        subdir = output_dir / mode

        if self.append_job:
            job_dir = subdir / f"job_{uuid.uuid4()}"
        else:
            # Clear all previous jobs when not in append mode
            existing_jobs = list(subdir.glob("job_*"))
            if existing_jobs:
                logger.info(f"Clearing {len(existing_jobs)} existing jobs")
                for job in existing_jobs:
                    if job.is_dir():
                        shutil.rmtree(job)
            # Use timestamp-based naming for better uniqueness
            job_dir = subdir / f"job_{int(time.time())}"

        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            output_path = job_dir / "calc_runner_output.json"
            output_path.write_text(calc_runner_output.model_dump_json(indent=2))

            self.plot_concurrency_vs_time_metrics(job_dir)

            logger.info("Wrote output to %s", job_dir)
        except Exception as e:
            logger.error("Failed to write output to %s: %s", job_dir, e)
            raise

    def run_offline(self) -> CalcRunnerOutput:
        """
        Run in offline mode.
        1. Read previous jobs in online mode and only append unique concurrency values to metrics_per_concurrency
        2. Calculate GPU estimates
        3. Write the output to the offline subdirectory
        """
        # Read all jobs in online mode and only append unique concurrency values to metrics_per_concurrency
        online_dir = Path(self.config.output_dir) / "online"
        if not online_dir.exists():
            logger.warning("Online directory %s does not exist. Skipping offline mode.", online_dir)
            return CalcRunnerOutput()

        # Get all job directories and sort by creation time (most recent first)
        job_dirs = [job_dir for job_dir in online_dir.iterdir() if job_dir.is_dir() and job_dir.name.startswith("job_")]
        job_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        logger.info("Found %d job directories, processing from most recent to oldest", len(job_dirs))

        for job_dir in job_dirs:
            calc_runner_output_path = job_dir / "calc_runner_output.json"
            if not calc_runner_output_path.exists():
                logger.warning("Calc runner output file %s does not exist. Skipping job %s.",
                               calc_runner_output_path,
                               job_dir.name)
                continue
            try:
                calc_output = CalcRunnerOutput.model_validate_json(calc_runner_output_path.read_text())
            except ValidationError as e:
                logger.exception("Failed to validate calc runner output file %s. Skipping job %s.",
                                 calc_runner_output_path,
                                 e,
                                 exc_info=True)
                continue

            # Extract sizing metrics from per_concurrency_data
            for concurrency, data in calc_output.per_concurrency_data.items():
                metrics = data.sizing_metrics
                if concurrency not in self.metrics_per_concurrency:
                    logger.info("Adding concurrency %s from job %s (most recent available).", concurrency, job_dir.name)
                    self.metrics_per_concurrency[concurrency] = metrics
                else:
                    # Skip since we already have this concurrency from a more recent job
                    logger.debug("Concurrency %s already exists from a more recent job. Skipping job %s.",
                                 concurrency,
                                 job_dir.name)

        if not self.metrics_per_concurrency:
            logger.warning("No valid sizing_metrics_per_concurrency found in offline mode.")
            return CalcRunnerOutput()

        # calculate gpu estimates
        calc_runner_output = self._build_calc_runner_output()

        # write the offline output
        self.write_output(self.config.output_dir, calc_runner_output)

        return calc_runner_output

    async def run_online(self) -> CalcRunnerOutput:
        """
        Create a MultiEvaluationRunner with concurrency overrides.
        Run in online mode.
        1.Run the workflow
        2. Collect profiler results and usage stats
        3. Calculate GPU estimates
        4. Write the output to the online subdirectory
        """
        # Override the concurrency and alias keys in the config
        concurrency_key = "eval.general.max_concurrency"
        alias_key = "eval.general.workflow_alias"

        overrides = {
            c: ((concurrency_key, str(c)), (alias_key, "wf_concurrency_" + str(c)))
            for c in self.config.concurrencies
        }

        # Adjust the dataset size to a multiple of the concurrency and passes
        adjust_dataset_size = True
        num_passes = self.config.num_passes

        # Instantiate the base config
        eval_run_config = EvaluationRunConfig(config_file=self.config.config_file,
                                              adjust_dataset_size=adjust_dataset_size,
                                              num_passes=num_passes)

        # Instantiate the multi-evaluation run config
        config = MultiEvaluationRunConfig(base_config=eval_run_config,
                                          overrides=overrides,
                                          endpoint=self.config.endpoint,
                                          endpoint_timeout=self.config.endpoint_timeout)

        # Instantiate and run multi-evaluation runner
        runner = MultiEvaluationRunner(config)
        await runner.run_all()
        if not runner.evaluation_run_outputs:
            logger.warning("No evaluation run outputs found. Skipping online mode.")
            return CalcRunnerOutput()

        # Stash evaluation outputs for post-processing
        self.eval_outputs = {
            concurrency:
                EvalRunnerOutputForSizingMetrics(profiler_results=output.profiler_results,
                                                 usage_stats=output.usage_stats,
                                                 workflow_interrupted=output.workflow_interrupted)
            for concurrency, output in runner.evaluation_run_outputs.items()
        }

        # Calculate sizing metrics per concurrency
        # if the workflow was interrupted, the metrics are not eligible for slope-based GPU estimation
        for concurrency, eval_output in self.eval_outputs.items():
            per_item_metrics = {
                item_id:
                    SizingMetricPerItem(llm_latency=item_metrics.llm_latency, workflow_runtime=item_metrics.runtime)
                for item_id, item_metrics in eval_output.usage_stats.usage_stats_items.items()
            }

            # if the workflow was interrupted, the metrics are not eligible for slope-based GPU estimation
            eligible_for_slope_based_estimation = not eval_output.workflow_interrupted
            self.metrics_per_concurrency[concurrency] = SizingMetricsPerConcurrency(
                llm_latency_p95=eval_output.profiler_results.llm_latency_ci.p95,
                workflow_runtime_p95=eval_output.profiler_results.workflow_runtime_metrics.p95,
                total_runtime=eval_output.usage_stats.total_runtime,
                per_item_metrics=per_item_metrics,
                eligible_for_slope_based_estimation=eligible_for_slope_based_estimation)

        # calculate gpu estimates
        calc_runner_output = self._build_calc_runner_output()

        # plot the metrics and write the output
        self.write_output(self.config.output_dir, calc_runner_output)

        return calc_runner_output

    async def run(self) -> CalcRunnerOutput:
        """
        online mode:
        1. Run the workflow
        2. Collect profiler results and usage stats
        3. Calculate GPU estimates
        4. Write the output to the online subdirectory

        offline mode:
        1. Read previous jobs in online mode and only append unique concurrency values to metrics_per_concurrency
        2. Calculate GPU estimates
        3. Write the output to the offline subdirectory
        """
        if self.config.offline_mode:
            return self.run_offline()
        else:
            return await self.run_online()
