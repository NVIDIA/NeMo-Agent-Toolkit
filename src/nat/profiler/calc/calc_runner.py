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

import copy
import logging
import shutil
import time
import uuid
from pathlib import Path

from pydantic import ValidationError

from aiq.eval.config import EvaluationRunConfig
from aiq.eval.runners.config import MultiEvaluationRunConfig
from aiq.eval.runners.multi_eval_runner import MultiEvaluationRunner
from aiq.profiler.calc.calculations import LinearFitResult
from aiq.profiler.calc.calculations import calc_gpu_estimate_based_on_slope
from aiq.profiler.calc.calculations import calc_gpu_estimate_for_single_concurrency
from aiq.profiler.calc.calculations import compute_slope
from aiq.profiler.calc.data_models import CalcAlerts
from aiq.profiler.calc.data_models import CalcData
from aiq.profiler.calc.data_models import CalcRunnerConfig
from aiq.profiler.calc.data_models import CalcRunnerOutput
from aiq.profiler.calc.data_models import FitConfig
from aiq.profiler.calc.data_models import FitResults
from aiq.profiler.calc.data_models import GPUEstimates
from aiq.profiler.calc.data_models import SizingMetricPerItem
from aiq.profiler.calc.data_models import SizingMetrics
from aiq.profiler.calc.data_models import SizingMetricsAlerts

logger = logging.getLogger(__name__)


class LinearFitAnalyzer:
    """Handles linear regression analysis for concurrency vs time metrics."""

    def __init__(self, fit_config: FitConfig):
        self.fit_config = fit_config
        self.llm_latency_fit: LinearFitResult | None = None
        self.wf_runtime_fit: LinearFitResult | None = None

    def analyze_metrics(self, sizing_metrics_per_concurrency: dict[int, SizingMetrics]) -> dict[int, CalcAlerts]:
        """
        Analyze metrics and return alerts including outlier information.

        Returns:
            dict[int, CalcAlerts]: Alerts per concurrency including outlier flags
        """
        alerts_per_concurrency = {}

        # Need at least 2 points for linear regression
        if len(sizing_metrics_per_concurrency) < 2:
            logger.warning("Need at least 2 concurrencies for linear analysis")
            # Return empty alerts for all concurrencies
            for concurrency in sizing_metrics_per_concurrency.keys():
                alerts_per_concurrency[concurrency] = CalcAlerts()
            return alerts_per_concurrency

        # Calculate linear fits
        concurrencies = list(sizing_metrics_per_concurrency.keys())
        latencies = [run.llm_latency_p95 for run in sizing_metrics_per_concurrency.values()]
        try:
            self.llm_latency_fit = compute_slope(concurrencies, latencies, self.fit_config)
            logger.info("Computed latency fit: slope=%.4f, R²=%.3f",
                        self.llm_latency_fit.slope,
                        self.llm_latency_fit.r_squared)
        except ValueError as e:
            logger.warning("Failed to compute latency fit: %s", e)
            self.llm_latency_fit = None

        runtimes = [run.workflow_runtime_p95 for run in sizing_metrics_per_concurrency.values()]
        try:
            self.wf_runtime_fit = compute_slope(concurrencies, runtimes, self.fit_config)
            logger.info("Computed runtime fit: slope=%.4f, R²=%.3f",
                        self.wf_runtime_fit.slope,
                        self.wf_runtime_fit.r_squared)
        except ValueError as e:
            logger.warning("Failed to compute runtime fit: %s", e)
            self.wf_runtime_fit = None

        # Add outlier information to alerts
        for concurrency in sizing_metrics_per_concurrency.keys():
            alerts = CalcAlerts()

            # Check for latency outliers
            if self.llm_latency_fit and concurrency in self.llm_latency_fit.outliers_removed:
                alerts.outlier_llm_latency = True

            # Check for runtime outliers
            if self.wf_runtime_fit and concurrency in self.wf_runtime_fit.outliers_removed:
                alerts.outlier_workflow_runtime = True

            alerts_per_concurrency[concurrency] = alerts

        return alerts_per_concurrency


class CalcRunner:
    """
    Calculator for GPU sizing based on concurrency vs. time metrics.
    """

    def __init__(self, config: CalcRunnerConfig):
        """
        Initialize CalcRunner with a config file and a list of concurrencies.
        """
        self.config = config

        # Sizing metrics per concurrency, collected from the evaluation runs
        # This is used as input to calculate the GPU estimates and alerts
        self.metrics_per_concurrency: dict[int, SizingMetrics] = {}

        self.valid_concurrencies: list = []

        # GPU estimates and alerts
        self.gpu_estimates_per_concurrency: dict[int, GPUEstimates] = {}
        self.alerts_per_concurrency: dict[int, CalcAlerts] = {}

        # Linear fit analyzer for outlier detection and trend analysis
        self.linear_analyzer = LinearFitAnalyzer(self.config.fit_config)

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

        # if the same value is repeated in the concurrencies list, raise an error
        if len(self.config.concurrencies) != len(set(self.config.concurrencies)):
            raise ValueError("Concurrencies list contains duplicate values.")

        # The value of the concurrencies has to be greater than 0
        if any(concurrency <= 0 for concurrency in self.config.concurrencies):
            raise ValueError("Concurrencies list contains values less than or equal to 0.")

        if self.config.offline_mode:
            # In offline mode target test parameters are needed to estimate the GPU count
            if self.target_llm_latency <= 0 and self.target_wf_runtime <= 0:
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
            if self.target_llm_latency <= 0 and self.target_wf_runtime <= 0:
                logger.warning("Both target_llm_latency and target_workflow_runtime are 0. "
                               "No SLA will be enforced.")
            if self.test_gpu_count <= 0:
                logger.warning("Test GPU count is 0. Tests will be run but the GPU count will not be estimated.")
            if self.target_users <= 0:
                logger.warning("Target users is 0. Tests will be run but the GPU count will not be estimated.")

    @property
    def target_llm_latency(self) -> float:
        return self.config.target_llm_latency_p95

    @property
    def target_wf_runtime(self) -> float:
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

    def _calc_gpu_estimates_based_on_slope(self,
                                           sizing_metrics_per_concurrency: dict[int, SizingMetrics],
                                           use_latency: bool,
                                           use_runtime: bool) -> GPUEstimates:
        """
        Calculate GPU estimates based on the linear fit results
        """
        gpu_estimate_by_wf_runtime = None
        gpu_estimate_by_llm_latency = None

        if use_runtime and self.linear_analyzer.wf_runtime_fit:
            fit = self.linear_analyzer.wf_runtime_fit
            gpu_estimate_by_wf_runtime = calc_gpu_estimate_based_on_slope(target_time_metric=self.target_wf_runtime,
                                                                          target_users=self.target_users,
                                                                          test_gpu_count=self.test_gpu_count,
                                                                          observed_slope=fit.slope,
                                                                          observed_intercept=fit.intercept)
            logger.info(
                "[GPU Estimation %s] Runtime slope=%.4f, intercept=%.4f, R²=%.3f, outliers_removed=%s, estimate=%.2f",
                "offline" if self.config.offline_mode else "online",
                fit.slope,
                fit.intercept,
                fit.r_squared,
                fit.outliers_removed,
                gpu_estimate_by_wf_runtime)

        if use_latency and self.linear_analyzer.llm_latency_fit:
            fit = self.linear_analyzer.llm_latency_fit
            gpu_estimate_by_llm_latency = calc_gpu_estimate_based_on_slope(target_time_metric=self.target_llm_latency,
                                                                           target_users=self.target_users,
                                                                           test_gpu_count=self.test_gpu_count,
                                                                           observed_slope=fit.slope,
                                                                           observed_intercept=fit.intercept)
            logger.info(
                "[GPU Estimation %s] Latency slope=%.4f, intercept=%.4f, R²=%.3f, outliers_removed=%s, estimate=%.2f",
                "offline" if self.config.offline_mode else "online",
                fit.slope,
                fit.intercept,
                fit.r_squared,
                fit.outliers_removed,
                gpu_estimate_by_llm_latency)

        return GPUEstimates(gpu_estimate_by_wf_runtime=gpu_estimate_by_wf_runtime,
                            gpu_estimate_by_llm_latency=gpu_estimate_by_llm_latency)

    def _calc_gpu_estimates_per_concurrency(self, sizing_metrics_per_concurrency: dict[int, SizingMetrics]):
        """Calculate per-concurrency GPU estimates and existing alerts."""
        use_latency = self.target_llm_latency > 0
        use_runtime = self.target_wf_runtime > 0

        logger.info("Calculating per-concurrency metrics for %d concurrencies", len(sizing_metrics_per_concurrency))
        logger.info("Target users: %d, Test GPU count: %d", self.target_users, self.test_gpu_count)
        logger.info("Using targets - Latency: %s, Runtime: %s",
                    "Yes" if use_latency else "No",
                    "Yes" if use_runtime else "No")

        for concurrency, metrics_per_concurrency in sizing_metrics_per_concurrency.items():
            observed_latency = metrics_per_concurrency.llm_latency_p95
            observed_runtime = metrics_per_concurrency.workflow_runtime_p95

            # Get ROUGH GPU estimates per concurrency. This is not used for the final GPU estimation.
            # It is only available for information purposes.
            gpu_estimates = calc_gpu_estimate_for_single_concurrency(target_llm_latency=self.target_llm_latency,
                                                                     target_workflow_runtime=self.target_wf_runtime,
                                                                     target_users=self.target_users,
                                                                     test_concurrency=concurrency,
                                                                     test_gpu_count=self.test_gpu_count,
                                                                     observed_latency=observed_latency,
                                                                     observed_runtime=observed_runtime)

            # Store the GPU estimates directly (no need to reconstruct the same object)
            self.gpu_estimates_per_concurrency[concurrency] = gpu_estimates

            # Calculate out-of-range items based on per-item metrics (only if targets are specified)
            num_items_greater_than_target_latency = 0
            num_items_greater_than_target_runtime = 0

            if (use_latency or use_runtime) and metrics_per_concurrency.per_item_metrics:
                for item_metrics in metrics_per_concurrency.per_item_metrics.values():
                    if use_latency and item_metrics.llm_latency > self.target_llm_latency:
                        num_items_greater_than_target_latency += 1
                    if use_runtime and item_metrics.workflow_runtime > self.target_wf_runtime:
                        num_items_greater_than_target_runtime += 1
            else:
                logger.debug("Skipping per-item processing for concurrency %d (no targets or no per-item data)",
                             concurrency)

            # Update existing alerts with the out-of-range data
            existing_alerts = self.alerts_per_concurrency.get(concurrency, CalcAlerts())
            existing_alerts.num_items_greater_than_target_latency = num_items_greater_than_target_latency
            existing_alerts.num_items_greater_than_target_runtime = num_items_greater_than_target_runtime
            self.alerts_per_concurrency[concurrency] = existing_alerts

            logger.debug("Concurrency %d: GPU estimate=%.2f, out-of-range items=%d",
                         concurrency,
                         gpu_estimates.gpu_estimate_by_wf_runtime,
                         num_items_greater_than_target_latency + num_items_greater_than_target_runtime)

        logger.info("Completed per-concurrency calculations:")
        logger.info("  - GPU estimates calculated for %d concurrencies", len(self.gpu_estimates_per_concurrency))

    def _validate_gpu_estimation_parameters(self, use_latency: bool, use_runtime: bool) -> bool:
        """Validate parameters required for GPU estimation."""
        if self.target_users <= 0:
            logger.warning("Target users must be greater than 0 for GPU estimation")
            return False

        if self.test_gpu_count <= 0:
            logger.warning("Test GPU count must be greater than 0 for GPU estimation")
            return False

        if not use_latency and not use_runtime:
            logger.warning("No targets time metrics specified")
            return False

        return True

    def _validate_metrics_data(self, sizing_metrics_per_concurrency: dict) -> dict:
        """Validate and filter metrics data."""
        valid_metrics = {}
        for concurrency, metrics in sizing_metrics_per_concurrency.items():
            if not metrics or not metrics.llm_latency_p95 or not metrics.workflow_runtime_p95:
                logger.warning("Invalid metrics for concurrency %d: missing required fields", concurrency)
                continue
            valid_metrics[concurrency] = metrics
        return valid_metrics

    def _calc_fit_and_gpu_estimate(self, sizing_metrics_per_concurrency: dict[int, SizingMetrics]) -> GPUEstimates:
        """
        Estimate GPU count to meet target latency and/or workflow runtime SLA
        for a given target user load.

        Returns:
        - GPU estimates based on the slope of the time vs concurrency
        - GPU estimates per concurrency (rough estimates)
        - Alerts per concurrency (outliers, etc.)
        """
        gpu_estimates = GPUEstimates()
        # Filter out concurrencies that are missing required metrics
        valid_metrics = self._validate_metrics_data(sizing_metrics_per_concurrency)
        if not valid_metrics:
            logger.warning("No valid metrics found for metrics calculation")
            return gpu_estimates

        # Filter out concurrencies that were interrupted
        valid_runs = {
            concurrency: metrics
            for concurrency, metrics in valid_metrics.items() if not metrics.alerts.workflow_interrupted
        }
        if not valid_runs:
            logger.warning("No valid runs found for slope-based estimation")
            return gpu_estimates

        self.valid_concurrencies = valid_runs.keys()

        # Perform linear analysis on valid runs, this is done even if GPU estimation is skipped
        self.alerts_per_concurrency = self.linear_analyzer.analyze_metrics(valid_runs)

        # Validate GPU estimation parameters
        use_latency = self.target_llm_latency > 0
        use_runtime = self.target_wf_runtime > 0
        if not self._validate_gpu_estimation_parameters(use_latency, use_runtime):
            return gpu_estimates

        logger.info("Starting GPU estimation with %d concurrencies", len(valid_metrics))
        logger.info("Target users: %d, Test GPU count: %d", self.target_users, self.test_gpu_count)
        logger.info("Target latency: %.3fs, Target runtime: %.3fs",
                    self.target_llm_latency if self.target_llm_latency > 0 else 0,
                    self.target_wf_runtime if self.target_wf_runtime > 0 else 0)

        # Calculate GPU estimates per-concurrency
        self._calc_gpu_estimates_per_concurrency(valid_runs)

        # Calculate overall gpu estimates using linear fits
        gpu_estimates = self._calc_gpu_estimates_based_on_slope(valid_runs, use_latency, use_runtime)

        return gpu_estimates

    def generate_calc_runner_output(self) -> CalcRunnerOutput:
        """
        Build CalcRunnerOutput from sizing metrics per concurrency.
        """
        if not self.metrics_per_concurrency:
            logger.warning("No metrics per concurrency found. Skipping generation of CalcRunnerOutput.")
            return CalcRunnerOutput()

        logger.info("Building CalcRunnerOutput from %d concurrency metrics", len(self.metrics_per_concurrency))

        # Calculate gpu estimates and per-concurrency metrics
        gpu_estimates = self._calc_fit_and_gpu_estimate(self.metrics_per_concurrency)

        # Group per-concurrency data (inputs to the calculator and outputs from the calculator)
        calc_data = {}
        for concurrency in self.metrics_per_concurrency.keys():
            # Inputs to the calculator
            tmp_sizing_metrics = self.metrics_per_concurrency[concurrency]
            # Outputs from the calculator
            tmp_gpu_estimates = self.gpu_estimates_per_concurrency.get(concurrency, GPUEstimates())
            tmp_alerts = self.alerts_per_concurrency.get(concurrency, CalcAlerts())

            calc_data[concurrency] = CalcData(gpu_estimates=tmp_gpu_estimates,
                                              alerts=tmp_alerts,
                                              sizing_metrics=tmp_sizing_metrics)

        if gpu_estimates.gpu_estimate_by_wf_runtime is not None:
            logger.info("GPU estimate by workflow runtime: %.2f", gpu_estimates.gpu_estimate_by_wf_runtime)
        if gpu_estimates.gpu_estimate_by_llm_latency is not None:
            logger.info("GPU estimate by LLM latency: %.2f", gpu_estimates.gpu_estimate_by_llm_latency)

        return CalcRunnerOutput(gpu_estimates=gpu_estimates,
                                calc_data=calc_data,
                                fit_results=FitResults(llm_latency_fit=self.linear_analyzer.llm_latency_fit,
                                                       wf_runtime_fit=self.linear_analyzer.wf_runtime_fit))

    def plot_concurrency_vs_time_metrics(self, output_dir: Path):
        """Plots concurrency vs. time metrics using pre-computed fits."""
        from aiq.profiler.calc.plot import plot_concurrency_vs_time_metrics as plot_metrics

        # Only plot if we have valid metrics and at least one fit
        if not self.metrics_per_concurrency:
            logger.warning("No metrics available for plotting")
            return

        # Filter to only valid runs for plotting
        valid_runs = {
            concurrency: metrics
            for concurrency, metrics in self.metrics_per_concurrency.items() if concurrency in self.valid_concurrencies
        }

        if not valid_runs:
            logger.warning("No valid runs available for plotting")
            return
        try:
            plot_metrics(
                metrics_per_concurrency=valid_runs,  # Only valid runs
                output_dir=output_dir,
                target_llm_latency=self.target_llm_latency,
                target_runtime=self.target_wf_runtime,
                llm_latency_fit=self.linear_analyzer.llm_latency_fit,  # May be None
                runtime_fit=self.linear_analyzer.wf_runtime_fit  # May be None
            )
        except Exception as e:
            logger.exception("Failed to plot concurrency vs. time metrics: %s", e, exc_info=True)
            logger.warning("Skipping plot of concurrency vs. time metrics")

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
            # Use timestamp-based naming
            job_dir = subdir / f"job_{int(time.time())}"

        job_dir.mkdir(parents=True, exist_ok=True)

        if self.config.plot_data:
            self.plot_concurrency_vs_time_metrics(job_dir)

        output_path = job_dir / "calc_runner_output.json"
        output_path.write_text(calc_runner_output.model_dump_json(indent=2))
        logger.info("Wrote output to %s", job_dir)

    def run_offline(self) -> CalcRunnerOutput:
        """
        Run in offline mode.
        1. Read previous jobs in online mode and create sizing metrics per concurrency
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

            # Extract sizing metrics from calc_data
            for concurrency, data in calc_output.calc_data.items():
                metrics = data.sizing_metrics
                if concurrency not in self.metrics_per_concurrency:
                    logger.info("Adding concurrency %s from job %s (most recent available).", concurrency, job_dir.name)
                    logger.info("Sizing metrics: %s", metrics)
                    self.metrics_per_concurrency[concurrency] = metrics
                else:
                    # Skip since we already have this concurrency from a more recent job
                    logger.debug("Concurrency %s already exists from a more recent job. Skipping job %s.",
                                 concurrency,
                                 job_dir.name)

        # calculate gpu estimates
        calc_runner_output = self.generate_calc_runner_output()

        # write the offline output
        self.write_output(self.config.output_dir, calc_runner_output)

        return calc_runner_output

    async def run_online(self) -> CalcRunnerOutput:
        """
        Create a MultiEvaluationRunner with concurrency overrides.
        Run in online mode.
        1. Run the workflow
        2. Create sizing metrics per concurrency from the profiler results and usage stats
        3. Calculate GPU estimates
        4. Write the output to the online subdirectory
        """
        # Override the concurrency and alias keys in the config
        concurrency_key = "eval.general.max_concurrency"
        alias_key = "eval.general.workflow_alias"
        # Ensure profiler base metrics are enabled via overrides
        profiler_base_metrics_key = "eval.general.profiler.base_metrics"

        # setup the base config
        eval_run_config = EvaluationRunConfig(config_file=self.config.config_file,
                                              adjust_dataset_size=True,
                                              num_passes=self.config.num_passes,
                                              endpoint=self.config.endpoint,
                                              endpoint_timeout=self.config.endpoint_timeout)

        # Create a copy of the base config and apply the overrides for each concurrency
        configs = {}
        for concurrency in self.config.concurrencies:
            config = copy.deepcopy(eval_run_config)
            override = ((concurrency_key, str(concurrency)), (alias_key, "wf_concurrency_" + str(concurrency)),
                        (profiler_base_metrics_key, "true"))
            config.override = override
            configs[concurrency] = config

        # Instantiate the multi-evaluation run config with the overrides for each concurrency
        config = MultiEvaluationRunConfig(configs=configs)

        # Instantiate and run multi-evaluation runner
        runner = MultiEvaluationRunner(config)
        evaluation_run_outputs = await runner.run_all()
        if not evaluation_run_outputs:
            logger.warning("No evaluation run outputs found. Skipping online mode.")
            return CalcRunnerOutput()

        # Calculate sizing metrics per concurrency
        # if the workflow was interrupted, the metrics are not eligible for slope-based GPU estimation
        for concurrency, eval_output in evaluation_run_outputs.items():
            profiler_results = eval_output.profiler_results
            usage_stats = eval_output.usage_stats
            workflow_interrupted = eval_output.workflow_interrupted

            per_item_metrics = {
                item_id:
                    SizingMetricPerItem(llm_latency=item_metrics.llm_latency, workflow_runtime=item_metrics.runtime)
                for item_id, item_metrics in eval_output.usage_stats.usage_stats_items.items()
            }

            # if the workflow was interrupted, the metrics are not eligible for slope-based GPU estimation
            llm_latency_p95 = profiler_results.llm_latency_ci.p95 \
                if profiler_results.llm_latency_ci else 0
            workflow_runtime_p95 = profiler_results.workflow_runtime_metrics.p95 \
                if profiler_results.workflow_runtime_metrics else 0
            self.metrics_per_concurrency[concurrency] = SizingMetrics(
                llm_latency_p95=llm_latency_p95,
                workflow_runtime_p95=workflow_runtime_p95,
                total_runtime=usage_stats.total_runtime,
                per_item_metrics=per_item_metrics,
                alerts=SizingMetricsAlerts(workflow_interrupted=workflow_interrupted))

        # calculate gpu estimates
        calc_runner_output = self.generate_calc_runner_output()

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
