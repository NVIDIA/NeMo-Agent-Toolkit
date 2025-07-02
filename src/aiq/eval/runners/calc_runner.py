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
import math
import shutil
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import ValidationError

from aiq.eval.config import CalcRunnerConfig
from aiq.eval.config import CalcRunnerOutput
from aiq.eval.config import EvaluationRunConfig
from aiq.eval.config import GPUEstimates
from aiq.eval.config import GPUEstimatesPerConcurrency
from aiq.eval.config import MultiEvaluationRunConfig
from aiq.eval.config import OutOfRangeRunsPerConcurrency
from aiq.eval.config import SizingMetricPerItem
from aiq.eval.config import SizingMetricsPerConcurrency
from aiq.eval.runners.multi_eval_runner import MultiEvaluationRunner
from aiq.eval.usage_stats import UsageStats
from aiq.profiler.data_models import ProfilerResults

logger = logging.getLogger(__name__)


class CalcRunner:
    """
    Runs MultiEvaluationRunner for a list of concurrencies.
    """

    def __init__(self, config: CalcRunnerConfig):
        """
        Initialize CalcRunner with a config file and a list of concurrencies.
        """
        self.config = config

        #  profiler results and usage stats per-concurrency
        self.profiler_results: dict[int, ProfilerResults] = {}
        self.usage_stats: dict[int, UsageStats] = {}

        self.metrics_per_concurrency: dict[int, SizingMetricsPerConcurrency] = {}

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

    def plot_concurrency_vs_p95_metrics(self, output_dir: Path):
        """
        Plots concurrency vs. p95 latency and workflow runtime using metrics_per_concurrency.
        """
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

        plt.plot(df["concurrency"], df["llm_latency_p95"], label="p95 LLM Latency (s)", marker="o")
        plt.plot(df["concurrency"], df["workflow_runtime_p95"], label="p95 Workflow Runtime (s)", marker="x")

        plt.xlabel("Concurrency")
        plt.ylabel("Time (seconds)")
        plt.title("Concurrency vs. p95 LLM Latency and Workflow Runtime")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "concurrency_vs_p95_metrics.png")
        plt.close()

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
            if self.config.offline_mode:
                raise ValueError("Append job is not supported in offline mode.")
            job_dir = subdir / f"job_{uuid.uuid4()}"
        else:
            # Clear previous jobs
            for job in subdir.glob("job_*"):
                if job.is_dir():
                    shutil.rmtree(job)
            job_dir = subdir / "job_0"

        job_dir.mkdir(parents=True, exist_ok=True)

        output_path = job_dir / "calc_runner_output.json"
        output_path.write_text(calc_runner_output.model_dump_json(indent=2))

        self.plot_concurrency_vs_p95_metrics(job_dir)

        logger.info("Wrote output to %s", job_dir)

    def _calc_p95_required_gpus(self,
                                valid_runs: list[tuple[int, SizingMetricsPerConcurrency]],
                                use_latency: bool,
                                use_runtime: bool) -> GPUEstimates:
        """
        Get a gpu estimate based on every valid run and return the 95th percentile.
        """
        # maintain the gpu estimation for each concurrency
        gpu_estimates = {}
        gpu_estimates_by_wf_runtime = {}
        gpu_estimates_by_llm_latency = {}

        for concurrency, metrics in valid_runs:
            observed_latency = metrics.llm_latency_p95
            observed_runtime = metrics.workflow_runtime_p95

            llm_latency_multiplier = observed_latency / self.target_latency if use_latency else 1.0
            wf_runtime_multiplier = observed_runtime / self.target_runtime if use_runtime else 1.0

            gpu_estimates_by_wf_runtime[concurrency] = (self.target_users /
                                                        concurrency) * wf_runtime_multiplier * self.test_gpu_count
            gpu_estimates_by_llm_latency[concurrency] = (self.target_users /
                                                         concurrency) * llm_latency_multiplier * self.test_gpu_count
            gpu_estimates[concurrency] = (
                self.target_users / concurrency) * wf_runtime_multiplier * llm_latency_multiplier * self.test_gpu_count
            logger.info(
                "[GPU Estimation %s] Concurrency=%s, Latency=%.3fs, Runtime=%.3fs"
                "GPUs required=%.2f (wf_runtime_based=%.2f, llm_latency_based=%.2f)",
                "offline" if self.config.offline_mode else "online",
                concurrency,
                observed_latency,
                observed_runtime,
                gpu_estimates[concurrency],
                gpu_estimates_by_wf_runtime[concurrency],
                gpu_estimates_by_llm_latency[concurrency])

        if not gpu_estimates:
            logger.warning("No valid GPU estimates available.")
            return GPUEstimates()

        # Use 95th percentile
        gpu_estimate_p95 = np.percentile(list(gpu_estimates.values()), 95)

        # Return the estimate corresponding to the highest concurrency that passed the SLA as min_required_gpus
        gpu_estimate_min = gpu_estimates[max(gpu_estimates.keys())]

        logger.info("[GPU Estimation %s] min_required_gpus=%.2f, p95_required_gpus=%.2f",
                    "offline" if self.config.offline_mode else "online",
                    gpu_estimate_min,
                    gpu_estimate_p95)

        return GPUEstimates(gpu_estimate_p95=math.ceil(gpu_estimate_p95), gpu_estimate_min=math.ceil(gpu_estimate_min))

    def _calculate_per_concurrency_metrics(
        self, sizing_metrics_per_concurrency: dict[int, SizingMetricsPerConcurrency]
    ) -> tuple[dict[int, GPUEstimatesPerConcurrency], dict[int, OutOfRangeRunsPerConcurrency]]:
        """
        Calculate per-concurrency GPU estimates and out-of-range runs.
        """
        use_latency = self.target_latency > 0
        use_runtime = self.target_runtime > 0

        gpu_estimates_per_concurrency = {}
        out_of_range_runs_per_concurrency = {}

        for concurrency, metrics_per_concurrency in sizing_metrics_per_concurrency.items():
            if not metrics_per_concurrency or not metrics_per_concurrency.llm_latency_p95 or\
                    not metrics_per_concurrency.workflow_runtime_p95:
                continue

            observed_latency = metrics_per_concurrency.llm_latency_p95
            observed_runtime = metrics_per_concurrency.workflow_runtime_p95

            # Compute multipliers
            llm_latency_multiplier = observed_latency / self.target_latency if use_latency else 1.0
            wf_runtime_multiplier = observed_runtime / self.target_runtime if use_runtime else 1.0

            gpu_estimate_by_wf_runtime = (self.target_users / concurrency) * wf_runtime_multiplier * self.test_gpu_count

            gpu_estimate_by_llm_latency = (self.target_users /
                                           concurrency) * llm_latency_multiplier * self.test_gpu_count

            gpu_estimate = (self.target_users /
                            concurrency) * wf_runtime_multiplier * llm_latency_multiplier * self.test_gpu_count

            gpu_estimates_per_concurrency[concurrency] = GPUEstimatesPerConcurrency(
                gpu_estimate_by_wf_runtime=gpu_estimate_by_wf_runtime,
                gpu_estimate_by_llm_latency=gpu_estimate_by_llm_latency,
                gpu_estimate=gpu_estimate)

            # Calculate out-of-range runs based on per-item metrics
            num_runs_greater_than_target_latency = 0
            num_runs_greater_than_target_runtime = 0

            if use_latency or use_runtime:
                for item_metrics in metrics_per_concurrency.per_item_metrics.values():
                    if use_latency and item_metrics.llm_latency > self.target_latency:
                        num_runs_greater_than_target_latency += 1
                    if use_runtime and item_metrics.workflow_runtime > self.target_runtime:
                        num_runs_greater_than_target_runtime += 1

            out_of_range_runs_per_concurrency[concurrency] = OutOfRangeRunsPerConcurrency(
                num_runs_greater_than_target_latency=num_runs_greater_than_target_latency,
                num_runs_greater_than_target_runtime=num_runs_greater_than_target_runtime)

        return gpu_estimates_per_concurrency, out_of_range_runs_per_concurrency

    def _filter_valid_runs(
        self, sizing_metrics_per_concurrency: dict[int, SizingMetricsPerConcurrency]
    ) -> list[tuple[int, SizingMetricsPerConcurrency]]:
        """
        Get valid runs across all concurrencies that meet the SLA.
        """
        use_latency = self.target_latency > 0
        use_runtime = self.target_runtime > 0

        valid_runs = []
        for concurrency, metrics in sizing_metrics_per_concurrency.items():
            if (not metrics or not metrics.llm_latency_p95 or not metrics.workflow_runtime_p95):
                continue

            latency = metrics.llm_latency_p95
            runtime = metrics.workflow_runtime_p95

            latency_ok = not use_latency or latency <= self.target_latency
            runtime_ok = not use_runtime or runtime <= self.target_runtime

            if latency_ok and runtime_ok:
                valid_runs.append((concurrency, metrics))

        return valid_runs

    def _calc_gpu_count(
        self, sizing_metrics_per_concurrency: dict[int, SizingMetricsPerConcurrency]
    ) -> tuple[GPUEstimates, dict[int, GPUEstimatesPerConcurrency], dict[int, OutOfRangeRunsPerConcurrency]]:
        """
        Estimate GPU count to meet target latency and/or workflow runtime SLA
        for a given target user load.
        This formula is derived from the following assumptions:
        - The workflow runtime increases linearly with the number of users.
        - The LLM latency increases linearly with the number of users.
        - The number of GPUs required increases linearly with the number of users.

        The formula if both constraints are set:
            G_required = (U_target / C_test) * (L_obs / L_target) * (R_obs / R_target) * G_test
        The formula if only latency constraint is set:
            G_required = (U_target / C_test) * (L_obs / L_target) * G_test
        The formula if only runtime constraint is set:
            G_required = (U_target / C_test) * (R_obs / R_target) * G_test

        where:
            - U_target: Target number of users
            - C_test: Test concurrency
            - L_obs: Observed LLM latency
            - L_target: Target LLM latency
            - R_obs: Observed workflow runtime
            - R_target: Target workflow runtime
            - G_test: Test GPU count

        Runs that don't meet the SLA are ignored for e.g. if the target runtime is 2mins,
        but the observed runtime is 3mins, the run is ignored.
        """

        if self.target_users <= 0 or self.test_gpu_count <= 0 or \
                (self.target_latency <= 0 and self.target_runtime <= 0):
            return GPUEstimates(), {}, {}

        # Calculate per-concurrency metrics and gpu estimates
        gpu_estimates_per_concurrency, out_of_range_runs_per_concurrency = \
            self._calculate_per_concurrency_metrics(sizing_metrics_per_concurrency)

        # Get valid runs for overall GPU estimation across all concurrencies
        valid_runs = self._filter_valid_runs(sizing_metrics_per_concurrency)

        if not valid_runs:
            logger.warning("No valid test run met both latency/runtime targets.")
            return GPUEstimates(), gpu_estimates_per_concurrency, out_of_range_runs_per_concurrency

        # Calculate overall gpu estimates for each concurrency that passed the SLA
        use_latency = self.target_latency > 0
        use_runtime = self.target_runtime > 0
        gpu_estimates = self._calc_p95_required_gpus(valid_runs, use_latency, use_runtime)
        return gpu_estimates, gpu_estimates_per_concurrency, out_of_range_runs_per_concurrency

    def _build_calc_runner_output(self) -> CalcRunnerOutput:
        """
        Build CalcRunnerOutput from sizing metrics per concurrency.
        This is a common function used by both offline and online modes.
        """
        if not self.metrics_per_concurrency:
            logger.warning("No metrics per concurrency found. Skipping build of CalcRunnerOutput.")
            return CalcRunnerOutput()

        # Calculate gpu estimates
        gpu_estimates, gpu_estimates_per_concurrency, out_of_range_runs_per_concurrency = \
            self._calc_gpu_count(self.metrics_per_concurrency)

        return CalcRunnerOutput(gpu_estimates=gpu_estimates,
                                gpu_estimates_per_concurrency=gpu_estimates_per_concurrency,
                                out_of_range_runs_per_concurrency=out_of_range_runs_per_concurrency,
                                sizing_metrics_per_concurrency=self.metrics_per_concurrency)

    def run_offline(self) -> CalcRunnerOutput:
        """
        Run in offline mode.
        1. Read previous jobs in online mode and only append unique concurrency values to metrics_per_concurrency
        2. Calculate GPU estimates
        3. Write the output to the offline subdirectory
        """
        if not self.config.output_dir:
            raise ValueError("Output directory is not set in offline mode.")

        # Read all jobs in online mode and only append unique concurrency values to metrics_per_concurrency
        online_dir = Path(self.config.output_dir) / "online"
        if not online_dir.exists():
            logger.warning("Online directory %s does not exist. Skipping offline mode.", online_dir)
            return CalcRunnerOutput()

        for job_dir in online_dir.iterdir():
            if job_dir.is_dir() and job_dir.name.startswith("job_"):
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

                for concurrency, metrics in calc_output.sizing_metrics_per_concurrency.items():
                    if concurrency not in self.metrics_per_concurrency:
                        logger.info("Adding concurrency %s from job %s.", concurrency, job_dir.name)
                        self.metrics_per_concurrency[concurrency] = metrics
                    else:
                        # log a warning and skip
                        logger.warning("Concurrency %s already exists in offline mode. Skipping job %s.",
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
        config = MultiEvaluationRunConfig(base_config=eval_run_config, overrides=overrides)

        # Instantiate and run multi-evaluation runner
        runner = MultiEvaluationRunner(config)
        await runner.run_all()
        if not runner.evaluation_run_outputs:
            logger.warning("No evaluation run outputs found. Skipping online mode.")
            return CalcRunnerOutput()

        # Stash profiler results for post-processing
        self.profiler_results = {
            concurrency: output.profiler_results
            for concurrency, output in runner.evaluation_run_outputs.items()
        }

        # Stash usage stats for post-processing
        self.usage_stats = {
            concurrency: output.usage_stats
            for concurrency, output in runner.evaluation_run_outputs.items()
        }

        # Calculate sizing metrics per concurrency
        for concurrency, profiler_results in self.profiler_results.items():
            if concurrency not in self.usage_stats:
                logger.warning("Missing usage stats for concurrency %s. Skipping.", concurrency)
                continue
            per_item_metrics = {
                item_id:
                    SizingMetricPerItem(llm_latency=item_metrics.llm_latency, workflow_runtime=item_metrics.runtime)
                for item_id, item_metrics in self.usage_stats[concurrency].usage_stats_items.items()
            }

            self.metrics_per_concurrency[concurrency] = SizingMetricsPerConcurrency(
                llm_latency_p95=profiler_results.llm_latency_ci.p95,
                workflow_runtime_p95=profiler_results.workflow_runtime_metrics.p95,
                total_runtime=self.usage_stats[concurrency].total_runtime,
                per_item_metrics=per_item_metrics)

        # calculate gpu estimates
        calc_runner_output = self._build_calc_runner_output()

        # plot the metrics and write the output
        self.write_output(self.config.output_dir, calc_runner_output)

        return calc_runner_output

    async def run(self) -> CalcRunnerOutput:
        """
        Create a MultiEvaluationRunner with concurrency overrides.

        Each concurrency value is used to override the `eval.general.max_concurrency`
        key in the config.
        """
        if self.config.offline_mode:
            return self.run_offline()
        else:
            return await self.run_online()
