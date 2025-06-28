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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aiq.eval.config import CalcRunnerConfig
from aiq.eval.config import CalcRunnerOutput
from aiq.eval.config import EvaluationRunOutput
from aiq.eval.config import GPUEstimation
from aiq.eval.config import MetricPerConcurrency
from aiq.eval.config import MultiEvaluationRunConfig
from aiq.eval.runners.multi_eval_runner import MultiEvaluationRunner
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
        # only store profiler results per-concurrency
        self.profiler_results: dict[int, ProfilerResults] = {}

    @property
    def target_latency(self) -> float:
        return self.config.target_p95_latency

    @property
    def target_runtime(self) -> float:
        return self.config.target_p95_workflow_runtime

    @property
    def target_users(self) -> int:
        return self.config.target_users

    @property
    def test_gpu_count(self) -> int:
        return self.config.test_gpu_count

    def plot_concurrency_vs_p95_metrics(self, output_dir: Path):
        """
        Plots concurrency vs. p95 latency and workflow runtime using ProfileResults.
        """
        rows = []

        for concurrency, profiler_results in self.profiler_results.items():
            if not profiler_results or not profiler_results.llm_latency_ci or \
                    not profiler_results.workflow_runtime_metrics:
                continue

            latency = profiler_results.llm_latency_ci.p95
            workflow_runtime = profiler_results.workflow_runtime_metrics.p95

            if latency and workflow_runtime:
                rows.append({
                    "concurrency": concurrency, "p95_latency": latency, "p95_workflow_runtime": workflow_runtime
                })

        if not rows:
            logger.warning("No profile data available to plot.")
            return

        df = pd.DataFrame(rows).sort_values("concurrency")

        plt.plot(df["concurrency"], df["p95_latency"], label="p95 Latency (s)", marker="o")
        plt.plot(df["concurrency"], df["p95_workflow_runtime"], label="p95 Workflow Runtime (s)", marker="x")

        plt.xlabel("Concurrency")
        plt.ylabel("Time (seconds)")
        plt.title("Concurrency vs. p95 Latency and Workflow Runtime")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "concurrency_vs_p95_metrics.png")
        plt.close()

    def calc_min_required_gpus(self,
                               valid_runs: list[tuple[int, EvaluationRunOutput]],
                               use_latency: bool,
                               use_runtime: bool) -> float:
        """
        Estimate the minimum number of GPUs required to meet the target latency and/or workflow runtime SLA
        for a given target user load.
        """
        best_concurrency = max(valid_runs, key=lambda x: x[0])[0]
        best_output = next(output for concurrency, output in valid_runs if concurrency == best_concurrency)
        observed_latency = best_output.profiler_results.llm_latency_ci.p95
        observed_runtime = best_output.profiler_results.workflow_runtime_metrics.p95

        multiplier = 1.0
        if use_latency:
            multiplier *= observed_latency / self.target_latency
        if use_runtime:
            multiplier *= observed_runtime / self.target_runtime
        return (self.target_users / best_concurrency) * multiplier * self.test_gpu_count

    def calc_p95_required_gpus(self,
                               valid_runs: list[tuple[int, EvaluationRunOutput]],
                               use_latency: bool,
                               use_runtime: bool) -> float:
        """
        Get a gpu estimate based on every valid run and return the 95th percentile.
         """
        gpu_estimates = []

        for concurrency, output in valid_runs:
            observed_latency = output.profiler_results.llm_latency_ci.p95
            observed_runtime = output.profiler_results.workflow_runtime_metrics.p95

            multiplier = 1.0
            if use_latency:
                multiplier *= observed_latency / self.target_latency
            if use_runtime:
                multiplier *= observed_runtime / self.target_runtime

            required = (self.target_users / concurrency) * multiplier * self.test_gpu_count
            logger.info("[GPU Estimation] Concurrency=%s, Latency=%.3fs, Runtime=%.3fs GPUs required=%.2f",
                        concurrency,
                        observed_latency,
                        observed_runtime,
                        required)
            gpu_estimates.append(required)

        if not gpu_estimates:
            logger.warning("No valid GPU estimates available.")
            return -1

        # Use 95th percentile
        p95_gpu_estimate = np.percentile(gpu_estimates, 95)

        return math.ceil(p95_gpu_estimate)

    def calc_gpu_count(self) -> GPUEstimation:
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

        # Config parameters
        target_latency = self.config.target_p95_latency
        target_runtime = self.config.target_p95_workflow_runtime
        target_users = self.config.target_users
        test_gpu_count = self.config.test_gpu_count

        if target_users <= 0 or test_gpu_count <= 0:
            raise ValueError("Target users and test GPU count must be > 0.")
        if target_latency <= 0 and target_runtime <= 0:
            raise ValueError("At least one of target_p95_latency or target_p95_workflow_runtime must be > 0.")

        use_latency = target_latency > 0
        use_runtime = target_runtime > 0

        # Filter valid runs that meet the SLA
        valid_runs = []
        for concurrency, profiler_results in self.profiler_results.items():
            if not profiler_results or not profiler_results.llm_latency_ci or\
                    not profiler_results.workflow_runtime_metrics:
                continue

            latency = profiler_results.llm_latency_ci.p95
            runtime = profiler_results.workflow_runtime_metrics.p95

            latency_ok = not use_latency or latency <= target_latency
            runtime_ok = not use_runtime or runtime <= target_runtime

            if latency_ok and runtime_ok:
                valid_runs.append((concurrency, profiler_results))

        if not valid_runs:
            logger.warning("No valid test run met both latency/runtime targets.")
            return -1

        # Use highest passing concurrency to get the minimum number of GPUs required
        min_required_gpus = self.calc_min_required_gpus(valid_runs, use_latency, use_runtime)

        # If the best run is noisy we may end up under-estimating the number of GPUs required.
        # We will also calculate p95_required_gpus to account for variablility across runs
        p95_required_gpus = self.calc_p95_required_gpus(valid_runs, use_latency, use_runtime)

        logger.info("[GPU Estimation] min_required_gpus=%.2f, p95_required_gpus=%.2f",
                    min_required_gpus,
                    p95_required_gpus)

        return GPUEstimation(min_required_gpus=min_required_gpus, p95_required_gpus=p95_required_gpus)

    async def run(self) -> CalcRunnerOutput:
        """
        Create a MultiEvaluationRunner with concurrency overrides.

        Each concurrency value is used to override the `eval.general.max_concurrency`
        key in the config.
        """
        config_s = "eval.general.max_concurrency"
        overrides = {c: ((config_s, str(c)), ) for c in self.config.concurrencies}

        config = MultiEvaluationRunConfig(base_config=self.config.config_file, overrides=overrides)
        runner = MultiEvaluationRunner(config)
        await runner.run_all()
        self.profiler_results = {
            concurrency: output.profiler_results
            for concurrency, output in runner.evaluation_run_outputs.items()
        }

        metrics_per_concurrency = {}
        for concurrency, profiler_results in self.profiler_results.items():
            metrics_per_concurrency[concurrency] = MetricPerConcurrency(
                p95_latency=profiler_results.llm_latency_ci.p95,
                p95_workflow_runtime=profiler_results.workflow_runtime_metrics.p95)

        # plot the metrics
        if self.config.plot_output_dir:
            self.plot_concurrency_vs_p95_metrics(self.config.plot_output_dir)

        return CalcRunnerOutput(max_tested_concurrency=max(self.config.concurrencies),
                                gpu_estimation=self.calc_gpu_count(),
                                metrics_per_concurrency=metrics_per_concurrency)
