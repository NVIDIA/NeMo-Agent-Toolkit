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
import pandas as pd

from aiq.eval.config import CalcRunnerConfig
from aiq.eval.config import CalcRunnerOutput
from aiq.eval.config import EvaluationRunOutput
from aiq.eval.config import MetricPerConcurrency
from aiq.eval.config import MultiEvaluationRunConfig
from aiq.eval.runners.multi_eval_runner import MultiEvaluationRunner

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
        # results per-concurrency
        self.results: dict[int, EvaluationRunOutput] = {}

    def plot_concurrency_vs_p95_metrics(self, output_dir: Path):
        """
        Plots concurrency vs. p95 latency and workflow runtime using ProfileResults.
        """
        rows = []

        for run_id, output in self.results.items():
            profiler_results = output.profiler_results
            concurrency = int(run_id.split("_")[-1])

            latency = profiler_results.llm_latency_ci.p95
            workflow_runtime = profiler_results.workflow_runtime_metrics.p95

            if latency and workflow_runtime:
                rows.append({
                    "concurrency": concurrency,
                    "p95_latency": latency.p95,
                    "p95_workflow_runtime": workflow_runtime.p95
                })

        if not rows:
            print("No profile data available to plot.")
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

    def calc_gpu_count(self) -> float:
        """
        Estimate the number of GPUs required to meet latency SLO for a target number of users.

        This uses the highest concurrency run where the p95 latency is still within the target.

        Formula:
            G_required = (U_target / C_test) * (L_obs / L_target) * G_test

        Where:
            - U_target: desired number of users to support
            - C_test: concurrency level used in test
            - L_obs: observed p95 latency
            - L_target: target p95 latency
            - G_test: number of GPUs used during the test
        """
        target_latency = self.config.target_p95_latency
        target_users = self.config.target_users
        test_gpu_count = self.config.test_gpu_count

        if target_latency <= 0:
            raise ValueError("Target p95 latency must be greater than 0.")
        if test_gpu_count <= 0:
            raise ValueError("Test GPU count must be greater than 0.")
        if target_users <= 0:
            raise ValueError("Target user count must be greater than 0.")

        # Find all datapoints that meet the latency target
        valid_runs = [(int(concurrency.split("_")[-1]), output) for concurrency, output in self.results.items()
                      if output.profiler_results.llm_latency_ci.p95 <= target_latency]

        if not valid_runs:
            logger.warning("No valid runs found that meet the latency target.")
            return -1

        # Use the highest concurrency that passed
        best_concurrency, best_output = max(valid_runs, key=lambda x: x[0])
        observed_latency = best_output.profiler_results.llm_latency_ci.p95

        required_gpus = ((target_users / best_concurrency) * (observed_latency / target_latency) * test_gpu_count)

        # Optional: round up to whole number of GPUs
        return math.ceil(required_gpus)

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
        self.results = runner.evaluation_run_outputs

        metrics_per_concurrency = {}
        for run_id, output in self.results.items():
            concurrency = int(run_id.split("_")[-1])
            metrics_per_concurrency[concurrency] = MetricPerConcurrency(
                p95_latency=output.profiler_results.llm_latency_ci.p95,
                p95_workflow_runtime=output.profiler_results.workflow_runtime_metrics.p95)

        # plot the metrics
        if self.config.plot_output_dir:
            self.plot_concurrency_vs_p95_metrics(self.config.plot_output_dir)

        return CalcRunnerOutput(max_tested_concurrency=max(self.config.concurrencies),
                                estimated_gpu_count=self.calc_gpu_count(),
                                metrics_per_concurrency=metrics_per_concurrency)
