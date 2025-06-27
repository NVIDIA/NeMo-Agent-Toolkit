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

        for concurrency, output in self.results.items():
            profiler_results = output.profiler_results
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

    def calc_gpu_count(self) -> float:
        """
        Estimate GPU count to meet target latency and/or workflow runtime SLO
        for a given target user load.

        Formula (if both constraints set):
            G_required = (U_target / C_test) * (L_obs / L_target) * (R_obs / R_target) * G_test

        Formula (if only latency constraint set):
            G_required = (U_target / C_test) * (L_obs / L_target) * G_test

        Formula (if only runtime constraint set):
            G_required = (U_target / C_test) * (R_obs / R_target) * G_test
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

        # Filter valid runs
        valid_runs = []
        for concurrency, output in self.results.items():
            if not output.profiler_results or not output.profiler_results.llm_latency_ci or\
                    not output.profiler_results.workflow_runtime_metrics:
                continue

            latency = output.profiler_results.llm_latency_ci.p95
            runtime = output.profiler_results.workflow_runtime_metrics.p95

            latency_ok = not use_latency or latency <= target_latency
            runtime_ok = not use_runtime or runtime <= target_runtime

            if latency_ok and runtime_ok:
                valid_runs.append((concurrency, output))

        if not valid_runs:
            logger.warning("No valid test run met both latency/runtime targets.")
            return -1

        # Use highest passing concurrency
        best_concurrency, best_output = max(valid_runs, key=lambda x: x[0])
        observed_latency = best_output.profiler_results.llm_latency_ci.p95
        observed_runtime = best_output.profiler_results.workflow_runtime_metrics.p95

        multiplier = 1.0
        if use_latency:
            multiplier *= observed_latency / target_latency
        if use_runtime:
            multiplier *= observed_runtime / target_runtime

        required_gpus = (target_users / best_concurrency) * multiplier * test_gpu_count

        logger.info(f"[GPU Estimation] concurrency={best_concurrency}, "
                    f"obs_latency={observed_latency:.3f}s, target_latency={target_latency}, "
                    f"obs_runtime={observed_runtime:.3f}s, target_runtime={target_runtime}, "
                    f"users={target_users}, test_gpus={test_gpu_count} → "
                    f"required_gpus={required_gpus:.2f}")

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
        for concurrency, output in self.results.items():
            metrics_per_concurrency[concurrency] = MetricPerConcurrency(
                p95_latency=output.profiler_results.llm_latency_ci.p95,
                p95_workflow_runtime=output.profiler_results.workflow_runtime_metrics.p95)

        # plot the metrics
        if self.config.plot_output_dir:
            self.plot_concurrency_vs_p95_metrics(self.config.plot_output_dir)

        return CalcRunnerOutput(max_tested_concurrency=max(self.config.concurrencies),
                                estimated_gpu_count=self.calc_gpu_count(),
                                metrics_per_concurrency=metrics_per_concurrency)
