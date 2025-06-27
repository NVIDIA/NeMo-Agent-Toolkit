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

import matplotlib.pyplot as plt
import pandas as pd

from aiq.eval.config import CalcRunnerConfig
from aiq.eval.config import EvaluationRunOutput
from aiq.eval.config import MultiEvaluationRunConfig
from aiq.eval.runners.multi_eval_runner import MultiEvaluationRunner


class CalcRunner:
    """
    Runs MultiEvaluationRunner for a list of concurrencies.
    """

    def __init__(self, config: CalcRunnerConfig):
        """
        Initialize CalcRunner with a config file and a list of concurrencies.
        """
        self.config = config
        self.results: dict[str, EvaluationRunOutput] = {}

    def plot_concurrency_vs_p95_metrics(self):
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
        plt.show()

    def calc_gpu_count(self):
        """
        Calculate the number of GPUs needed to support the target users.
        """
        pass

    async def run(self):
        """
        Create a MultiEvaluationRunner with concurrency overrides.

        Each concurrency value is used to override the `eval.general.max_concurrency`
        key in the config.
        """
        config_s = "eval.general.max_concurrency"
        overrides = {f"concurrency_{c}": ((config_s, str(c)), ) for c in self.config.concurrencies}

        config = MultiEvaluationRunConfig(base_config=self.config.config_file, overrides=overrides)
        runner = MultiEvaluationRunner(config)
        await runner.run_all()
        self.results = runner.evaluation_run_outputs
