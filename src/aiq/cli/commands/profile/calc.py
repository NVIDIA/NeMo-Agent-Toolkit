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

import asyncio
import logging
from pathlib import Path

import click
from tabulate import tabulate

from aiq.eval.config import CalcRunnerConfig
from aiq.eval.config import CalcRunnerOutput
from aiq.eval.runners.calc_runner import CalcRunner

logger = logging.getLogger(__name__)


@click.command("calc", help="Estimate GPU count and plot metrics for a workflow")
@click.option(
    "--config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="A YAML config file for the workflow and evaluation.",
)
@click.option(
    "--target_llm_latency",
    type=float,
    required=False,
    default=0,
    help="Target p95 LLM latency (seconds). Can be set to 0 to ignore.",
)
@click.option(
    "--target_workflow_runtime",
    type=float,
    required=False,
    default=0,
    help="Target p95 workflow runtime (seconds). Can be set to 0 to ignore.",
)
@click.option(
    "--target_users",
    type=int,
    required=False,
    default=0,
    help="Target number of users to support.",
)
@click.option(
    "--test_gpu_count",
    type=int,
    required=False,
    default=0,
    help="Number of GPUs used in the test.",
)
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    default=None,
    help="Directory to save plots and results (optional).",
)
@click.option(
    "--concurrencies",
    type=str,
    required=False,
    default="1,2,4,8",
    help="Comma-separated list of concurrency values to test (e.g., 1,2,4,8). Default: 1,2,4,8",
)
@click.option(
    "--reps",
    type=int,
    required=False,
    default=1,
    help="Number of repetitions for the evaluation. Default: 1",
)
@click.pass_context
def calc_command(ctx,
                 config_file,
                 target_llm_latency,
                 target_workflow_runtime,
                 target_users,
                 test_gpu_count,
                 output_dir,
                 concurrencies,
                 reps):
    """Estimate GPU count and plot metrics for a workflow profile."""
    # Only use CLI concurrencies, with default
    concurrencies_list = [int(x) for x in concurrencies.split(",") if x.strip()]

    if target_llm_latency == 0 and target_workflow_runtime == 0:
        click.echo("Both --target_llm_latency and --target_workflow_runtime are 0. "
                   "No SLA will be enforced.")

    if test_gpu_count <= 0:
        click.echo("Test GPU count is 0. Tests will be run but the GPU count will not be estimated.")

    if target_users <= 0:
        click.echo("Target users is 0. Tests will be run but the GPU count will not be estimated.")

    # Build CalcRunnerConfig
    runner_config = CalcRunnerConfig(
        config_file=config_file,
        concurrencies=concurrencies_list,
        target_p95_latency=target_llm_latency,
        target_p95_workflow_runtime=target_workflow_runtime,
        target_users=target_users,
        test_gpu_count=test_gpu_count,
        output_dir=output_dir,
        reps=reps,
    )

    async def run_calc() -> CalcRunnerOutput:
        runner = CalcRunner(runner_config)
        result = await runner.run()
        return result

    def print_results(result: CalcRunnerOutput, runner_config: CalcRunnerConfig):
        click.echo(f"Estimated GPU count: {result.gpu_estimation.min_required_gpus}")
        click.echo(f"Estimated GPU count (95th percentile): {result.gpu_estimation.p95_required_gpus}")

        # Print header with target numbers
        click.echo(f"Targets: LLM Latency ≤ {runner_config.target_p95_latency}s, "
                   f"Workflow Runtime ≤ {runner_config.target_p95_workflow_runtime}s, "
                   f"Users = {runner_config.target_users}")
        click.echo(f"Test parameters: GPUs = {runner_config.test_gpu_count}")

        # Print results as a table
        table = []
        for concurrency, metrics in result.metrics_per_concurrency.items():
            gpu_estimate = result.gpu_estimation.gpu_estimates.get(concurrency, None)
            table.append(
                [concurrency, metrics.p95_latency, metrics.p95_workflow_runtime, metrics.total_runtime, gpu_estimate])
        headers = ["Concurrency", "p95 Latency", "p95 Workflow Runtime", "Total Runtime", "GPU Estimate"]
        click.echo(tabulate(table, headers=headers, tablefmt="github"))

    result = asyncio.run(run_calc())
    print_results(result, runner_config)
