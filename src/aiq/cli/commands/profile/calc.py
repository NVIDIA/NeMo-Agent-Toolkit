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
from aiq.eval.config import GPUEstimatesPerConcurrency
from aiq.eval.config import OutOfRangeRunsPerConcurrency
from aiq.eval.runners.calc_runner import CalcRunner

logger = logging.getLogger(__name__)


@click.command("calc", help="Estimate GPU count and plot metrics for a workflow")
@click.option(
    "--config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    default=None,
    help="A YAML config file for the workflow and evaluation. This is not needed in offline mode.",
)
@click.option(
    "--offline_mode",
    is_flag=True,
    required=False,
    default=False,
    help="Run in offline mode. This is used to estimate the GPU count for a workflow without running the workflow. ")
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
    "--num_passes",
    type=int,
    required=False,
    default=0,
    help="Number of passes at each concurrency for the evaluation."
    " If set to 0 the dataset is adjusted to a multiple of the concurrency. Default: 0",
)
@click.option(
    "--append_job",
    is_flag=True,
    required=False,
    default=False,
    help="Append a job to the output directory. "
    "By default append is set to False and the content of the online directory is overwritten.",
)
@click.pass_context
def calc_command(ctx,
                 config_file,
                 offline_mode,
                 target_llm_latency,
                 target_workflow_runtime,
                 target_users,
                 test_gpu_count,
                 output_dir,
                 concurrencies,
                 num_passes,
                 append_job):
    """Estimate GPU count and plot metrics for a workflow profile."""
    # Only use CLI concurrencies, with default
    concurrencies_list = [int(x) for x in concurrencies.split(",") if x.strip()]

    # Dont allow a concurrency of 0
    if 0 in concurrencies_list:
        click.echo("Concurrency of 0 is not allowed.")
        return

    # Check if the parameters are valid in online and offline mode
    if offline_mode:
        # In offline mode target test parameters are needed to estimate the GPU count
        if target_llm_latency == 0 and target_workflow_runtime == 0:
            click.echo("Both --target_llm_latency and --target_workflow_runtime are 0. "
                       "Cannot estimate the GPU count.")
            return
        if test_gpu_count <= 0:
            click.echo("Test GPU count is 0. Cannot estimate the GPU count.")
            return
        if target_users <= 0:
            click.echo("Target users is 0. Cannot estimate the GPU count.")
            return
        if append_job:
            click.echo("Appending jobs is not supported in offline mode.")
            return
    else:
        if not config_file:
            click.echo("Config file is required in online mode.")
            return
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
        target_llm_latency_p95=target_llm_latency,
        target_workflow_runtime_p95=target_workflow_runtime,
        target_users=target_users,
        test_gpu_count=test_gpu_count,
        output_dir=output_dir,
        num_passes=num_passes,
        offline_mode=offline_mode,
        append_job=append_job,
    )

    async def run_calc() -> CalcRunnerOutput:
        runner = CalcRunner(runner_config)
        result = await runner.run()
        return result

    def print_results(results: CalcRunnerOutput):

        # Print header with target numbers
        click.echo(f"Targets: LLM Latency ≤ {runner_config.target_llm_latency_p95}s, "
                   f"Workflow Runtime ≤ {runner_config.target_workflow_runtime_p95}s, "
                   f"Users = {runner_config.target_users}")
        click.echo(f"Test parameters: GPUs = {runner_config.test_gpu_count}")

        click.echo(f"Estimated GPU count: {results.gpu_estimates.gpu_estimate_min}")
        click.echo(f"Estimated GPU count (95th percentile): {results.gpu_estimates.gpu_estimate_p95}")

        # Print per concurrency results as a table
        click.echo("Per concurrency results:")
        table = []
        for concurrency, metrics in results.sizing_metrics_per_concurrency.items():
            gpu_estimates_per_concurrency = results.gpu_estimates_per_concurrency.get(
                concurrency, GPUEstimatesPerConcurrency())
            out_of_range_per_concurrency = results.out_of_range_runs_per_concurrency.get(
                concurrency, OutOfRangeRunsPerConcurrency())
            table.append([
                concurrency,
                metrics.llm_latency_p95,
                metrics.workflow_runtime_p95,
                metrics.total_runtime,
                out_of_range_per_concurrency.num_runs_greater_than_target_latency,
                out_of_range_per_concurrency.num_runs_greater_than_target_runtime,
                gpu_estimates_per_concurrency.gpu_estimate,
                gpu_estimates_per_concurrency.gpu_estimate_by_llm_latency,
                gpu_estimates_per_concurrency.gpu_estimate_by_wf_runtime,
            ])
        headers = [
            "Concurrency",
            "LLM Latency",
            "WF Runtime",
            "Total Runtime",
            "Latency Fails",
            "Runtime Fails",
            "GPUs (Overall)",
            "GPUs (LLM Latency)",
            "GPUs (WF Runtime)",
        ]
        click.echo(tabulate(table, headers=headers, tablefmt="github"))

    results = asyncio.run(run_calc())
    print_results(results)
