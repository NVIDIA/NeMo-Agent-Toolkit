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

from aiq.eval.config import CalcRunnerConfig
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
    required=True,
    help="Target p95 LLM latency (seconds). Can be set to 0 to ignore.",
)
@click.option(
    "--target_workflow_runtime",
    type=float,
    required=True,
    help="Target p95 workflow runtime (seconds). Can be set to 0 to ignore.",
)
@click.option(
    "--target_users",
    type=int,
    required=True,
    help="Target number of users to support.",
)
@click.option(
    "--test_gpu_count",
    type=int,
    required=True,
    help="Number of GPUs used in the test.",
)
@click.option(
    "--plot_output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    default=None,
    help="Directory to save plots (optional).",
)
@click.option(
    "--concurrencies",
    type=str,
    required=False,
    default="1,2,4,8",
    help="Comma-separated list of concurrency values to test (e.g., 1,2,4,8). Default: 1,2,4,8",
)
@click.pass_context
def calc_command(ctx,
                 config_file,
                 target_llm_latency,
                 target_workflow_runtime,
                 target_users,
                 test_gpu_count,
                 plot_output_dir,
                 concurrencies):
    """Estimate GPU count and plot metrics for a workflow profile."""
    # Enforce that at least one of the targets is non-zero
    if target_llm_latency == 0 and target_workflow_runtime == 0:
        raise click.UsageError("At least one of --target_llm_latency or --target_workflow_runtime must be non-zero.")

    # Only use CLI concurrencies, with default
    concurrencies_list = [int(x) for x in concurrencies.split(",") if x.strip()]

    # Build CalcRunnerConfig
    runner_config = CalcRunnerConfig(
        config_file=config_file,
        concurrencies=concurrencies_list,
        target_p95_latency=target_llm_latency,
        target_p95_workflow_runtime=target_workflow_runtime,
        target_users=target_users,
        test_gpu_count=test_gpu_count,
        plot_output_dir=plot_output_dir,
    )

    async def run_calc():
        runner = CalcRunner(runner_config)
        result = await runner.run()

        click.echo(f"Estimated GPU count: {result.gpu_estimation.min_required_gpus}")
        click.echo(f"Estimated GPU count (95th percentile): {result.gpu_estimation.p95_required_gpus}")

        # Print results as a table
        try:
            from tabulate import tabulate
            table = []
            for concurrency, metrics in result.metrics_per_concurrency.items():
                gpu_estimate = metrics.gpu_estimate if hasattr(metrics, 'gpu_estimate') else None
                table.append([concurrency, metrics.p95_latency, metrics.p95_workflow_runtime, gpu_estimate])
            headers = ["Concurrency", "p95 Latency", "p95 Workflow Runtime", "GPU Estimate"]
            click.echo(tabulate(table, headers=headers, tablefmt="github"))
        except ImportError:
            # Fallback to manual formatting
            click.echo(
                f"{'Concurrency':>12} | {'p95 Latency':>12} | {'p95 Workflow Runtime':>20} | {'GPU Estimate':>12}")
            click.echo("-" * 65)
            for concurrency, metrics in result.metrics_per_concurrency.items():
                gpu_estimate = metrics.gpu_estimate if hasattr(metrics, 'gpu_estimate') else None
                click.echo(f"{concurrency:12} | {metrics.p95_latency:12} | "
                           f"{metrics.p95_workflow_runtime:20} | {gpu_estimate:12}")

    asyncio.run(run_calc())
