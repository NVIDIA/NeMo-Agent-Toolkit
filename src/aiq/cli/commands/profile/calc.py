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
import yaml

from aiq.eval.config import CalcRunnerConfig
from aiq.eval.runners.calc_runner import CalcRunner

logger = logging.getLogger(__name__)


@click.command("calc", help="Estimate GPU count and plot metrics for a workflow profile.")
@click.option(
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="A YAML config file for the workflow and evaluation.",
)
@click.option(
    "--target_llm_latency",
    type=float,
    required=True,
    help="Target p95 LLM latency (seconds).",
)
@click.option(
    "--target_workflow_runtime",
    type=float,
    required=True,
    help="Target p95 workflow runtime (seconds).",
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
@click.pass_context
def calc_command(ctx,
                 config,
                 target_llm_latency,
                 target_workflow_runtime,
                 target_users,
                 test_gpu_count,
                 plot_output_dir):
    """Estimate GPU count and plot metrics for a workflow profile."""
    # Load the config YAML
    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)

    # Extract concurrencies from config or set a default
    concurrencies = config_dict.get("concurrencies", [1, 2, 4, 8, 16])

    # Build CalcRunnerConfig
    runner_config = CalcRunnerConfig(
        config_file=config,
        concurrencies=concurrencies,
        target_p95_latency=target_llm_latency,
        target_p95_workflow_runtime=target_workflow_runtime,
        target_users=target_users,
        test_gpu_count=test_gpu_count,
        plot_output_dir=plot_output_dir,
    )

    async def run_calc():
        runner = CalcRunner(runner_config)
        result = await runner.run()
        logger.info(f"Max tested concurrency: {result.max_tested_concurrency}")
        logger.info(f"Estimated GPU count: {result.estimated_gpu_count}")
        logger.info(f"Metrics per concurrency: {result.metrics_per_concurrency}")
        click.echo(f"Max tested concurrency: {result.max_tested_concurrency}")
        click.echo(f"Estimated GPU count: {result.estimated_gpu_count}")
        click.echo(f"Metrics per concurrency: {result.metrics_per_concurrency}")

    asyncio.run(run_calc())
