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

from aiq.profiler.calc.calc_runner import CalcRunner
from aiq.profiler.calc.data_models import CalcRunnerConfig
from aiq.profiler.calc.data_models import CalcRunnerOutput

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
    "--calc_output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    default=None,
    help="Directory to save plots and results (optional).",
)
@click.option(
    "--concurrencies",
    type=str,
    required=False,
    default="1,2,3,4,5,6,7,8,9,10",
    help="Comma-separated list of concurrency values to test (e.g., 1,2,4,8). Default: 1,2,3,4,5,6,7,8,9,10",
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
    "--append_calc_outputs",
    is_flag=True,
    required=False,
    default=False,
    help="Append calc outputs to the output directory. "
    "By default append is set to False and the content of the online directory is overwritten.",
)
@click.option(
    "--endpoint",
    type=str,
    required=False,
    default=None,
    help="Endpoint to use for the workflow if it is remote(optional).",
)
@click.option(
    "--endpoint_timeout",
    type=int,
    required=False,
    default=300,
    help="Timeout for the remote workflow endpoint in seconds (default: 300).",
)
@click.pass_context
def calc_command(ctx,
                 config_file,
                 offline_mode,
                 target_llm_latency,
                 target_workflow_runtime,
                 target_users,
                 test_gpu_count,
                 calc_output_dir,
                 concurrencies,
                 num_passes,
                 append_calc_outputs,
                 endpoint,
                 endpoint_timeout):
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
        if append_calc_outputs:
            click.echo("Appending calc outputs is not supported in offline mode.")
            return
        if not calc_output_dir:
            click.echo("Output directory is required in offline mode.")
            return
    else:
        if not config_file:
            click.echo("Config file is required in online mode.")
            return
        if target_llm_latency == 0 and target_workflow_runtime == 0:
            click.echo("Both --target_llm_latency and --target_workflow_runtime are 0. "
                       "GPU count will not be estimated.")
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
        output_dir=calc_output_dir,
        num_passes=num_passes,
        offline_mode=offline_mode,
        append_job=append_calc_outputs,
        endpoint=endpoint,
        endpoint_timeout=endpoint_timeout,
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

        # Check if there are any GPU estimates to determine if we should show GPU estimate columns
        has_llm_latency_gpu_estimates = any(data.gpu_estimates.gpu_estimate_by_llm_latency is not None
                                            for data in results.calc_data.values())
        has_wf_runtime_gpu_estimates = any(data.gpu_estimates.gpu_estimate_by_wf_runtime is not None
                                           for data in results.calc_data.values())

        # Check if there are any interrupted workflows or outliers to determine if we should show the alerts column
        has_alerts = any(data.sizing_metrics.alerts.workflow_interrupted or data.alerts.outlier_llm_latency
                         or data.alerts.outlier_workflow_runtime for data in results.calc_data.values())

        # Print per concurrency results as a table
        click.echo("Per concurrency results:")

        # Show alerts legend if there are any alerts
        if has_alerts:
            click.echo("Alerts!: W = Workflow interrupted, L = LLM latency outlier, R = Workflow runtime outlier")

        table = []
        for concurrency, data in results.calc_data.items():
            metrics = data.sizing_metrics
            gpu_estimates_per_concurrency = data.gpu_estimates
            sizing_metrics_alerts = data.sizing_metrics.alerts
            calc_alerts = data.alerts

            row = []

            # Only include alerts column if there are any interrupted workflows (first column)
            if has_alerts:
                alerts = []
                if sizing_metrics_alerts.workflow_interrupted:
                    alerts.append("W")
                if calc_alerts.outlier_llm_latency:
                    alerts.append("L")
                if calc_alerts.outlier_workflow_runtime:
                    alerts.append("R")

                # Show ! followed by all alert characters
                if alerts:
                    row.append(f"!{''.join(alerts)}")
                else:
                    row.append("")

            row.extend([
                concurrency,
                metrics.llm_latency_p95,
                metrics.workflow_runtime_p95,
                metrics.total_runtime,
            ])

            # Only include GPU estimate columns if there are actual estimates of that type
            if has_llm_latency_gpu_estimates:
                row.append(gpu_estimates_per_concurrency.gpu_estimate_by_llm_latency)
            if has_wf_runtime_gpu_estimates:
                row.append(gpu_estimates_per_concurrency.gpu_estimate_by_wf_runtime)

            table.append(row)

        headers = []

        # Only include alerts header if there are any alerts (first column)
        if has_alerts:
            headers.append("Alerts")

        headers.extend([
            "Concurrency",
            "p95 LLM Latency",
            "p95 WF Runtime",
            "Total Runtime",
        ])

        # Only include GPU estimate headers if there are actual estimates of that type
        if has_llm_latency_gpu_estimates:
            headers.append("GPUs (LLM Latency, Rough)")
        if has_wf_runtime_gpu_estimates:
            headers.append("GPUs (WF Runtime, Rough)")

        click.echo(tabulate(table, headers=headers, tablefmt="github"))

        # Display slope-based GPU estimates at the end
        click.echo("")  # Add blank line for separation
        click.echo(click.style("=== GPU ESTIMATES ===", fg="bright_blue", bold=True))
        if results.gpu_estimates.gpu_estimate_by_wf_runtime is not None:
            click.echo(
                click.style(
                    f"Estimated GPU count (Workflow Runtime): {results.gpu_estimates.gpu_estimate_by_wf_runtime:.1f}",
                    fg="green",
                    bold=True))
        if results.gpu_estimates.gpu_estimate_by_llm_latency is not None:
            click.echo(
                click.style(
                    f"Estimated GPU count (LLM Latency): {results.gpu_estimates.gpu_estimate_by_llm_latency:.1f}",
                    fg="green",
                    bold=True))

    results = asyncio.run(run_calc())
    print_results(results)
