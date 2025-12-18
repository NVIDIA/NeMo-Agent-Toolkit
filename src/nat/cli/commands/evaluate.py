# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nat.eval.config import EvaluationRunOutput
from nat.eval.evaluate import EvaluationRun
from nat.eval.evaluate import EvaluationRunConfig

logger = logging.getLogger(__name__)


@click.group(name=__name__, invoke_without_command=True, help="Evaluate a workflow with the specified dataset.")
@click.option(
    "--config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="A JSON/YAML file that sets the parameters for the workflow and evaluation.",
)
@click.option(
    "--dataset",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="A json file with questions and ground truth answers. This will override the dataset path in the config file.",
)
@click.option(
    "--result_json_path",
    type=str,
    default="$",
    help=("A JSON path to extract the result from the workflow. Use this when the workflow returns "
          "multiple objects or a dictionary. For example, '$.output' will extract the 'output' field "
          "from the result."),
)
@click.option(
    "--skip_workflow",
    is_flag=True,
    default=False,
    help="Skip the workflow execution and use the provided dataset for evaluation. "
    "In this case the dataset should have the 'generated_' columns.",
)
@click.option(
    "--skip_completed_entries",
    is_flag=True,
    default=False,
    help="Skip the dataset entries that have a generated answer.",
)
@click.option(
    "--endpoint",
    type=str,
    default=None,
    help="Use endpoint for running the workflow. Example: http://localhost:8000/generate",
)
@click.option(
    "--endpoint_timeout",
    type=int,
    default=300,
    help="HTTP response timeout in seconds. Only relevant if endpoint is specified.",
)
@click.option(
    "--reps",
    type=int,
    default=1,
    help="Number of repetitions for the evaluation.",
)
@click.option(
    "--override",
    type=(str, str),
    multiple=True,
    help="Override config values using dot notation (e.g., --override llms.nim_llm.temperature 0.7)",
)
@click.option(
    "--user_id",
    type=str,
    default=None,
    help="User ID to use for workflow session.",
)
@click.pass_context
def eval_command(ctx, **kwargs) -> None:
    """ Evaluate datasets with the specified mechanism"""
    pass


def write_tabular_output(eval_run_output: EvaluationRunOutput):
    """Write evaluation results in a tabular format."""

    # Print header with workflow status and runtime
    workflow_status = "INTERRUPTED" if eval_run_output.workflow_interrupted else "COMPLETED"
    total_runtime = eval_run_output.usage_stats.total_runtime if eval_run_output.usage_stats else 0.0

    click.echo("")
    click.echo(click.style("=== EVALUATION SUMMARY ===", fg="bright_blue", bold=True))
    click.echo(f"Workflow Status: {workflow_status} (workflow_output.json)")
    click.echo(f"Total Runtime: {total_runtime:.2f}s")

    # Include profiler stats if available
    if eval_run_output.profiler_results:
        profiler_results = eval_run_output.profiler_results
        if profiler_results.workflow_runtime_metrics:
            wf_metrics = profiler_results.workflow_runtime_metrics
            click.echo(f"Workflow Runtime (p95): {wf_metrics.p95:.2f}s")
        if profiler_results.llm_latency_ci:
            llm_metrics = profiler_results.llm_latency_ci
            click.echo(f"LLM Latency (p95): {llm_metrics.p95:.2f}s")

    # Build the evaluation results table
    if not eval_run_output.evaluation_results:
        return

    click.echo("")
    click.echo("Per evaluator results:")

    table = []
    for evaluator_name, eval_output in eval_run_output.evaluation_results:
        row = []

        # Add evaluator name and average score
        row.append(evaluator_name)

        # Format average score
        if isinstance(eval_output.average_score, int | float):
            row.append(f"{eval_output.average_score:.4f}")
        else:
            row.append(str(eval_output.average_score))

        # Add output file if available
        output_file = None
        for file_path in eval_run_output.evaluator_output_files:
            if file_path.stem.startswith(f"{evaluator_name}_") or file_path.stem == evaluator_name:
                output_file = file_path.name
                break
        row.append(output_file if output_file else "N/A")

        table.append(row)

    # Build headers
    headers = ["Evaluator", "Avg Score", "Output File"]

    click.echo(tabulate(table, headers=headers, tablefmt="github"))
    click.echo("")


async def run_and_evaluate(config: EvaluationRunConfig):
    # Run evaluation
    eval_runner = EvaluationRun(config=config)
    eval_run_output = await eval_runner.run_and_evaluate()

    write_tabular_output(eval_run_output)


@eval_command.result_callback(replace=True)
def process_nat_eval(
    processors,
    *,
    config_file: Path,
    dataset: Path,
    result_json_path: str,
    skip_workflow: bool,
    skip_completed_entries: bool,
    endpoint: str,
    endpoint_timeout: int,
    reps: int,
    override: tuple[tuple[str, str], ...],
    user_id: str | None,
):
    """
    Process the eval command and execute the evaluation. Here the config_file, if provided, is checked for its existence
    on disk.
    """
    # Cannot skip_workflow if endpoint is specified
    if skip_workflow and endpoint:
        raise click.UsageError("The options '--skip_workflow' and '--endpoint' are mutually exclusive. "
                               "Please use only one of them.")

    # You cannot run multiple repetitions if you are skipping the workflow or skipping completed entries
    if reps > 1 and (skip_workflow or skip_completed_entries):
        raise click.UsageError("The options '--reps' and '--skip_workflow' or '--skip_completed_entries' are mutually "
                               "exclusive. You cannot run multiple repetitions if you are skipping the workflow or "
                               "have a partially completed dataset.")

    # Create the configuration object
    # Only include user_id if explicitly provided via CLI, otherwise use the default
    config_kwargs = {
        "config_file": config_file,
        "dataset": str(dataset) if dataset else None,
        "result_json_path": result_json_path,
        "skip_workflow": skip_workflow,
        "skip_completed_entries": skip_completed_entries,
        "endpoint": endpoint,
        "endpoint_timeout": endpoint_timeout,
        "reps": reps,
        "override": override,
    }
    if user_id is not None:
        config_kwargs["user_id"] = user_id
    config = EvaluationRunConfig(**config_kwargs)
    asyncio.run(run_and_evaluate(config))
