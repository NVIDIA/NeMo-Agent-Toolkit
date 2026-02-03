# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Red teaming CLI command."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.group(name=__name__, invoke_without_command=True, help="Run red teaming evaluation with multiple scenarios.")
@click.option(
    "--red_team_config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="A YAML/JSON file containing red teaming configuration (evaluator, scenarios, etc.).",
)
@click.option(
    "--config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="A JSON/YAML file that sets the parameters for the base workflow. "
    "Overrides base_workflow in red_team_config if both are provided.",
)
@click.option(
    "--dataset",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="A JSON file with questions and ground truth answers. "
    "Overrides the dataset path in the config file.",
)
@click.option(
    "--result_json_path",
    type=str,
    default="$",
    help="A JSON path to extract the result from the workflow. "
    "For example, '$.output' extracts the 'output' field.",
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
    help="Override config values for the base workflow config using dot notation "
    "(e.g., --override llms.nim_llm.temperature 0.7)",
)
@click.pass_context
def red_team_command(ctx, **kwargs) -> None:
    """Run red teaming evaluation with multiple scenarios."""
    pass


@red_team_command.result_callback(replace=True)
def process_red_team_eval(
    processors,
    *,
    red_team_config: Path | None,
    config_file: Path | None,
    dataset: Path | None,
    result_json_path: str,
    endpoint: str | None,
    endpoint_timeout: int,
    reps: int,
    override: tuple[tuple[str, str], ...],
):
    """Process the red team eval command and execute the evaluation."""
    from nat.eval.runners.red_teaming_runner import RedTeamingRunner
    from nat.runtime.loader import load_config

    from .red_teaming_utils import load_red_teaming_config

    # Must have at least one of these
    if red_team_config is None and config_file is None:
        raise click.ClickException("Either --red_team_config or --config_file must be provided.")

    # Load configs
    rt_config = None
    if red_team_config is not None:
        rt_config = load_red_teaming_config(red_team_config)
        base_workflow_path = config_file or rt_config.base_workflow
        if base_workflow_path is None:
            raise click.ClickException(
                "No base workflow specified. Set 'base_workflow' in red_team_config or provide --config_file.")
        base_workflow_config = load_config(base_workflow_path)
    else:
        assert config_file is not None
        base_workflow_config = load_config(config_file)

    # Create and run the runner
    runner = RedTeamingRunner(
        config=rt_config,
        base_workflow_config=base_workflow_config,
        dataset_path=str(dataset) if dataset else None,
        result_json_path=result_json_path,
        endpoint=endpoint,
        endpoint_timeout=endpoint_timeout,
        reps=reps,
        overrides=override,
    )

    try:
        _ = asyncio.run(runner.run())
    except ValueError as e:
        raise click.ClickException(str(e)) from e
