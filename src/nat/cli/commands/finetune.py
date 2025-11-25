# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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
"""CLI command for running finetuning."""

import logging
from pathlib import Path

import click

from nat.data_models.finetuning import FinetuneRunConfig
from nat.finetuning.finetuning_runtime import run_finetuning_sync

logger = logging.getLogger(__name__)


@click.command(name="finetune", help="Run finetuning on a workflow using collected trajectories.")
@click.option("--config_file",
              required=True,
              type=click.Path(exists=True, path_type=Path, resolve_path=True),
              help="Path to the configuration file containing finetuning settings")
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
@click.option("--override",
              "-o",
              multiple=True,
              type=(str, str),
              help="Override config values (e.g., -o finetuning.num_epochs 5)")
@click.option(
    "--validation_dataset",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="Validation dataset file path for periodic validation",
)
@click.option(
    "--validation_interval",
    type=int,
    default=5,
    help="Run validation every N epochs",
)
@click.option(
    "--validation_config_file",
    type=click.Path(exists=True, path_type=Path, resolve_path=True),
    required=False,
    help="Optional separate config file for validation runs",
)
@click.pass_context
def finetune_command(
    processors,  # pylint: disable=unused-argument
    *,
    config_file: Path,
    dataset: Path,
    result_json_path: str,
    endpoint: str,
    endpoint_timeout: int,
    override: tuple[tuple[str, str], ...],
    validation_dataset: Path,
    validation_interval: int,
    validation_config_file: Path,
):
    """
    Run finetuning based on the configuration file.

    This command will:
    1. Load the configuration with finetuning settings
    2. Initialize the finetuning runner
    3. Run evaluation to collect trajectories
    4. Submit trajectories for training
    5. Monitor training progress
    """
    logger.info("Starting finetuning with config: %s", config_file)

    # Apply overrides if provided
    if override:
        logger.info("Applying config overrides: %s", override)
        # TODO: Implement config override logic similar to other commands

    try:
        # Run the finetuning process
        run_finetuning_sync(
            FinetuneRunConfig(
                config_file=config_file,
                dataset=dataset,
                result_json_path=result_json_path,
                endpoint=endpoint,
                endpoint_timeout=endpoint_timeout,
                override=override,
                validation_dataset=validation_dataset,
                validation_interval=validation_interval,
                validation_config_file=validation_config_file,
            ))

        logger.info("Finetuning completed successfully")
    except Exception as e:
        logger.error("Finetuning failed: %s", e)
        raise click.ClickException(str(e))
