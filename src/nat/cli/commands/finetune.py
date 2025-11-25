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

from nat.finetuning.finetuning_runtime import run_finetuning_sync

logger = logging.getLogger(__name__)


@click.command(
    name="finetune",
    help="Run finetuning on a workflow using collected trajectories."
)
@click.option(
    "--config_file",
    required=True,
    type=click.Path(exists=True, path_type=Path, resolve_path=True),
    help="Path to the configuration file containing finetuning settings"
)
@click.option(
    "--override",
    "-o",
    multiple=True,
    type=(str, str),
    help="Override config values (e.g., -o finetuning.num_epochs 5)"
)
@click.pass_context
def finetune_command(
    ctx: click.Context,
    config_file: Path,
    override: tuple[tuple[str, str], ...]
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
        run_finetuning_sync(config_file)
        logger.info("Finetuning completed successfully")
    except Exception as e:
        logger.error("Finetuning failed: %s", e)
        raise click.ClickException(str(e))