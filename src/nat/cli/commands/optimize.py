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

from nat.data_models.optimizer import OptimizerRunConfig
from nat.profiler.parameter_optimization.optimizer_runtime import optimize_config

logger = logging.getLogger(__name__)


@click.group(name=__name__, invoke_without_command=True, help="Optimize a workflow with the specified dataset.")
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
@click.pass_context
def optimizer_command(ctx, **kwargs) -> None:
    """ Optimize workflow with the specified dataset"""
    pass


async def run_optimizer(config: OptimizerRunConfig):
    await optimize_config(config)


@optimizer_command.result_callback(replace=True)
def run_optimizer_callback(
    processors,  # pylint: disable=unused-argument
    *,
    config_file: Path,
    dataset: Path,
    result_json_path: str,
    endpoint: str,
    endpoint_timeout: int,
):
    """Run the optimizer with the provided config file and dataset."""
    config = OptimizerRunConfig(
        config_file=config_file,
        dataset=dataset,
        result_json_path=result_json_path,
        endpoint=endpoint,
        endpoint_timeout=endpoint_timeout,
    )

    asyncio.run(run_optimizer(config))
