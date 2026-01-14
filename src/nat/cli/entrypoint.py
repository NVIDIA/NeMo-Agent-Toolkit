# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import sys
import time

import click
import nest_asyncio2
from dotenv import load_dotenv

from nat.utils.log_levels import LOG_LEVELS
from nat.utils.log_utils import setup_logging as log_utils_setup_logging

from .plugin_loader import discover_and_load_cli_plugins

# Load environment variables from .env file, if it exists
load_dotenv()

# Apply at the beginning of the file to avoid issues with asyncio
nest_asyncio2.apply()


def setup_logging(log_level: str):
    """Configure logging with the specified level"""
    numeric_level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
    log_utils_setup_logging(numeric_level)
    return numeric_level


def get_version():
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version
    try:
        # Use the distro name to get the version
        return version("nvidia-nat")
    except PackageNotFoundError:
        return "unknown"


@click.group(name="nat", chain=False, invoke_without_command=True, no_args_is_help=True)
@click.version_option(version=get_version())
@click.option('--log-level',
              type=click.Choice(LOG_LEVELS.keys(), case_sensitive=False),
              default='INFO',
              help='Set the logging level')
@click.pass_context
def cli(ctx: click.Context, log_level: str):
    """Main entrypoint for the NAT CLI"""

    ctx_dict = ctx.ensure_object(dict)

    # Setup logging
    numeric_level = setup_logging(log_level)

    nat_logger = logging.getLogger("nat")
    nat_logger.setLevel(numeric_level)

    logger = logging.getLogger(__package__)

    # Set the parent logger for all of the llm examples to use morpheus so we can take advantage of configure_logging
    logger.parent = nat_logger
    logger.setLevel(numeric_level)

    ctx_dict["start_time"] = time.time()
    ctx_dict["log_level"] = log_level


# Discover and load ALL CLI commands (core + plugins) via entry points
discover_and_load_cli_plugins(cli)

# Aliases - need to get start_command from the loaded commands
start_cmd = cli.commands.get("start")
if start_cmd and hasattr(start_cmd, "get_command"):
    cli.add_command(start_cmd.get_command(None, "console"), name="run")  # type: ignore
    cli.add_command(start_cmd.get_command(None, "fastapi"), name="serve")  # type: ignore


@cli.result_callback()
@click.pass_context
def after_pipeline(ctx: click.Context, pipeline_start_time: float, *_, **__):
    logger = logging.getLogger(__name__)

    end_time = time.time()

    ctx_dict = ctx.ensure_object(dict)

    start_time = ctx_dict["start_time"]

    # Reset the terminal colors, not using print to avoid an additional newline
    for stream in (sys.stdout, sys.stderr):
        stream.write("\x1b[0m")

    logger.debug("Total time: %.2f sec", end_time - start_time)

    if (pipeline_start_time is not None):
        logger.debug("Pipeline runtime: %.2f sec", end_time - pipeline_start_time)
