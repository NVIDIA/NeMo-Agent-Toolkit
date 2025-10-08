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

# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nest_asyncio
from dotenv import load_dotenv

from nat.utils.log_levels import LOG_LEVELS

from .commands.configure.configure import configure_command
from .commands.evaluate import eval_command
from .commands.info.info import info_command
from .commands.mcp.mcp import mcp_command
from .commands.object_store.object_store import object_store_command
from .commands.optimize import optimizer_command
from .commands.registry.registry import registry_command
from .commands.sizing.sizing import sizing
from .commands.start import start_command
from .commands.uninstall import uninstall_command
from .commands.validate import validate_command
from .commands.workflow.workflow import workflow_command

# Load environment variables from .env file, if it exists
load_dotenv()

# Apply at the beginning of the file to avoid issues with asyncio
nest_asyncio.apply()


def setup_logging(log_level: str):
    """Configure logging with the specified level"""
    numeric_level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)-8s - %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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


cli.add_command(configure_command, name="configure")
cli.add_command(eval_command, name="eval")
cli.add_command(info_command, name="info")
cli.add_command(registry_command, name="registry")
cli.add_command(start_command, name="start")
cli.add_command(uninstall_command, name="uninstall")
cli.add_command(validate_command, name="validate")
cli.add_command(workflow_command, name="workflow")
cli.add_command(sizing, name="sizing")
cli.add_command(optimizer_command, name="optimize")
cli.add_command(object_store_command, name="object-store")
cli.add_command(mcp_command, name="mcp")

# Aliases
cli.add_command(start_command.get_command(None, "console"), name="run")  # type: ignore
cli.add_command(start_command.get_command(None, "fastapi"), name="serve")  # type: ignore


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
