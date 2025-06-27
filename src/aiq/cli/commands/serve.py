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

from pathlib import Path
import click
import os
import asyncio
import logging

logger = logging.getLogger(__name__)


@click.command(name="serve")
@click.option("--config_file",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
              required=True,
              help="A JSON/YAML file that sets the parameters for the workflow.")
@click.option("--override",
              nargs=2,
              multiple=True,
              type=str,
              help="Override config values using dot notation (e.g., --override llms.nim_llm.temperature 0.7)")
@click.option("--root_path", type=str, default="", help="The root path for the API")
@click.option("--host", type=str, default="localhost", help="Host to bind the server to")
@click.option("--port", type=int, default=8000, help="Port to bind the server to")
@click.option("--reload", type=bool, default=False, help="Enable auto-reload for development")
@click.option("--workers", type=int, default=1, help="Number of workers to run")
@click.option("--use_gunicorn", type=bool, default=False, help="Use Gunicorn to run the FastAPI app")
@click.pass_context
def serve_command(ctx: click.Context,
                  config_file: Path,
                  override: tuple[tuple[str, str], ...],
                  **kwargs) -> None:
    """
    Serve a FastAPI endpoint for the workflow based on the supplied configuration file.
    Each server runs with a single configuration on a specified port.
    
    Examples:
    aiq serve --config_file=friday_config.yml --port=8001
    aiq serve --config_file=oncall_config.yml --port=8000
    """
    
    logger.info("Starting server with config file: '%s'", config_file)
    logger.info("Server will run on port: %d", kwargs.get('port', 8000))
    
    # Import and run the FastAPI functionality directly
    from aiq.runtime.loader import PluginTypes, discover_and_register_plugins, load_config
    from aiq.cli.cli_utils.config_override import load_and_override_config
    from aiq.utils.data_models.schema_validator import validate_schema
    from aiq.data_models.config import AIQConfig
    from aiq.cli.type_registry import GlobalTypeRegistry
    from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
    
    # Ensure all objects are loaded
    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)
    discover_and_register_plugins(PluginTypes.FRONT_END)
    
    # Load and validate configuration - use first available config file for frontend setup
    config = load_and_override_config(config_file, override)
    config = validate_schema(config, AIQConfig)
    
    logger.info("Configuration loaded successfully")
    
    # Get the FastAPI front end
    front_end_plugin = config.general.front_end
    
    if not isinstance(front_end_plugin, FastApiFrontEndConfig):
        raise click.ClickException("The specified configuration does not use the FastAPI front end")
    
    # Override the FastAPI configuration with command line parameters
    front_end_plugin.host = kwargs.get('host', front_end_plugin.host)
    front_end_plugin.port = kwargs.get('port', front_end_plugin.port)
    front_end_plugin.reload = kwargs.get('reload', front_end_plugin.reload)
    front_end_plugin.workers = kwargs.get('workers', front_end_plugin.workers)
    front_end_plugin.use_gunicorn = kwargs.get('use_gunicorn', front_end_plugin.use_gunicorn)
    front_end_plugin.root_path = kwargs.get('root_path', front_end_plugin.root_path)
    
    logger.info("FastAPI configuration updated - Host: %s, Port: %d", front_end_plugin.host, front_end_plugin.port)
    
    # Create and run the FastAPI plugin
    from aiq.front_ends.fastapi.fastapi_front_end_plugin import FastApiFrontEndPlugin
    
    front_end_plugin_instance = FastApiFrontEndPlugin(config)
    
    async def run_plugin():
        await front_end_plugin_instance.run()
    
    logger.info("Starting FastAPI server...")
    return asyncio.run(run_plugin()) 