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

from __future__ import annotations

import importlib.metadata
import logging
import time
from contextlib import asynccontextmanager
from enum import IntFlag
from enum import auto
from functools import reduce

from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.cli.type_registry import GlobalTypeRegistry
from aiq.data_models.config import AIQConfig
from aiq.runtime.session import AIQSessionManager
from aiq.utils.data_models.schema_validator import validate_schema
from aiq.utils.debugging_utils import is_debugger_attached
from aiq.utils.io.yaml_tools import yaml_load
from aiq.utils.type_utils import StrPath

logger = logging.getLogger(__name__)


class PluginTypes(IntFlag):
    COMPONENT = auto()
    """
    A plugin that is a component of the workflow. This includes tools, LLMs, retrievers, etc.
    """
    FRONT_END = auto()
    """
    A plugin that is a front end for the workflow. This includes FastAPI, Gradio, etc.
    """
    EVALUATOR = auto()
    """
    A plugin that is an evaluator for the workflow. This includes evaluators like RAGAS, SWE-bench, etc.
    """
    REGISTRY_HANDLER = auto()

    # Convenience flag for groups of plugin types
    CONFIG_OBJECT = COMPONENT | FRONT_END | EVALUATOR
    """
    Any plugin that can be specified in the AIQ Toolkit configuration file.
    """
    ALL = COMPONENT | FRONT_END | EVALUATOR | REGISTRY_HANDLER
    """
    All plugin types
    """


def load_config(config_file: StrPath) -> AIQConfig:
    """
    This is the primary entry point for loading an AIQ Toolkit configuration file. It ensures that all plugins are
    loaded and then validates the configuration file against the AIQConfig schema.

    Parameters
    ----------
    config_file : StrPath
        The path to the configuration file

    Returns
    -------
    AIQConfig
        The validated AIQConfig object
    """

    # Ensure all of the plugins are loaded
    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

    config_yaml = yaml_load(config_file)

    # Validate configuration adheres to AIQ Toolkit schemas
    validated_aiq_config = validate_schema(config_yaml, AIQConfig)

    return validated_aiq_config


@asynccontextmanager
async def load_workflow(config_file: StrPath, max_concurrency: int = -1):
    """
    Load the AIQ Toolkit configuration file and create an AIQRunner object. This is the primary entry point for running
    AIQ Toolkit workflows.

    Parameters
    ----------
    config_file : StrPath
        The path to the configuration file
    max_concurrency : int, optional
        The maximum number of parallel workflow invocations to support. Specifying 0 or -1 will allow an unlimited
        count, by default -1
    """

    # Load the config object
    config = load_config(config_file)

    # Must yield the workflow function otherwise it cleans up
    async with WorkflowBuilder.from_config(config=config) as workflow:

        yield AIQSessionManager(workflow.build(), max_concurrency=max_concurrency)


def discover_entrypoints(plugin_type: PluginTypes):
    """
    Discover all the requested plugin types which were registered via an entry point group and return them.
    """

    entry_points = importlib.metadata.entry_points()

    plugin_groups = []

    # Add the specified plugin type to the list of groups to load
    if (plugin_type & PluginTypes.COMPONENT):
        plugin_groups.extend(["aiq.plugins", "aiq.components"])
    if (plugin_type & PluginTypes.FRONT_END):
        plugin_groups.append("aiq.front_ends")
    if (plugin_type & PluginTypes.REGISTRY_HANDLER):
        plugin_groups.append("aiq.registry_handlers")
    if (plugin_type & PluginTypes.EVALUATOR):
        plugin_groups.append("aiq.evaluators")

    # Get the entry points for the specified groups
    aiq_plugins = reduce(lambda x, y: x + y, [entry_points.select(group=y) for y in plugin_groups])

    return aiq_plugins


def discover_and_register_plugins(plugin_type: PluginTypes):
    """
    Discover all the requested plugin types which were registered via an entry point group and register them into the
    GlobalTypeRegistry.
    """

    # Get the entry points for the specified groups
    aiq_plugins = discover_entrypoints(plugin_type)

    count = 0

    # Pause registration hooks for performance. This is useful when loading a large number of plugins.
    with GlobalTypeRegistry.get().pause_registration_changed_hooks():

        for entry_point in aiq_plugins:
            try:
                logger.debug("Loading module '%s' from entry point '%s'...", entry_point.module, entry_point.name)

                start_time = time.time()

                entry_point.load()

                elapsed_time = (time.time() - start_time) * 1000

                logger.debug("Loading module '%s' from entry point '%s'...Complete (%f ms)",
                             entry_point.module,
                             entry_point.name,
                             elapsed_time)

                # Log a warning if the plugin took a long time to load. This can be useful for debugging slow imports.
                # The threshold is 300 ms if no plugins have been loaded yet, and 100 ms otherwise. Triple the threshold
                # if a debugger is attached.
                if (elapsed_time > (300.0 if count == 0 else 100.0) * (3 if is_debugger_attached() else 1)):
                    logger.warning(
                        "Loading module '%s' from entry point '%s' took a long time (%f ms). "
                        "Ensure all imports are inside your registered functions.",
                        entry_point.module,
                        entry_point.name,
                        elapsed_time)

            except ImportError:
                logger.warning("Failed to import plugin '%s'", entry_point.name, exc_info=True)
                # Optionally, you can mark the plugin as unavailable or take other actions

            except Exception:
                logger.exception("An error occurred while loading plugin '%s': {e}", entry_point.name, exc_info=True)

            finally:
                count += 1
