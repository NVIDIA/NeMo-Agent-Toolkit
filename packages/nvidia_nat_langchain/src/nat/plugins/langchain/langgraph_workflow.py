# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib.util
import logging
import os
import sys
import uuid
from collections.abc import AsyncGenerator
from collections.abc import Callable
from pathlib import Path
from types import NoneType
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.messages import MessageLikeRepresentation
from langchain_core.messages.utils import convert_to_messages
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.state import StateGraph
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import DirectoryPath
from pydantic import Field
from pydantic import FilePath

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

GraphDefType = Callable[[RunnableConfig], CompiledStateGraph | StateGraph] | CompiledStateGraph

logger = logging.getLogger(__name__)


class LanggraphWrapperInput(BaseModel):
    """Input model for the LangGraph wrapper."""

    model_config = ConfigDict(extra="allow")

    messages: list[MessageLikeRepresentation] | PromptValue


class LanggraphWrapperOutput(BaseModel):
    """Output model for the LangGraph wrapper."""

    model_config = ConfigDict(extra="allow")

    messages: list[BaseMessage]


class LanggraphWrapperConfig(FunctionBaseConfig, name="langgraph_wrapper"):
    """Configuration model for the LangGraph wrapper."""

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    dependencies: list[DirectoryPath] = Field(default_factory=list)
    graph: str
    env: FilePath | dict[str, str] | None = None


class LanggraphWrapperFunction(Function[LanggraphWrapperInput, NoneType, LanggraphWrapperOutput]):
    """Function for the LangGraph wrapper."""

    def __init__(self, *, config: LanggraphWrapperConfig, description: str | None = None, graph: CompiledStateGraph):
        """Initialize the LangGraph wrapper function.

        Args:
            config: The configuration for the LangGraph wrapper.
            description: The description of the LangGraph wrapper.
            graph: The graph to wrap.
        """

        super().__init__(config=config, description=description, converters=[LanggraphWrapperFunction.convert_to_str])

        self._graph = graph

    def _convert_input(self, value: Any) -> LanggraphWrapperInput:

        # If the value is not a list, wrap it in a list to be compatible with the graph input and use the normal
        # conversion logic
        if (not isinstance(value, list)):
            value = [value]

        # Convert the value to message format using LangChain utils. Ensures is compatible with the message format
        messages = convert_to_messages(value)

        return LanggraphWrapperInput(messages=messages)

    async def _ainvoke(self, value: LanggraphWrapperInput) -> LanggraphWrapperOutput:

        try:
            # Check if the graph is an async context manager (e.g., from @asynccontextmanager)
            if hasattr(self._graph, '__aenter__') and hasattr(self._graph, '__aexit__'):
                logger.info("Graph is an async context manager")
                async with self._graph as graph:
                    output = await graph.ainvoke(value.model_dump())
            else:
                output = await self._graph.ainvoke(value.model_dump())
            return LanggraphWrapperOutput.model_validate(output)
        except Exception as e:
            raise RuntimeError(f"Error in LangGraph workflow: {e}") from e

    async def _astream(self, value: LanggraphWrapperInput) -> AsyncGenerator[LanggraphWrapperOutput, None]:
        try:

            if hasattr(self._graph, '__aenter__') and hasattr(self._graph, '__aexit__'):
                logger.info("Graph is an async context manager")
                async with self._graph as graph:
                    async for output in graph.astream(value.model_dump()):
                        yield LanggraphWrapperOutput.model_validate(output)
            else:
                async for output in self._graph.astream(value.model_dump()):
                    yield LanggraphWrapperOutput.model_validate(output)
        except Exception as e:
            raise RuntimeError(f"Error in LangGraph workflow: {e}") from e

    @staticmethod
    def convert_to_str(value: LanggraphWrapperOutput) -> str:
        """Convert the output to a string."""
        if not value.messages:
            return ""

        return value.messages[-1].text


@register_function(config_type=LanggraphWrapperConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def register(config: LanggraphWrapperConfig, b: Builder):

    # Process the dependencies. This is a list of either paths or names of packages to add to the env. For now, we only
    # support paths.
    added_paths = []
    try:
        for dependency in config.dependencies:
            if os.path.exists(dependency) and os.path.isdir(dependency):
                # Add the dependency to the environment
                sys.path.append(dependency)
                added_paths.append(dependency)
            else:
                raise ValueError(f"Dependency '{dependency}' (from langgraph_wrapper.dependencies) is not a "
                                 "valid directory. At the moment, we only support directories. Packages "
                                 "need to be installed in the environment before they can be used.")

        # Process the env. This is a path to a .env file to load into the environment or a list of environment variables
        # to set.
        if config.env is not None:
            if isinstance(config.env, Path):
                if os.path.exists(config.env) and os.path.isfile(config.env):
                    load_dotenv(config.env, override=True)
                else:
                    raise ValueError(
                        f"Env '{config.env}' is not a valid file. At the moment, we only support .env files.")
            elif isinstance(config.env, dict):
                for key, value in config.env.items():
                    os.environ[key] = value
            else:
                raise ValueError(
                    f"Env '{config.env}' is not a valid type. At the moment, we only support strings and dictionaries.")

        # Now process the graph.
        # Check that config.graph contains exactly one colon
        if config.graph.count(":") != 1:
            raise ValueError(
                f"Graph definition path '{config.graph}' must contain exactly one colon to split module and name "
                f"(e.g., '/path/to/module.py:graph_name'). Found {config.graph.count(':')}.")

        # Split the graph path into module and name
        module_path, name = config.graph.rsplit(":", 1)

        unique_module_name = f"langgraph_workflow_{uuid.uuid4().hex[:8]}"

        spec = importlib.util.spec_from_file_location(unique_module_name, module_path)

        if spec is None:
            raise ValueError(f"Spec not found for module: {module_path}")

        module = importlib.util.module_from_spec(spec)

        if module is None:
            raise ValueError(f"Module not found for module: {module_path}")

        sys.modules[unique_module_name] = module

        if spec.loader is not None:
            spec.loader.exec_module(module)
        else:
            raise ValueError(f"Loader not found for module: {module_path}")

        graph_def: GraphDefType = getattr(module, name)

        if isinstance(graph_def, CompiledStateGraph):
            graph = graph_def
        elif callable(graph_def):
            graph = graph_def(RunnableConfig())

            if isinstance(graph, StateGraph):
                graph = graph.compile()
        else:
            raise ValueError(
                f"Graph definition {name} is not a valid graph definition. It must be a CompiledStateGraph or a "
                f"callable that returns a CompiledStateGraph. Got {type(graph_def)}.")

        yield LanggraphWrapperFunction(config=config, description=config.description, graph=graph)
    finally:
        # Remove only the paths we've added to sys.path to restore sys.path to its original state
        for dependency in added_paths:
            if dependency in sys.path:
                sys.path.remove(dependency)
