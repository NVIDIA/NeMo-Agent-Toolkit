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

"""Agent-Spec workflow wrapper for NAT.

This module provides integration between Oracle Agent-Spec and NeMo Agent Toolkit
by converting Agent-Spec YAML configurations into executable NAT Functions.

Flow:
1. User defines Agent-Spec YAML (standardized agent/flow specification)
2. NAT workflow config references agent-spec with _type: agent_spec_wrapper
3. NAT plugin system discovers and loads agent-spec integration (via entry points)
4. AgentSpecLoader converts Agent-Spec YAML → LangGraph CompiledStateGraph
5. AgentSpecWrapperFunction wraps the graph (following LanggraphWrapperFunction pattern)
6. NAT runtime executes it with evaluation, profiling, observability, middleware, etc.
"""

import logging
import yaml
from collections.abc import AsyncGenerator
from pathlib import Path
from types import NoneType
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import convert_to_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import FilePath

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class AgentSpecWrapperInput(BaseModel):
    """Input model for the Agent-Spec wrapper."""

    model_config = ConfigDict(extra="allow")

    messages: list[BaseMessage]


class AgentSpecWrapperOutput(BaseModel):
    """Output model for the Agent-Spec wrapper."""

    model_config = ConfigDict(extra="allow")

    messages: list[BaseMessage]


class AgentSpecWrapperConfig(FunctionBaseConfig, name="agent_spec_wrapper"):
    """Configuration model for the Agent-Spec wrapper.

    This config allows users to reference an Agent-Spec YAML file that will be
    converted to a LangGraph CompiledStateGraph and executed as a NAT Function.

    Example usage in NAT workflow config:
        workflow:
          _type: agent_spec_wrapper
          spec_file: "my_agent_spec.yaml"
          description: "Oracle agent-spec agent"

    Attributes:
        spec_file: Path to the Agent-Spec YAML configuration file (required).
        description: Optional description of the workflow.
        tool_registry: Optional dictionary mapping tool names to LangGraph tools
            or callables. If provided, these tools will be available to the
            Agent-Spec workflow.
    """

    model_config = ConfigDict(extra="forbid")

    description: str = Field(default="", description="Description of the Agent-Spec workflow")
    spec_file: FilePath = Field(description="Path to the Agent-Spec YAML configuration file")
    tool_registry: dict[str, Any] | None = Field(
        default=None,
        description="Optional dictionary mapping tool names to LangGraph tools or callables",
    )
    components_registry: dict[str, Any] | None = Field(
        default=None,
        description="Optional dictionary mapping component IDs to LangGraph components (e.g., LLMs). "
        "Can be used to override components defined in the Agent-Spec YAML, such as providing "
        "a NAT LLM instead of the one specified in the YAML.",
    )
    checkpointer: Any | None = Field(
        default=None,
        description="Optional LangGraph checkpointer (e.g., MemorySaver). Required when using "
        "ClientTool in Agent-Spec YAML. If not provided and ClientTool is detected, "
        "MemorySaver will be used automatically.",
    )


class AgentSpecWrapperFunction(Function[AgentSpecWrapperInput, NoneType, AgentSpecWrapperOutput]):
    """Function wrapper for Agent-Spec workflows.

    This function wraps a LangGraph CompiledStateGraph that was created from an
    Agent-Spec YAML configuration. It follows the same pattern as LanggraphWrapperFunction
    to handle input/output conversion and delegate execution to the underlying graph.
    """

    def __init__(
        self,
        *,
        config: AgentSpecWrapperConfig,
        description: str | None = None,
        graph: CompiledStateGraph,
    ):
        """Initialize the Agent-Spec wrapper function.

        Args:
            config: The configuration for the Agent-Spec wrapper.
            description: The description of the Agent-Spec wrapper.
            graph: The compiled LangGraph state graph from Agent-Spec.
        """
        super().__init__(config=config, description=description, converters=[AgentSpecWrapperFunction.convert_to_str])
        self._graph = graph

    def _convert_input(self, value: Any) -> AgentSpecWrapperInput:
        """Convert input to AgentSpecWrapperInput format."""
        # If the value is not a list, wrap it in a list to be compatible with the graph input
        if not isinstance(value, list):
            value = [value]

        # Convert the value to message format using LangChain utils
        messages = convert_to_messages(value)
        return AgentSpecWrapperInput(messages=messages)

    async def _ainvoke(self, value: AgentSpecWrapperInput) -> AgentSpecWrapperOutput:
        """Invoke the Agent-Spec workflow."""
        try:
            # Check if the graph is an async context manager
            if hasattr(self._graph, "__aenter__") and hasattr(self._graph, "__aexit__"):
                logger.debug("Graph is an async context manager")
                async with self._graph as graph:
                    output = await graph.ainvoke(value.model_dump())
            else:
                output = await self._graph.ainvoke(value.model_dump())
            return AgentSpecWrapperOutput.model_validate(output)
        except Exception as e:
            raise RuntimeError(f"Error executing Agent-Spec workflow: {e}") from e

    async def _astream(self, value: AgentSpecWrapperInput) -> AsyncGenerator[AgentSpecWrapperOutput, None]:
        """Stream results from the Agent-Spec workflow."""
        try:
            if hasattr(self._graph, "__aenter__") and hasattr(self._graph, "__aexit__"):
                logger.debug("Graph is an async context manager")
                async with self._graph as graph:
                    async for output in graph.astream(value.model_dump()):
                        yield AgentSpecWrapperOutput.model_validate(output)
            else:
                async for output in self._graph.astream(value.model_dump()):
                    yield AgentSpecWrapperOutput.model_validate(output)
        except Exception as e:
            raise RuntimeError(f"Error streaming Agent-Spec workflow: {e}") from e

    @staticmethod
    def convert_to_str(value: AgentSpecWrapperOutput) -> str:
        """Convert the output to a string."""
        if not value.messages:
            return ""
        return value.messages[-1].text


@register_function(config_type=AgentSpecWrapperConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def register(config: AgentSpecWrapperConfig, b: Builder):
    """Register the Agent-Spec wrapper function.

    This function:
    1. Loads the Agent-Spec YAML file
    2. Converts it to a LangGraph CompiledStateGraph using AgentSpecLoader
    3. Wraps it as a NAT Function using AgentSpecWrapperFunction

    The wrapped function can then be used in NAT workflows and will benefit from
    all NAT features: evaluation, profiling, observability, middleware, etc.

    Args:
        config: The Agent-Spec wrapper configuration.
        b: The NAT Builder instance (unused for now, but required by the decorator).

    Yields:
        AgentSpecWrapperFunction: The wrapped function ready for use in NAT workflows.

    Raises:
        ImportError: If pyagentspec adapters module is not available. Install with:
            uv pip install "pyagentspec[langgraph,langgraph_mcp]>=26.1.0"
    """
    try:
        from pyagentspec.adapters.langgraph import AgentSpecLoader
    except ImportError as e:
        raise ImportError(
            "Failed to import pyagentspec.adapters.langgraph. "
            "Install pyagentspec with langgraph extras:\n"
            "  uv pip install \"pyagentspec[langgraph,langgraph_mcp]>=26.1.0\"\n\n"
            "Note: The root pyproject.toml includes override-dependencies to resolve "
            "langchain-core version conflicts between pyagentspec and nvidia-nat-langchain."
        ) from e

    # Read the Agent-Spec YAML file
    spec_path = Path(config.spec_file)
    if not spec_path.exists():
        raise ValueError(f"Agent-Spec file '{spec_path}' does not exist.")

    with open(spec_path, "r", encoding="utf-8") as f:
        spec_yaml = f.read()

    # Determine if checkpointer is needed (ClientTool requires it)
    # If config provides one, use it; otherwise auto-detect and use MemorySaver if needed
    checkpointer = config.checkpointer
    if checkpointer is None:
        # Check if YAML contains ClientTool - if so, we need a checkpointer
        import yaml
        try:
            spec_dict = yaml.safe_load(spec_yaml)
            # Check if tools section contains ClientTool
            if isinstance(spec_dict, dict):
                tools = spec_dict.get("tools", [])
                has_client_tool = any(
                    isinstance(tool, dict) and tool.get("component_type") == "ClientTool"
                    for tool in tools
                )
                if has_client_tool:
                    from langgraph.checkpoint.memory import MemorySaver
                    checkpointer = MemorySaver()
                    logger.debug("Auto-detected ClientTool, using MemorySaver checkpointer")
        except Exception as e:
            logger.debug(f"Could not parse YAML to detect ClientTool: {e}")

    # Create AgentSpecLoader with optional tool registry and checkpointer
    loader = AgentSpecLoader(
        tool_registry=config.tool_registry or {},
        checkpointer=checkpointer,
    )

    # Load the Agent-Spec YAML and convert to LangGraph CompiledStateGraph
    # If components_registry is provided, use it to override components (e.g., LLMs)
    try:
        if config.components_registry:
            graph = loader.load_yaml(spec_yaml, components_registry=config.components_registry)
        else:
            graph = loader.load_yaml(spec_yaml)
    except Exception as e:
        raise RuntimeError(f"Failed to load Agent-Spec configuration from '{spec_path}': {e}") from e

    # Validate that we got a CompiledStateGraph
    if not isinstance(graph, CompiledStateGraph):
        raise ValueError(
            f"Agent-Spec loader returned unexpected type: {type(graph)}. "
            "Expected CompiledStateGraph."
        )

    # Wrap the graph as a NAT Function (following LanggraphWrapperFunction pattern)
    yield AgentSpecWrapperFunction(config=config, description=config.description, graph=graph)
