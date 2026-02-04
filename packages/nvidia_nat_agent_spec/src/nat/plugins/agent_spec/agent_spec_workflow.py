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

import json
import logging
import yaml
from collections.abc import AsyncGenerator
from pathlib import Path
from types import NoneType
from typing import Any, Self

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import convert_to_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import FilePath
from pydantic import model_validator

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
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

    Converts an Agent-Spec YAML/JSON configuration to a LangGraph CompiledStateGraph
    and executes it as a NAT Function.

    Exactly one of spec_file, spec_yaml, or spec_json must be provided.

    Usage recommendations:
    - For NAT YAML workflow configs: Use spec_file (cleaner, more maintainable)
    - For programmatic/API use: Use spec_yaml or spec_json (dynamic configs, templating)

    Component reuse limitation:
    Agent-Spec embeds component configurations (e.g., LLMs) directly in the spec definition,
    which limits component reuse across agents. Unlike other NAT plugins that support automatic
    LLM resolution via llm_name, Agent-Spec requires explicit component IDs for component reuse.
    Use components_registry to manually map component IDs to NAT-managed components (e.g., LLMs)
    for sharing components across multiple Agent-Spec workflows.
    """

    model_config = ConfigDict(extra="forbid")

    description: str = Field(default="", description="Description of the Agent-Spec workflow")
    spec_file: FilePath | None = Field(
        default=None,
        description="Path to the Agent-Spec YAML/JSON configuration file. "
        "Recommended for NAT YAML workflow configurations."
    )
    spec_yaml: str | None = Field(
        default=None,
        description="Inline Agent-Spec YAML content as a string. "
        "Primarily intended for programmatic use (API-driven configs, dynamic generation). "
        "For NAT YAML workflow configs, prefer spec_file for better readability."
    )
    spec_json: str | None = Field(
        default=None,
        description="Inline Agent-Spec JSON content as a string. "
        "Primarily intended for programmatic use (API-driven configs, dynamic generation). "
        "For NAT YAML workflow configs, prefer spec_file for better readability."
    )
    tool_registry: dict[str, Any] | None = Field(
        default=None,
        description="Optional dictionary mapping tool names to LangGraph tools or callables. "
        "If both tool_registry and tool_names are provided, tools from tool_names will be "
        "added first, then tool_registry will override any name conflicts.",
    )
    tool_names: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list,
        description="Optional list of NAT tool names/groups to expose to the Agent-Spec runtime. "
        "Tools are resolved from NAT's tool registry using builder.get_tools(). "
        "Use FunctionRef('tool_name') for individual tools or FunctionGroupRef('group_name') for tool groups.",
    )
    components_registry: dict[str, Any] | None = Field(
        default=None,
        description="Optional dictionary mapping component IDs to LangGraph components (e.g., LLMs). "
        "Can be used to override components defined in the Agent-Spec YAML, such as providing "
        "a NAT LLM instead of the one specified in the YAML. "
        "Note: Unlike other NAT plugins that support llm_name for automatic LLM resolution, "
        "Agent-Spec requires explicit component IDs (from the 'id' field) for component reuse. "
        "For embedded Agent-Spec configs (where LLM configs are embedded in the YAML), "
        "manual mapping via components_registry is the recommended approach for component reuse. "
        "To use this, ensure your Agent-Spec YAML includes an 'id' field in the LLM config, "
        "then map that ID to your NAT LLM: components_registry={'llm_id': nat_llm}.",
    )
    checkpointer: Any | None = Field(
        default=None,
        description="Optional LangGraph checkpointer (e.g., MemorySaver). Required when using "
        "ClientTool in Agent-Spec YAML. If not provided and ClientTool is detected, "
        "MemorySaver will be used automatically.",
    )
    max_history: int = Field(
        default=15,
        description="Maximum number of messages to keep in conversation history.",
    )

    @model_validator(mode="after")
    def _validate_sources(self) -> Self:
        """Ensure exactly one of spec_file, spec_yaml, or spec_json is provided."""
        provided = [self.spec_file, self.spec_yaml, self.spec_json]
        cnt = sum(1 for v in provided if v is not None)
        if cnt != 1:
            raise ValueError(
                "Exactly one of spec_file, spec_yaml, or spec_json must be provided. "
                f"Found {cnt} provided: "
                f"spec_file={'provided' if self.spec_file else 'None'}, "
                f"spec_yaml={'provided' if self.spec_yaml else 'None'}, "
                f"spec_json={'provided' if self.spec_json else 'None'}"
            )
        return self


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
            from langchain_core.messages import trim_messages

            # Trim message history if max_history is configured
            state = value.model_dump()
            if self.config.max_history > 0:
                messages = trim_messages(
                    messages=[m.model_dump() for m in value.messages],
                    max_tokens=self.config.max_history,
                    strategy="last",
                    token_counter=len,
                    start_on="human",
                    include_system=True,
                )
                state["messages"] = messages

            # Check if the graph is an async context manager
            if hasattr(self._graph, "__aenter__") and hasattr(self._graph, "__aexit__"):
                logger.debug("Graph is an async context manager")
                async with self._graph as graph:
                    output = await graph.ainvoke(state)
            else:
                output = await self._graph.ainvoke(state)
            return AgentSpecWrapperOutput.model_validate(output)
        except Exception as e:
            raise RuntimeError(f"Error executing Agent-Spec workflow: {e}") from e

    async def _astream(self, value: AgentSpecWrapperInput) -> AsyncGenerator[AgentSpecWrapperOutput, None]:
        """Stream results from the Agent-Spec workflow."""
        try:
            from langchain_core.messages import trim_messages

            # Trim message history if max_history is configured
            state = value.model_dump()
            if self.config.max_history > 0:
                messages = trim_messages(
                    messages=[m.model_dump() for m in value.messages],
                    max_tokens=self.config.max_history,
                    strategy="last",
                    token_counter=len,
                    start_on="human",
                    include_system=True,
                )
                state["messages"] = messages

            if hasattr(self._graph, "__aenter__") and hasattr(self._graph, "__aexit__"):
                logger.debug("Graph is an async context manager")
                async with self._graph as graph:
                    async for output in graph.astream(state):
                        yield AgentSpecWrapperOutput.model_validate(output)
            else:
                async for output in self._graph.astream(state):
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
async def register(config: AgentSpecWrapperConfig, builder: Builder):
    """Register the Agent-Spec wrapper function.

    Loads Agent-Spec configuration and converts it to a LangGraph CompiledStateGraph
    using AgentSpecLoader, then wraps it as a NAT Function.
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

    # Determine the format and read the Agent-Spec content
    if config.spec_file:
        # Read from file
        spec_path = Path(config.spec_file)
        if not spec_path.exists():
            raise ValueError(f"Agent-Spec file '{spec_path}' does not exist.")

        with open(spec_path, "r", encoding="utf-8") as f:
            spec_content = f.read()

        # Determine format from file extension
        ext = spec_path.suffix.lower()
        spec_format = "json" if ext == ".json" else "yaml"
        source_description = f"file '{spec_path}'"
    elif config.spec_yaml:
        # Use inline YAML
        assert config.spec_yaml is not None  # Type narrowing: validator ensures this is set
        spec_content = config.spec_yaml
        spec_format = "yaml"
        source_description = "inline YAML"
    else:
        # Use inline JSON (config.spec_json)
        assert config.spec_json is not None  # Type narrowing: validator ensures this is set
        spec_content = config.spec_json
        spec_format = "json"
        source_description = "inline JSON"

    # Auto-detect checkpointer if ClientTool is used
    checkpointer = config.checkpointer
    if checkpointer is None:
        try:
            if spec_format == "json":
                spec_dict = json.loads(spec_content)
            else:
                spec_dict = yaml.safe_load(spec_content)

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
            logger.debug(f"Could not parse {spec_format} to detect ClientTool: {e}")

    # Build tool registry from tool_names and/or tool_registry
    tool_registry = {}
    if config.tool_names:
        try:
            tools = await builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
            if tools:
                tool_registry = {tool.name: tool for tool in tools}
        except Exception as e:
            logger.warning(f"Failed to resolve tools from tool_names: {e}")

    if config.tool_registry:
        tool_registry.update(config.tool_registry)

    loader = AgentSpecLoader(
        tool_registry=tool_registry,
        checkpointer=checkpointer,
    )
    try:
        if spec_format == "json":
            if config.components_registry:
                graph = loader.load_json(spec_content, components_registry=config.components_registry)
            else:
                graph = loader.load_json(spec_content)
        else:
            if config.components_registry:
                graph = loader.load_yaml(spec_content, components_registry=config.components_registry)
            else:
                graph = loader.load_yaml(spec_content)
    except Exception as e:
        raise RuntimeError(f"Failed to load Agent-Spec configuration from {source_description}: {e}") from e

    if not isinstance(graph, CompiledStateGraph):
        raise ValueError(
            f"Agent-Spec loader returned unexpected type: {type(graph)}. "
            "Expected CompiledStateGraph."
        )

    yield AgentSpecWrapperFunction(config=config, description=config.description, graph=graph)
