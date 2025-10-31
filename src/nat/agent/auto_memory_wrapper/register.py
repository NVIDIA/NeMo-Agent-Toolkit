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

import logging
from typing import Any

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_function
from nat.data_models.agent import AgentBaseConfig
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import MemoryRef

logger = logging.getLogger(__name__)


class AutoMemoryAgentConfig(AgentBaseConfig, name="auto_memory_agent"):
    """
    Wraps any NAT agent to provide automatic memory capture and retrieval.

    This agent automatically captures user messages, retrieves relevant context,
    and stores agent responses without requiring the LLM to invoke memory tools.

    **Use this when:**
    - You want guaranteed memory capture (not dependent on LLM tool calling)
    - You need consistent memory operations across all interactions
    - Your memory backend (Zep, Mem0) is designed for automatic memory management

    **Use tool-based memory when:**
    - You want the LLM to decide when to access memory
    - Memory operations should be selective based on context

    **Requirements:**
    - Inner agent MUST have `use_openai_api: true` to properly receive memory context
    - This is necessary to pass multiple messages (including system messages) to the agent

    **Example:**

    .. code-block:: yaml

        functions:
          my_react_agent:
            _type: react_agent
            llm_name: nim_llm
            tool_names: [calculator, web_search]
            use_openai_api: true  # REQUIRED for auto_memory_agent

        memory:
          zep_memory:
            _type: nat.plugins.zep_cloud/zep_memory

        workflow:
          _type: auto_memory_agent
          inner_agent_name: my_react_agent
          memory_name: zep_memory
          llm_name: nim_llm
          verbose: true

    **Multi-tenant User Isolation:**

    User ID is automatically extracted from runtime context (user_manager.get_id()) for proper
    multi-tenant memory isolation. Set user_manager via SessionManager.session() in production.
    Defaults to "default_user" for testing/development. See README.md for deployment examples.
    """

    # Memory configuration
    memory_name: MemoryRef = Field(..., description="Name of the memory backend (from memory section of config)")

    # Reference to inner agent by NAME (not inline config)
    inner_agent_name: FunctionRef = Field(..., description="Name of the agent workflow to wrap with automatic memory")

    # Feature flags
    save_user_messages_to_memory: bool = Field(
        default=True, description="Automatically save user messages to memory before agent processing")
    retrieve_memory_for_every_response: bool = Field(
        default=True,
        description=("Automatically retrieve memory context before agent processing. "
                     "Set to false for save-only mode or when using tool-based retrieval."))
    save_ai_messages_to_memory: bool = Field(
        default=True, description="Automatically save AI agent responses to memory after generation")

    # Memory retrieval configuration
    search_params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Backend-specific search parameters passed to memory_editor.search().\n"
            "Common parameters:\n"
            "  - top_k (int): Maximum results to return (default: 5)\n"
            "  - mode (str): For Zep, 'basic' (fast) or 'summary' (comprehensive)\n\n"
            "Additional parameters:\n"
            "  - Any additional parameters that the chosen memory backend supports in its search function\n\n"))

    # Memory addition configuration
    add_params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Backend-specific parameters passed to memory_editor.add_items().\n"
            "For Zep:\n"
            "  - ignore_roles (list[str]): Role types to exclude from graph memory (e.g., ['assistant'])\n"
            "    Available roles: norole, system, assistant, user, function, tool\n\n"
            "Additional parameters:\n"
            "  - Any additional parameters that the chosen memory backend supports in its add_items function\n\n"))


@register_function(config_type=AutoMemoryAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def auto_memory_agent(config: AutoMemoryAgentConfig, builder: Builder):
    """
    Build the auto-memory agent that wraps another agent.

    The inner agent is retrieved as a Function that receives a ChatRequest with
    multiple messages (including system messages with memory context). It manages
    its own internal state (ReActGraphState, etc.) and the wrapper never manipulates
    that state.
    """
    from langchain_core.messages.human import HumanMessage
    from langgraph.graph.state import CompiledStateGraph

    from nat.agent.auto_memory_wrapper.agent import AutoMemoryWrapperGraph
    from nat.agent.auto_memory_wrapper.state import AutoMemoryWrapperState
    from nat.agent.base import AGENT_LOG_PREFIX
    from nat.builder.function_info import FunctionInfo

    # Get memory editor from builder
    memory_editor = await builder.get_memory_client(config.memory_name)

    # Get inner agent as a Function (not a dict config)
    # This gives us a function that accepts ChatRequest with multiple messages
    inner_agent_fn = await builder.get_function(config.inner_agent_name)

    # Validate that inner agent has use_openai_api = True
    # This is required to pass multiple messages (including system messages with memory context)
    inner_agent_config = builder.get_function_config(config.inner_agent_name)
    if hasattr(inner_agent_config, 'use_openai_api') and not inner_agent_config.use_openai_api:
        raise ValueError(f"Auto-memory wrapper requires inner agent '{config.inner_agent_name}' "
                         f"to have 'use_openai_api: true'. This is necessary to pass multiple "
                         f"messages (including system messages with memory context) to the inner agent. "
                         f"Please add 'use_openai_api: true' to your '{config.inner_agent_name}' configuration.")

    # Calculate recursion_limit based on inner agent's configuration
    # This ensures the wrapper is transparent - users only configure the inner agent's limits
    # and the wrapper automatically accounts for its own overhead
    if hasattr(inner_agent_config, 'max_tool_calls'):
        # ReAct agent and similar agents use max_tool_calls
        inner_max_calls = inner_agent_config.max_tool_calls
    elif hasattr(inner_agent_config, 'max_iterations'):
        # Some agents use max_iterations as an alias
        inner_max_calls = inner_agent_config.max_iterations
    elif hasattr(inner_agent_config, 'tool_call_max_retries'):
        # ReWOO agent uses tool_call_max_retries - needs more steps per retry
        inner_max_calls = inner_agent_config.tool_call_max_retries * 3
    else:
        # Safe default for agents without explicit limits
        inner_max_calls = 15

    # Use same calculation formula as react_agent for consistency
    # Formula: (max_tool_calls + 1) * 2 allows proper tool calling cycles with retries
    # See src/nat/agent/react_agent/register.py:145 for reference
    inner_agent_recursion = (inner_max_calls + 1) * 2

    # Create wrapper
    wrapper_graph = AutoMemoryWrapperGraph(inner_agent_fn=inner_agent_fn,
                                           memory_editor=memory_editor,
                                           save_user_messages=config.save_user_messages_to_memory,
                                           retrieve_memory=config.retrieve_memory_for_every_response,
                                           save_ai_responses=config.save_ai_messages_to_memory,
                                           search_params=config.search_params,
                                           add_params=config.add_params)

    # Calculate total recursion limit: wrapper overhead + inner agent needs
    wrapper_node_count = wrapper_graph.get_wrapper_node_count()
    total_recursion_limit = wrapper_node_count + inner_agent_recursion

    logger.debug(f"{AGENT_LOG_PREFIX} Auto-memory wrapper calculated recursion_limit={total_recursion_limit} "
                 f"(wrapper_overhead={wrapper_node_count} + inner_agent={inner_agent_recursion})")

    # Build the graph
    graph: CompiledStateGraph = wrapper_graph.build_graph()

    async def _response_fn(input_message: str) -> str:
        """
        Main workflow entry function for the auto-memory agent.

        Args:
            input_message (str): The input message to process

        Returns:
            str: The response from the wrapped agent
        """
        try:
            message = HumanMessage(content=input_message)
            state = AutoMemoryWrapperState(messages=[message])

            # Pass calculated recursion_limit to ensure wrapper + inner agent have enough steps
            result_dict = await graph.ainvoke(state, config={'recursion_limit': total_recursion_limit})
            result_state = AutoMemoryWrapperState(**result_dict)

            output_message = result_state.messages[-1]
            return str(output_message.content)

        except Exception as ex:
            logger.exception(f"{AGENT_LOG_PREFIX} Auto-memory agent failed with exception")
            if config.verbose:
                return str(ex)
            return f"Auto-memory agent failed with exception: {ex}"

    try:
        yield FunctionInfo.from_fn(_response_fn, description=config.description)
    except GeneratorExit:
        logger.exception("%s Workflow exited early!", AGENT_LOG_PREFIX)
    finally:
        logger.debug("%s Cleaning up auto_memory_agent workflow.", AGENT_LOG_PREFIX)
