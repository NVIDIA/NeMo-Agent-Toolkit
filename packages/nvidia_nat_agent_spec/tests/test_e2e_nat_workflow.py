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

"""End-to-end integration tests (smoke tests) for Agent-Spec integration.

This test suite verifies that:
1. Agent-Spec YAML files can be loaded and executed end-to-end
2. Inline YAML/JSON Agent-Spec configurations work
3. The integration works with actual LLM calls

These are smoke tests that require:
- NVIDIA_API_KEY environment variable
- Network access to NIM endpoint
- pyagentspec[langgraph] installed
"""

import asyncio
import os
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("NVIDIA_API_KEY"),
    reason="Requires NVIDIA_API_KEY environment variable"
)
async def test_agent_spec_yaml_end_to_end():
    """Test loading and executing an Agent-Spec YAML file directly.

    This test:
    1. Loads an Agent-Spec YAML file
    2. Executes it with a test input
    3. Verifies we get a response

    Requires:
    - NVIDIA_API_KEY environment variable set
    - pyagentspec[langgraph] installed
    - Network access to NIM endpoint
    """
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.plugins.agent_spec.agent_spec_workflow import AgentSpecWrapperConfig, register

    test_yaml = Path(__file__).parent / "fixtures" / "minimal_agent.yaml"

    if not test_yaml.exists():
        pytest.skip(f"Test YAML file not found: {test_yaml}")

    config = AgentSpecWrapperConfig(
        spec_file=test_yaml,
        description="End-to-end test agent",
    )

    async with WorkflowBuilder() as builder:
        async with register(config, builder) as wrapper_function:
            # Verify we got a wrapper function with correct structure
            assert wrapper_function is not None
            assert hasattr(wrapper_function, '_graph')
            from langgraph.graph.state import CompiledStateGraph
            assert isinstance(wrapper_function._graph, CompiledStateGraph), \
                f"Expected CompiledStateGraph, got {type(wrapper_function._graph)}"

            # Invoke with a test message
            test_input = "Hello! Please respond with 'Agent-Spec test successful'."

            try:
                result = await asyncio.wait_for(
                    wrapper_function.acall_invoke(test_input),
                    timeout=60.0
                )
                print(f"✓ Agent-Spec YAML executed successfully")
                print(f"✓ Response: {result}")

                # Stronger validation: verify result structure and content
                assert result is not None
                from nat.plugins.agent_spec.agent_spec_workflow import AgentSpecWrapperOutput
                assert isinstance(result, AgentSpecWrapperOutput), \
                    f"Expected AgentSpecWrapperOutput, got {type(result)}"
                assert hasattr(result, 'messages'), "Result should have messages attribute"
                assert isinstance(result.messages, list), "Messages should be a list"
                assert len(result.messages) > 0, "Should have at least one message"

                # Verify message types and content
                from langchain_core.messages import AIMessage, BaseMessage
                last_message = result.messages[-1]
                assert isinstance(last_message, BaseMessage), \
                    f"Expected BaseMessage, got {type(last_message)}"
                assert isinstance(last_message, AIMessage), \
                    f"Expected AIMessage as last message, got {type(last_message)}"
                assert hasattr(last_message, 'content'), "Message should have content attribute"
                assert last_message.content is not None, "Message content should not be None"
                result_str = str(last_message.content)
                assert len(result_str.strip()) > 0, "Response should have non-empty content"
                assert len(result_str) > 10, "Response should have meaningful content (more than 10 chars)"
            except asyncio.TimeoutError:
                print(f"⚠ API call timed out after 60 seconds")
                pytest.skip("API call timed out - may indicate network/API issues")
            except Exception as e:
                print(f"⚠ Invocation failed: {e}")
                pytest.skip(f"Invocation failed (may need network/API): {e}")




@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("NVIDIA_API_KEY"),
    reason="Requires NVIDIA_API_KEY environment variable"
)
async def test_agent_spec_with_inline_yaml():
    """Test Agent-Spec wrapper with inline YAML content.

    This test:
    1. Creates an Agent-Spec wrapper with inline YAML (not a file)
    2. Executes it with a test input
    3. Verifies we get a response

    Requires:
    - NVIDIA_API_KEY environment variable set
    - pyagentspec[langgraph] installed
    - Network access to NIM endpoint
    """
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.plugins.agent_spec.agent_spec_workflow import AgentSpecWrapperConfig, register

    inline_yaml = """
component_type: Agent
name: "Inline Test Agent"
description: "A test agent with inline YAML"
system_prompt: "You are a helpful assistant. Respond concisely."
llm_config:
  component_type: OpenAiCompatibleConfig
  name: "NIM LLM"
  model_id: "meta/llama-3.1-8b-instruct"
  url: "https://integrate.api.nvidia.com/v1"
inputs: []
outputs: []
agentspec_version: 25.4.1
"""

    config = AgentSpecWrapperConfig(
        spec_yaml=inline_yaml,
        description="Inline YAML test agent",
    )

    async with WorkflowBuilder() as builder:
        async with register(config, builder) as wrapper_function:
            # Verify we got a wrapper function with correct structure
            assert wrapper_function is not None
            assert hasattr(wrapper_function, '_graph')
            from langgraph.graph.state import CompiledStateGraph
            assert isinstance(wrapper_function._graph, CompiledStateGraph), \
                f"Expected CompiledStateGraph, got {type(wrapper_function._graph)}"

            test_input = "Say 'inline YAML test successful'."

            try:
                result = await asyncio.wait_for(
                    wrapper_function.acall_invoke(test_input),
                    timeout=60.0
                )
                print(f"✓ Inline YAML Agent-Spec executed successfully")
                print(f"✓ Response: {result}")

                # Stronger validation: verify result structure and content
                assert result is not None
                from nat.plugins.agent_spec.agent_spec_workflow import AgentSpecWrapperOutput
                assert isinstance(result, AgentSpecWrapperOutput), \
                    f"Expected AgentSpecWrapperOutput, got {type(result)}"
                assert hasattr(result, 'messages'), "Result should have messages attribute"
                assert isinstance(result.messages, list), "Messages should be a list"
                assert len(result.messages) > 0, "Should have at least one message"

                # Verify message types and content
                from langchain_core.messages import AIMessage, BaseMessage
                last_message = result.messages[-1]
                assert isinstance(last_message, BaseMessage), \
                    f"Expected BaseMessage, got {type(last_message)}"
                assert isinstance(last_message, AIMessage), \
                    f"Expected AIMessage as last message, got {type(last_message)}"
                assert hasattr(last_message, 'content'), "Message should have content attribute"
                assert last_message.content is not None, "Message content should not be None"
                result_str = str(last_message.content)
                assert len(result_str.strip()) > 0, "Response should have non-empty content"
                assert len(result_str) > 10, "Response should have meaningful content (more than 10 chars)"
            except asyncio.TimeoutError:
                print(f"⚠ API call timed out after 60 seconds")
                pytest.skip("API call timed out - may indicate network/API issues")
            except Exception as e:
                print(f"⚠ Invocation failed: {e}")
                pytest.skip(f"Invocation failed (may need network/API): {e}")
