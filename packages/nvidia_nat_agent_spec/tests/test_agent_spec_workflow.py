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

"""Smoke tests for Agent-Spec workflow wrapper."""

import pytest
from pathlib import Path

from nat.plugins.agent_spec.agent_spec_workflow import (
    AgentSpecWrapperConfig,
    AgentSpecWrapperFunction,
    AgentSpecWrapperInput,
    AgentSpecWrapperOutput,
)


def test_imports():
    """Test that all imports work correctly."""
    # This is a basic smoke test to ensure the module can be imported
    assert AgentSpecWrapperConfig is not None
    assert AgentSpecWrapperFunction is not None
    assert AgentSpecWrapperInput is not None
    assert AgentSpecWrapperOutput is not None


def test_config_model():
    """Test that the config model is properly defined."""
    # Test that the config has the right name (full_type includes module path)
    assert AgentSpecWrapperConfig.full_type == "nat.plugins.agent_spec/agent_spec_wrapper"
    assert "agent_spec_wrapper" in AgentSpecWrapperConfig.full_type

    # Test that we can create a config instance with a valid file path
    # Use __file__ itself as a valid file path for testing
    config = AgentSpecWrapperConfig(
        spec_file=Path(__file__),  # Use the test file itself as a valid path
        description="Test agent",
    )
    assert config.description == "Test agent"
    assert config.tool_registry is None


def test_config_with_tool_registry():
    """Test config with tool registry."""
    config = AgentSpecWrapperConfig(
        spec_file=Path(__file__),  # Use the test file itself as a valid path
        tool_registry={"test_tool": lambda x: x},
    )
    assert config.tool_registry is not None
    assert "test_tool" in config.tool_registry


def test_config_with_checkpointer():
    """Test config with checkpointer."""
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    config = AgentSpecWrapperConfig(
        spec_file=Path(__file__),  # Use the test file itself as a valid path
        checkpointer=checkpointer,
    )
    assert config.checkpointer is not None
    assert config.checkpointer is checkpointer
    assert isinstance(config.checkpointer, MemorySaver)


@pytest.mark.skip(reason="Requires actual Agent-Spec YAML file and dependencies")
def test_load_agent_spec_yaml():
    """Test loading an Agent-Spec YAML file.

    This test is skipped by default as it requires:
    1. A valid Agent-Spec YAML file
    2. pyagentspec[langgraph] dependencies installed
    3. Potentially LLM API keys

    To run this test, create a test YAML file and unskip it.
    """
    # TODO: Create a minimal test YAML file
    # TODO: Test that AgentSpecLoader can load it
    # TODO: Test that we get a CompiledStateGraph back
    pass
