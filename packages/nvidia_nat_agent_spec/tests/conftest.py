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

"""Pytest configuration and shared fixtures for Agent-Spec plugin tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nat.builder.builder import Builder
from nat.plugins.agent_spec.agent_spec_workflow import (
    AgentSpecWrapperConfig,
    AgentSpecWrapperFunction,
)


@pytest.fixture(autouse=True)
def setup_api_key():
    """Set OPENAI_API_KEY from NVIDIA_API_KEY if not already set.

    Agent-Spec's LangGraph converter uses ChatOpenAI which expects OPENAI_API_KEY,
    but NIM users typically have NVIDIA_API_KEY set. This fixture ensures compatibility.
    """
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    # If NVIDIA_API_KEY is set but OPENAI_API_KEY is not, set it
    if nvidia_key and not openai_key:
        os.environ["OPENAI_API_KEY"] = nvidia_key
        yield
        # Clean up: only remove if we set it
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
    else:
        yield


@pytest.fixture
def mock_graph():
    """Create a mock CompiledStateGraph."""
    mock_graph = MagicMock()
    from langgraph.graph.state import CompiledStateGraph
    mock_graph.__class__ = CompiledStateGraph  # type: ignore[assignment]
    return mock_graph


@pytest.fixture
def wrapper_function(mock_graph):
    """Create an AgentSpecWrapperFunction instance."""
    config = AgentSpecWrapperConfig(
        spec_file=Path(__file__),
        description="Test wrapper",
    )
    return AgentSpecWrapperFunction(
        config=config,
        description="Test wrapper",
        graph=mock_graph,
    )


@pytest.fixture
def mock_builder():
    """Create a mock Builder instance."""
    return MagicMock(spec=Builder)


@pytest.fixture
def minimal_yaml_content():
    """Minimal valid Agent-Spec YAML content."""
    return """
component_type: Agent
name: test_agent
description: Test agent
llm_config:
  component_type: OpenAiCompatibleConfig
  name: Test LLM
  model_id: test-model
"""


@pytest.fixture
def temp_yaml_file(minimal_yaml_content):
    """Create a temporary YAML file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(minimal_yaml_content)
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink(missing_ok=True)
