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

"""Unit tests for register() function with mocked dependencies."""

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.plugins.agent_spec.agent_spec_workflow import AgentSpecWrapperConfig
from nat.plugins.agent_spec.agent_spec_workflow import register


class TestRegisterFunction:
    """Test cases for the register() function."""

    @pytest.fixture
    def yaml_with_client_tool(self):
        """Agent-Spec YAML content with ClientTool."""
        return """
component_type: Agent
name: test_agent
description: Test agent with tool
llm_config:
  component_type: OpenAiCompatibleConfig
  name: Test LLM
  model_id: test-model
tools:
  - component_type: ClientTool
    name: test_tool
    description: A test tool
"""

    def _create_mock_loader_instance(self, mock_graph, load_yaml_return=None):
        """Helper to create a mock loader instance."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.load_yaml.return_value = load_yaml_return or mock_graph
        return mock_loader_instance

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_basic_flow(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test basic register() flow with minimal config."""
        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
            description="Test agent",
        )

        # Execute
        async with register(config, mock_builder) as wrapper_function:
            # Verify
            assert wrapper_function is not None
            assert hasattr(wrapper_function, '_graph')
            assert wrapper_function._graph is mock_graph

        # Verify AgentSpecLoader was instantiated correctly
        mock_loader_class.assert_called_once()
        call_kwargs = mock_loader_class.call_args.kwargs
        assert call_kwargs['tool_registry'] == {}
        assert call_kwargs['checkpointer'] is None

        # Verify load_yaml was called
        mock_loader_instance.load_yaml.assert_called_once()

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_with_tool_registry(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test register() with tool registry."""
        mock_loader_class.return_value = self._create_mock_loader_instance(mock_graph)

        tool_registry = {"test_tool": lambda x: x}
        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
            tool_registry=tool_registry,
        )

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None

        # Verify tool_registry was passed to loader
        call_kwargs = mock_loader_class.call_args.kwargs
        assert call_kwargs['tool_registry'] == tool_registry

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_with_explicit_checkpointer(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test register() with explicitly provided checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver

        mock_loader_class.return_value = self._create_mock_loader_instance(mock_graph)

        checkpointer = MemorySaver()
        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
            checkpointer=checkpointer,
        )

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None

        # Verify checkpointer was passed to loader
        call_kwargs = mock_loader_class.call_args.kwargs
        assert call_kwargs['checkpointer'] is checkpointer

    @patch('langgraph.checkpoint.memory.MemorySaver')
    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_auto_detects_client_tool(
        self, mock_loader_class, mock_memory_saver_class, temp_yaml_file, mock_builder, mock_graph, yaml_with_client_tool
    ):
        """Test register() auto-detects ClientTool and creates checkpointer."""
        # Write YAML with ClientTool to temp file
        temp_yaml_file.write_text(yaml_with_client_tool)

        mock_checkpointer = MagicMock()
        mock_memory_saver_class.return_value = mock_checkpointer
        mock_loader_class.return_value = self._create_mock_loader_instance(mock_graph)

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
        )

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None

        # Verify MemorySaver was created
        mock_memory_saver_class.assert_called_once()

        # Verify checkpointer was passed to loader
        call_kwargs = mock_loader_class.call_args.kwargs
        assert call_kwargs['checkpointer'] is mock_checkpointer

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_with_components_registry(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test register() with components_registry."""
        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        components_registry = {"llm_id": MagicMock()}
        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
            components_registry=components_registry,
        )

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None

        # Verify load_yaml was called with components_registry
        mock_loader_instance.load_yaml.assert_called_once()
        call_args = mock_loader_instance.load_yaml.call_args
        assert 'components_registry' in call_args.kwargs
        assert call_args.kwargs['components_registry'] == components_registry

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_without_components_registry(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test register() without components_registry calls load_yaml without it."""
        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
        )

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None

        # Verify load_yaml was called without components_registry
        mock_loader_instance.load_yaml.assert_called_once()
        call_args = mock_loader_instance.load_yaml.call_args
        # Should be called with just the YAML string, no components_registry kwarg
        assert len(call_args.args) == 1  # Just the YAML string
        assert 'components_registry' not in call_args.kwargs

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    @patch('pathlib.Path.exists')
    async def test_register_file_not_found(self, mock_exists, mock_loader_class, mock_builder, temp_yaml_file):
        """Test register() raises ValueError when file doesn't exist."""
        # Make Path.exists() return False to simulate non-existent file
        mock_exists.return_value = False

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
        )

        with pytest.raises(ValueError, match="does not exist"):
            async with register(config, mock_builder):
                pass

        # AgentSpecLoader should not be instantiated if file doesn't exist
        mock_loader_class.assert_not_called()

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_import_error(self, mock_loader_class, temp_yaml_file, mock_builder):
        """Test register() raises ImportError when pyagentspec is not available."""
        # Make the import fail
        mock_loader_class.side_effect = ImportError("No module named 'pyagentspec'")

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
        )

        with pytest.raises(ImportError, match="Failed to import pyagentspec|No module named"):
            async with register(config, mock_builder):
                pass

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_loader_raises_runtime_error(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test register() wraps loader exceptions in RuntimeError."""
        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_instance.load_yaml.side_effect = ValueError("Invalid YAML structure")
        mock_loader_class.return_value = mock_loader_instance

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
        )

        with pytest.raises(RuntimeError, match="Failed to load Agent-Spec configuration"):
            async with register(config, mock_builder):
                pass

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_invalid_graph_type(self, mock_loader_class, temp_yaml_file, mock_builder):
        """Test register() raises ValueError when loader returns wrong type."""
        mock_loader_instance = self._create_mock_loader_instance("not a graph")
        mock_loader_class.return_value = mock_loader_instance

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
        )

        with pytest.raises(ValueError, match="unexpected type"):
            async with register(config, mock_builder):
                pass

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_yaml_parse_error_for_client_tool(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test register() handles YAML parse errors gracefully when detecting ClientTool."""
        temp_yaml_file.write_text("invalid: yaml: content: [unclosed")
        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
        )

        # Should not raise - YAML parse error should be caught and logged
        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None

        # Checkpointer should be None since YAML parsing failed
        call_kwargs = mock_loader_class.call_args.kwargs
        assert call_kwargs['checkpointer'] is None

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_yaml_without_tools_section(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test register() handles YAML without tools section."""
        yaml_no_tools = """
component_type: Agent
name: test_agent
llm_config:
  component_type: OpenAiCompatibleConfig
  model_id: test-model
"""
        temp_yaml_file.write_text(yaml_no_tools)
        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
        )

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None

        # Checkpointer should be None since no ClientTool detected
        call_kwargs = mock_loader_class.call_args.kwargs
        assert call_kwargs['checkpointer'] is None

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_preserves_description(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test register() preserves description in wrapper function."""
        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        description = "Custom description for test agent"
        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
            description=description,
        )

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None
            assert wrapper_function.description == description
