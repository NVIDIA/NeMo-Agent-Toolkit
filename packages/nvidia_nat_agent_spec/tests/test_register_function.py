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
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
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

    def _create_mock_loader_instance(self, mock_graph, load_yaml_return=None, load_json_return=None):
        """Helper to create a mock loader instance."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.load_yaml.return_value = load_yaml_return or mock_graph
        mock_loader_instance.load_json.return_value = load_json_return or mock_graph
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

    # Tests for inline YAML/JSON support
    def test_config_validator_requires_exactly_one_source(self):
        """Test validator ensures exactly one of spec_file, spec_yaml, or spec_json is provided."""
        # Test: None provided - should fail
        with pytest.raises(ValueError, match="Exactly one of spec_file, spec_yaml, or spec_json must be provided"):
            AgentSpecWrapperConfig()

        # Test: Multiple provided - should fail
        with pytest.raises(ValueError, match="Exactly one of spec_file, spec_yaml, or spec_json must be provided"):
            AgentSpecWrapperConfig(
                spec_file=Path(__file__),
                spec_yaml="test",
            )

        with pytest.raises(ValueError, match="Exactly one of spec_file, spec_yaml, or spec_json must be provided"):
            AgentSpecWrapperConfig(
                spec_yaml="test",
                spec_json='{"test": "value"}',
            )

        with pytest.raises(ValueError, match="Exactly one of spec_file, spec_yaml, or spec_json must be provided"):
            AgentSpecWrapperConfig(
                spec_file=Path(__file__),
                spec_yaml="test",
                spec_json='{"test": "value"}',
            )

        # Test: Exactly one provided - should succeed
        config1 = AgentSpecWrapperConfig(spec_file=Path(__file__))
        assert config1.spec_file == Path(__file__)

        config2 = AgentSpecWrapperConfig(spec_yaml="component_type: Agent")
        assert config2.spec_yaml == "component_type: Agent"

        config3 = AgentSpecWrapperConfig(spec_json='{"component_type": "Agent"}')
        assert config3.spec_json == '{"component_type": "Agent"}'

    @pytest.mark.parametrize("format_type,content,config_kwargs,expected_method,not_expected_method", [
        (
            "yaml",
            None,  # Will use minimal_yaml_content fixture
            {"spec_yaml": None, "description": "Test agent with inline YAML"},
            "load_yaml",
            "load_json",
        ),
        (
            "json",
            '{"component_type": "Agent", "name": "test_agent", "llm_config": {"component_type": "OpenAiCompatibleConfig", "model_id": "test-model"}}',
            {"spec_json": None, "description": "Test agent with inline JSON"},
            "load_json",
            "load_yaml",
        ),
    ])
    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_with_inline_content(
        self, mock_loader_class, mock_builder, mock_graph, minimal_yaml_content, format_type, content, config_kwargs, expected_method, not_expected_method
    ):
        """Test register() with inline YAML or JSON content."""
        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        # Use fixture content for YAML, provided content for JSON
        actual_content = minimal_yaml_content if format_type == "yaml" else content
        config_kwargs[list(config_kwargs.keys())[0]] = actual_content  # Set spec_yaml or spec_json

        config = AgentSpecWrapperConfig(**config_kwargs)

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None
            assert wrapper_function._graph is mock_graph

        # Verify correct method was called
        expected_loader_method = getattr(mock_loader_instance, expected_method)
        not_expected_loader_method = getattr(mock_loader_instance, not_expected_method)

        expected_loader_method.assert_called_once()
        not_expected_loader_method.assert_not_called()

        # Verify the content was passed
        call_args = expected_loader_method.call_args
        assert call_args.args[0] == actual_content

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_with_json_file(self, mock_loader_class, mock_builder, mock_graph):
        """Test register() with JSON file (spec_file with .json extension)."""
        import tempfile
        json_content = '{"component_type": "Agent", "name": "test_agent"}'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json_content)
            temp_json_path = Path(f.name)

        try:
            mock_loader_instance = self._create_mock_loader_instance(mock_graph)
            mock_loader_class.return_value = mock_loader_instance

            config = AgentSpecWrapperConfig(
                spec_file=temp_json_path,
                description="Test agent with JSON file",
            )

            async with register(config, mock_builder) as wrapper_function:
                assert wrapper_function is not None
                assert wrapper_function._graph is mock_graph

            # Verify load_json was called (not load_yaml) for .json file
            mock_loader_instance.load_json.assert_called_once()
            mock_loader_instance.load_yaml.assert_not_called()

            # Verify the JSON content was read and passed
            call_args = mock_loader_instance.load_json.call_args
            assert call_args.args[0] == json_content
        finally:
            temp_json_path.unlink(missing_ok=True)

    @pytest.mark.parametrize("format_type,content", [
        ("yaml", None),  # Will use minimal_yaml_content fixture
        ("json", '{"component_type": "Agent", "name": "test_agent"}'),
    ])
    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_inline_content_with_components_registry(
        self, mock_loader_class, mock_builder, mock_graph, minimal_yaml_content, format_type, content
    ):
        """Test register() with inline YAML or JSON and components_registry."""
        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        # Use fixture content for YAML, provided content for JSON
        actual_content = minimal_yaml_content if format_type == "yaml" else content
        components_registry = {"llm_id": MagicMock()}

        config_kwargs = {
            "components_registry": components_registry,
        }
        if format_type == "yaml":
            config_kwargs["spec_yaml"] = actual_content
        else:
            config_kwargs["spec_json"] = actual_content

        config = AgentSpecWrapperConfig(**config_kwargs)

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None

        # Verify correct method was called with components_registry
        loader_method = getattr(mock_loader_instance, f"load_{format_type}")
        loader_method.assert_called_once()
        call_args = loader_method.call_args
        assert 'components_registry' in call_args.kwargs
        assert call_args.kwargs['components_registry'] == components_registry

    @pytest.mark.parametrize("format_type,content", [
        ("yaml", None),  # Will use yaml_with_client_tool fixture
        ("json", '{"component_type": "Agent", "name": "test_agent", "tools": [{"component_type": "ClientTool", "name": "test_tool"}]}'),
    ])
    @patch('langgraph.checkpoint.memory.MemorySaver')
    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_inline_content_auto_detects_client_tool(
        self, mock_loader_class, mock_memory_saver_class, mock_builder, mock_graph, yaml_with_client_tool, format_type, content
    ):
        """Test register() auto-detects ClientTool in inline YAML or JSON."""
        # Use fixture content for YAML, provided content for JSON
        actual_content = yaml_with_client_tool if format_type == "yaml" else content

        mock_checkpointer = MagicMock()
        mock_memory_saver_class.return_value = mock_checkpointer
        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        config_kwargs = {}
        if format_type == "yaml":
            config_kwargs["spec_yaml"] = actual_content
        else:
            config_kwargs["spec_json"] = actual_content

        config = AgentSpecWrapperConfig(**config_kwargs)

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None

        # Verify MemorySaver was created
        mock_memory_saver_class.assert_called_once()

        # Verify checkpointer was passed to loader
        call_kwargs = mock_loader_class.call_args.kwargs
        assert call_kwargs['checkpointer'] is mock_checkpointer

    # Tests for tool_names integration
    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_with_tool_names(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test register() with tool_names resolves tools from NAT registry."""
        # Create mock tools
        mock_tool1 = MagicMock()
        mock_tool1.name = "weather_tool"
        mock_tool2 = MagicMock()
        mock_tool2.name = "calculator_tool"
        mock_tools = [mock_tool1, mock_tool2]

        # Mock builder.get_tools to return our mock tools (async)
        mock_builder.get_tools = AsyncMock(return_value=mock_tools)

        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
            tool_names=[FunctionRef("weather_tool"), FunctionRef("calculator_tool")],
        )

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None

        # Verify builder.get_tools was called
        mock_builder.get_tools.assert_called_once_with(
            tool_names=[FunctionRef("weather_tool"), FunctionRef("calculator_tool")],
            wrapper_type=LLMFrameworkEnum.LANGCHAIN
        )

        # Verify tool_registry was built correctly
        call_kwargs = mock_loader_class.call_args.kwargs
        tool_registry = call_kwargs['tool_registry']
        assert len(tool_registry) == 2
        assert tool_registry["weather_tool"] is mock_tool1
        assert tool_registry["calculator_tool"] is mock_tool2

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_with_tool_names_and_tool_registry(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test register() merges tool_names and tool_registry, with tool_registry overriding."""
        # Create mock tools from tool_names
        mock_tool1 = MagicMock()
        mock_tool1.name = "weather_tool"
        mock_tools = [mock_tool1]

        mock_builder.get_tools = MagicMock(return_value=mock_tools)

        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        # Custom tool_registry with one overlapping name and one new tool
        custom_tool_registry = {
            "weather_tool": MagicMock(),  # Override tool_names tool
            "custom_tool": MagicMock(),   # New tool
        }

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
            tool_names=[FunctionRef("weather_tool")],
            tool_registry=custom_tool_registry,
        )

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None

        # Verify tool_registry was merged correctly (tool_registry overrides)
        call_kwargs = mock_loader_class.call_args.kwargs
        tool_registry = call_kwargs['tool_registry']
        assert len(tool_registry) == 2
        # tool_registry should override tool_names for weather_tool
        assert tool_registry["weather_tool"] is custom_tool_registry["weather_tool"]
        assert tool_registry["custom_tool"] is custom_tool_registry["custom_tool"]

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_with_tool_names_empty_list(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test register() with empty tool_names list."""
        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
            tool_names=[],
        )

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None

        # Verify tool_registry is empty
        call_kwargs = mock_loader_class.call_args.kwargs
        assert call_kwargs['tool_registry'] == {}

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_with_tool_names_resolution_failure(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test register() handles tool_names resolution failure gracefully."""
        # Mock builder.get_tools to raise an error (async)
        mock_builder.get_tools = AsyncMock(side_effect=ValueError("Tool not found"))

        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
            tool_names=[FunctionRef("missing_tool")],
        )

        # Should not raise - error is logged as warning and continues
        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None

        # Verify tool_registry is empty (resolution failed)
        call_kwargs = mock_loader_class.call_args.kwargs
        assert call_kwargs['tool_registry'] == {}

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_with_tool_names_and_tool_registry_resolution_failure(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test register() uses tool_registry even if tool_names resolution fails."""
        # Mock builder.get_tools to raise an error (async)
        mock_builder.get_tools = AsyncMock(side_effect=ValueError("Tool not found"))

        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        custom_tool_registry = {"custom_tool": MagicMock()}

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
            tool_names=[FunctionRef("missing_tool")],
            tool_registry=custom_tool_registry,
        )

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None

        # Verify tool_registry still has custom tool even though tool_names failed
        call_kwargs = mock_loader_class.call_args.kwargs
        tool_registry = call_kwargs['tool_registry']
        assert len(tool_registry) == 1
        assert tool_registry["custom_tool"] is custom_tool_registry["custom_tool"]

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_with_function_group_ref(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test register() with FunctionGroupRef."""
        # Create mock tools from group
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tools = [mock_tool1, mock_tool2]

        mock_builder.get_tools = AsyncMock(return_value=mock_tools)

        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
            tool_names=[FunctionGroupRef("my_tool_group")],
        )

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None

        # Verify builder.get_tools was called with FunctionGroupRef
        mock_builder.get_tools.assert_called_once_with(
            tool_names=[FunctionGroupRef("my_tool_group")],
            wrapper_type=LLMFrameworkEnum.LANGCHAIN
        )

        # Verify tool_registry contains tools from group
        call_kwargs = mock_loader_class.call_args.kwargs
        tool_registry = call_kwargs['tool_registry']
        assert len(tool_registry) == 2
        assert tool_registry["tool1"] is mock_tool1
        assert tool_registry["tool2"] is mock_tool2

    # Tests for max_history message trimming
    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_with_max_history(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test register() configures max_history for message trimming."""
        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
            max_history=10,
        )

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None
            assert wrapper_function.config.max_history == 10

    @patch('pyagentspec.adapters.langgraph.AgentSpecLoader')
    async def test_register_max_history_default(self, mock_loader_class, temp_yaml_file, mock_builder, mock_graph):
        """Test register() uses default max_history value."""
        mock_loader_instance = self._create_mock_loader_instance(mock_graph)
        mock_loader_class.return_value = mock_loader_instance

        config = AgentSpecWrapperConfig(
            spec_file=temp_yaml_file,
        )

        async with register(config, mock_builder) as wrapper_function:
            assert wrapper_function is not None
            assert wrapper_function.config.max_history == 15  # Default value
