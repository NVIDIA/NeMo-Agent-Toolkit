# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from pydantic import Field

from nat.builder.function import Function
from nat.builder.workflow import Workflow
from nat.front_ends.mcp.tool_converter import create_function_wrapper
from nat.front_ends.mcp.tool_converter import get_function_description
from nat.front_ends.mcp.tool_converter import register_function_with_mcp


# Test schemas
class MockChatRequest(BaseModel):
    """Mock ChatRequest for testing."""
    __name__ = "ChatRequest"
    query: str


class MockRegularSchema(BaseModel):
    """Mock regular schema for testing."""
    name: str
    age: int = Field(default=25)


def create_mock_workflow_with_observability():
    """Create a mock workflow with proper observability setup."""
    mock_workflow = MagicMock(spec=Workflow)
    mock_workflow.exporter_manager = MagicMock()

    # Create a proper async context manager mock
    async_context_manager = AsyncMock()
    async_context_manager.__aenter__ = AsyncMock(return_value=None)
    async_context_manager.__aexit__ = AsyncMock(return_value=None)
    mock_workflow.exporter_manager.start.return_value = async_context_manager

    return mock_workflow


class TestCreateFunctionWrapper:
    """Test cases for create_function_wrapper function."""

    def test_create_wrapper_for_chat_request_function(self):
        """Test creating wrapper for function with ChatRequest schema."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        function_name = "test_function"
        schema = MockChatRequest

        # Act
        wrapper = create_function_wrapper(function_name, mock_function, schema, False, None)

        # Assert
        assert callable(wrapper)
        assert wrapper.__name__ == function_name
        sig = getattr(wrapper, '__signature__', None)
        assert sig is not None
        assert "query" in sig.parameters

    def test_create_wrapper_for_regular_function(self):
        """Test creating wrapper for function with regular schema."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        function_name = "regular_function"
        schema = MockRegularSchema

        # Act
        wrapper = create_function_wrapper(function_name, mock_function, schema, False, None)

        # Assert
        assert callable(wrapper)
        assert wrapper.__name__ == function_name
        sig = getattr(wrapper, '__signature__', None)
        assert sig is not None
        assert "name" in sig.parameters
        assert "age" in sig.parameters

    def test_create_wrapper_for_workflow(self):
        """Test creating wrapper for workflow function."""
        # Arrange
        mock_workflow = MagicMock(spec=Workflow)
        function_name = "test_workflow"
        schema = MockChatRequest

        # Act
        wrapper = create_function_wrapper(function_name, mock_workflow, schema, True, mock_workflow)

        # Assert
        assert callable(wrapper)
        assert wrapper.__name__ == function_name

    @patch('nat.front_ends.mcp.tool_converter.ContextState')
    async def test_wrapper_execution_with_observability(self, mock_context_state_class):
        """Test wrapper execution with observability context."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        mock_function.acall_invoke = AsyncMock(return_value="result")

        mock_workflow = create_mock_workflow_with_observability()

        # Mock ContextState.get()
        mock_context_state = MagicMock()
        mock_context_state_class.get.return_value = mock_context_state

        wrapper = create_function_wrapper("test_func", mock_function, MockRegularSchema, False, mock_workflow)

        # Act
        result = await wrapper(name="test", age=30)

        # Assert
        assert result == "result"
        mock_workflow.exporter_manager.start.assert_called_once_with(context_state=mock_context_state)
        mock_context_state_class.get.assert_called_once()

    async def test_wrapper_execution_without_workflow_fails(self):
        """Test wrapper execution fails without workflow context."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        wrapper = create_function_wrapper("test_func", mock_function, MockChatRequest, False, None)

        # Act & Assert
        with pytest.raises(RuntimeError, match="Workflow context is required for observability"):
            await wrapper(query="test")


class TestGetFunctionDescription:
    """Test cases for get_function_description function."""

    def test_get_description_from_workflow_description(self):
        """Test getting description from workflow's description attribute."""
        # Arrange
        mock_workflow = MagicMock(spec=Workflow)
        mock_workflow.description = "Direct workflow description"
        mock_workflow.config = MagicMock()

        # Act
        result = get_function_description(mock_workflow)

        # Assert
        assert result == "Direct workflow description"

    def test_get_description_from_workflow_config(self):
        """Test getting description from workflow config."""
        # Arrange
        mock_workflow = MagicMock(spec=Workflow)
        mock_workflow.description = None
        mock_workflow.config = MagicMock()
        mock_workflow.config.description = "Config description"

        # Act
        result = get_function_description(mock_workflow)

        # Assert
        assert result == "Config description"

    def test_get_description_from_function(self):
        """Test getting description from regular function."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        mock_function.description = "Function description"

        # Act
        result = get_function_description(mock_function)

        # Assert
        assert result == "Function description"

    def test_get_empty_description(self):
        """Test getting empty description when none available."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        mock_function.description = ""

        # Act
        result = get_function_description(mock_function)

        # Assert
        assert result == ""


class TestRegisterFunctionWithMcp:
    """Test cases for register_function_with_mcp function."""

    @patch('nat.front_ends.mcp.tool_converter.create_function_wrapper')
    @patch('nat.front_ends.mcp.tool_converter.get_function_description')
    @patch('nat.front_ends.mcp.tool_converter.logger')
    def test_register_function_with_mcp(self, mock_logger, mock_get_desc, mock_create_wrapper):
        """Test registering a regular function with MCP."""
        # Arrange
        mock_mcp = MagicMock()
        mock_function = MagicMock(spec=Function)
        mock_workflow = MagicMock(spec=Workflow)
        function_name = "test_function"

        mock_get_desc.return_value = "Test description"
        mock_wrapper = MagicMock()
        mock_create_wrapper.return_value = mock_wrapper

        # Act
        register_function_with_mcp(mock_mcp, function_name, mock_function, mock_workflow)

        # Assert - Check that logging happened (actual message order may vary)
        assert mock_logger.info.call_count >= 1
        mock_get_desc.assert_called_once_with(mock_function)
        mock_create_wrapper.assert_called_once_with(function_name,
                                                    mock_function,
                                                    mock_function.input_schema,
                                                    False,
                                                    mock_workflow)
        mock_mcp.tool.assert_called_once_with(name=function_name, description="Test description")

    @patch('nat.front_ends.mcp.tool_converter.create_function_wrapper')
    @patch('nat.front_ends.mcp.tool_converter.get_function_description')
    @patch('nat.front_ends.mcp.tool_converter.logger')
    def test_register_workflow_with_mcp(self, mock_logger, mock_get_desc, mock_create_wrapper):
        """Test registering a workflow with MCP."""
        # Arrange
        mock_mcp = MagicMock()
        mock_workflow = MagicMock(spec=Workflow)
        function_name = "test_workflow"

        mock_get_desc.return_value = "Workflow description"
        mock_wrapper = MagicMock()
        mock_create_wrapper.return_value = mock_wrapper

        # Act
        register_function_with_mcp(mock_mcp, function_name, mock_workflow, mock_workflow)

        # Assert - Check that logging happened (actual message order may vary)
        assert mock_logger.info.call_count >= 2  # Should log at least twice for workflow
        mock_get_desc.assert_called_once_with(mock_workflow)
        mock_create_wrapper.assert_called_once_with(function_name,
                                                    mock_workflow,
                                                    mock_workflow.input_schema,
                                                    True,
                                                    mock_workflow)
        mock_mcp.tool.assert_called_once_with(name=function_name, description="Workflow description")


class TestIntegrationScenarios:
    """Integration test scenarios combining multiple components."""

    @patch('nat.front_ends.mcp.tool_converter.ContextState')
    async def test_observability_context_propagation(self, mock_context_state_class):
        """Test that observability context is properly propagated."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        mock_function.acall_invoke = AsyncMock(return_value="result")

        mock_workflow = create_mock_workflow_with_observability()

        # Mock ContextState.get()
        mock_context_state = MagicMock()
        mock_context_state_class.get.return_value = mock_context_state

        # Create wrapper
        wrapper = create_function_wrapper("test_func", mock_function, MockRegularSchema, False, mock_workflow)

        # Act - Execute wrapper
        await wrapper(name="test", age=25)

        # Assert - Check that observability was started with correct context
        mock_workflow.exporter_manager.start.assert_called_once_with(context_state=mock_context_state)
        mock_context_state_class.get.assert_called_once()

    @patch('nat.front_ends.mcp.tool_converter.ContextState')
    async def test_error_handling_in_wrapper_execution(self, mock_context_state_class):
        """Test error handling during wrapper execution."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        mock_function.acall_invoke.side_effect = Exception("Test error")

        mock_workflow = create_mock_workflow_with_observability()

        # Mock ContextState.get()
        mock_context_state = MagicMock()
        mock_context_state_class.get.return_value = mock_context_state

        wrapper = create_function_wrapper("test_func", mock_function, MockRegularSchema, False, mock_workflow)

        # Act & Assert
        with pytest.raises(Exception, match="Test error"):
            await wrapper(name="test", age=25)

        # Observability context should still have been started
        mock_workflow.exporter_manager.start.assert_called_once_with(context_state=mock_context_state)
        mock_context_state_class.get.assert_called_once()
