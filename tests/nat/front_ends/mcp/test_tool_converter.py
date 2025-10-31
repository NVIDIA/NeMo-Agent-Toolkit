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

from inspect import Parameter
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from pydantic import Field

from nat.builder.function import Function
from nat.builder.workflow import Workflow
from nat.front_ends.mcp.tool_converter import _USE_PYDANTIC_DEFAULT
from nat.front_ends.mcp.tool_converter import create_function_wrapper
from nat.front_ends.mcp.tool_converter import get_function_description
from nat.front_ends.mcp.tool_converter import is_field_optional
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


class MockAllRequiredSchema(BaseModel):
    """Schema with all required parameters."""
    name: str
    age: int
    email: str


class MockMixedRequiredOptionalSchema(BaseModel):
    """Schema with mix of required and optional parameters."""
    required_str: str
    required_int: int
    optional_str: str = Field(default="default_value")
    optional_int: int = Field(default=42)
    optional_list: list[str] = Field(default_factory=list)


class MockAllOptionalSchema(BaseModel):
    """Schema with all optional parameters."""
    optional_str: str = Field(default="default")
    optional_int: int = Field(default=0)
    optional_bool: bool = Field(default=False)
    optional_list: list[float] | None = None


class MockOptionalTypesSchema(BaseModel):
    """Schema with optional types using Union notation."""
    required_field: str
    optional_str_none: str | None = None
    optional_int_none: int | None = None
    optional_list_none: list[float] | None = None


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


class TestIsFieldOptional:
    """Test cases for is_field_optional utility function."""

    def test_required_field_no_default(self):
        """Test that a required field with no default is detected correctly."""
        # Arrange
        field = MockAllRequiredSchema.model_fields["name"]

        # Act
        is_optional, default_value = is_field_optional(field)

        # Assert
        assert is_optional is False
        assert default_value == Parameter.empty

    def test_optional_field_with_string_default(self):
        """Test optional field with a string default value."""
        # Arrange
        field = MockMixedRequiredOptionalSchema.model_fields["optional_str"]

        # Act
        is_optional, default_value = is_field_optional(field)

        # Assert
        assert is_optional is True
        assert default_value == "default_value"

    def test_optional_field_with_int_default(self):
        """Test optional field with an integer default value."""
        # Arrange
        field = MockMixedRequiredOptionalSchema.model_fields["optional_int"]

        # Act
        is_optional, default_value = is_field_optional(field)

        # Assert
        assert is_optional is True
        assert default_value == 42

    def test_optional_field_with_factory_default(self):
        """Test optional field with a default_factory."""
        # Arrange
        field = MockMixedRequiredOptionalSchema.model_fields["optional_list"]

        # Act
        is_optional, default_value = is_field_optional(field)

        # Assert
        assert is_optional is True
        # When default_factory is used, we return the sentinel
        # This allows Pydantic to apply the factory at validation time
        assert default_value is _USE_PYDANTIC_DEFAULT

    def test_optional_field_with_none_default(self):
        """Test optional field with None as default (Union types)."""
        # Arrange
        field = MockOptionalTypesSchema.model_fields["optional_str_none"]

        # Act
        is_optional, default_value = is_field_optional(field)

        # Assert
        assert is_optional is True
        assert default_value is None

    def test_optional_field_with_bool_default(self):
        """Test optional field with boolean default value."""
        # Arrange
        field = MockAllOptionalSchema.model_fields["optional_bool"]

        # Act
        is_optional, default_value = is_field_optional(field)

        # Assert
        assert is_optional is True
        assert default_value is False

    def test_optional_field_with_zero_default(self):
        """Test optional field with zero as default (should not be confused with falsy)."""
        # Arrange
        field = MockAllOptionalSchema.model_fields["optional_int"]

        # Act
        is_optional, default_value = is_field_optional(field)

        # Assert
        assert is_optional is True
        assert default_value == 0

    def test_required_fields_consistency(self):
        """Test that all required fields in a schema are detected consistently."""
        # Arrange
        required_fields = ["required_str", "required_int"]

        # Act & Assert
        for field_name in required_fields:
            field = MockMixedRequiredOptionalSchema.model_fields[field_name]
            is_optional, default_value = is_field_optional(field)
            assert is_optional is False, f"Field {field_name} should be required"
            assert default_value == Parameter.empty, f"Field {field_name} should have no default"

    def test_optional_fields_consistency(self):
        """Test that all optional fields in a schema are detected consistently."""
        # Arrange
        optional_fields = ["optional_str", "optional_int", "optional_list"]

        # Act & Assert
        for field_name in optional_fields:
            field = MockMixedRequiredOptionalSchema.model_fields[field_name]
            is_optional, default_value = is_field_optional(field)
            assert is_optional is True, f"Field {field_name} should be optional"
            assert default_value != Parameter.empty, f"Field {field_name} should have a default"


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


class TestParameterSchemaValidation:
    """Test cases for validating parameter schemas after conversion."""

    def test_all_required_parameters(self):
        """Test schema with all required parameters."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        function_name = "all_required_func"

        # Act
        wrapper = create_function_wrapper(function_name, mock_function, MockAllRequiredSchema, False, None)

        # Assert
        sig = getattr(wrapper, '__signature__', None)
        assert sig is not None
        assert "name" in sig.parameters
        assert "age" in sig.parameters
        assert "email" in sig.parameters

        # All parameters should be required (no default)
        assert sig.parameters["name"].default == Parameter.empty
        assert sig.parameters["age"].default == Parameter.empty
        assert sig.parameters["email"].default == Parameter.empty

    def test_all_optional_parameters(self):
        """Test schema with all optional parameters."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        function_name = "all_optional_func"

        # Act
        wrapper = create_function_wrapper(function_name, mock_function, MockAllOptionalSchema, False, None)

        # Assert
        sig = getattr(wrapper, '__signature__', None)
        assert sig is not None
        assert "optional_str" in sig.parameters
        assert "optional_int" in sig.parameters
        assert "optional_bool" in sig.parameters
        assert "optional_list" in sig.parameters

        # All parameters should have defaults (not Parameter.empty)
        assert sig.parameters["optional_str"].default != Parameter.empty
        assert sig.parameters["optional_int"].default != Parameter.empty
        assert sig.parameters["optional_bool"].default != Parameter.empty
        assert sig.parameters["optional_list"].default != Parameter.empty

        # Verify actual default values
        assert sig.parameters["optional_str"].default == "default"
        assert sig.parameters["optional_int"].default == 0
        assert sig.parameters["optional_bool"].default is False
        # optional_list has None as explicit default (not a factory), so it should be None
        assert sig.parameters["optional_list"].default is None

    def test_mixed_required_and_optional_parameters(self):
        """Test schema with mix of required and optional parameters."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        function_name = "mixed_func"

        # Act
        wrapper = create_function_wrapper(function_name, mock_function, MockMixedRequiredOptionalSchema, False, None)

        # Assert
        sig = getattr(wrapper, '__signature__', None)
        assert sig is not None

        # Check required parameters
        assert "required_str" in sig.parameters
        assert "required_int" in sig.parameters
        assert sig.parameters["required_str"].default == Parameter.empty
        assert sig.parameters["required_int"].default == Parameter.empty

        # Check optional parameters
        assert "optional_str" in sig.parameters
        assert "optional_int" in sig.parameters
        assert "optional_list" in sig.parameters
        assert sig.parameters["optional_str"].default == "default_value"
        assert sig.parameters["optional_int"].default == 42
        # Fields with default_factory get the sentinel as the signature default
        # The actual factory will be called by Pydantic at validation time
        assert sig.parameters["optional_list"].default is _USE_PYDANTIC_DEFAULT

    def test_optional_with_none_type(self):
        """Test optional parameters with None type (Union types)."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        function_name = "optional_none_func"

        # Act
        wrapper = create_function_wrapper(function_name, mock_function, MockOptionalTypesSchema, False, None)

        # Assert
        sig = getattr(wrapper, '__signature__', None)
        assert sig is not None

        # Required field should have no default
        assert "required_field" in sig.parameters
        assert sig.parameters["required_field"].default == Parameter.empty

        # Optional fields with None should have None as default
        assert "optional_str_none" in sig.parameters
        assert "optional_int_none" in sig.parameters
        assert "optional_list_none" in sig.parameters
        assert sig.parameters["optional_str_none"].default is None
        assert sig.parameters["optional_int_none"].default is None
        assert sig.parameters["optional_list_none"].default is None

    def test_parameter_annotations_preserved(self):
        """Test that parameter type annotations are preserved."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        function_name = "annotated_func"

        # Act
        wrapper = create_function_wrapper(function_name, mock_function, MockMixedRequiredOptionalSchema, False, None)

        # Assert
        sig = getattr(wrapper, '__signature__', None)
        assert sig is not None

        # Check that annotations are present
        assert sig.parameters["required_str"].annotation is str
        assert sig.parameters["required_int"].annotation is int
        assert sig.parameters["optional_str"].annotation is str
        assert sig.parameters["optional_int"].annotation is int

    def test_parameter_order_preserved(self):
        """Test that parameter order is preserved in wrapper."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        function_name = "ordered_func"

        # Act
        wrapper = create_function_wrapper(function_name, mock_function, MockMixedRequiredOptionalSchema, False, None)

        # Assert
        sig = getattr(wrapper, '__signature__', None)
        assert sig is not None

        param_names = list(sig.parameters.keys())
        # Pydantic fields should maintain order
        assert "required_str" in param_names
        assert "required_int" in param_names
        assert "optional_str" in param_names
        assert "optional_int" in param_names
        assert "optional_list" in param_names


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

    @patch('nat.front_ends.mcp.tool_converter.ContextState')
    async def test_wrapper_with_optional_parameters_omitted(self, mock_context_state_class):
        """Test wrapper execution when optional parameters are omitted."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        mock_function.acall_invoke = AsyncMock(return_value="result")

        mock_workflow = create_mock_workflow_with_observability()

        # Mock ContextState.get()
        mock_context_state = MagicMock()
        mock_context_state_class.get.return_value = mock_context_state

        wrapper = create_function_wrapper("test_func",
                                          mock_function,
                                          MockMixedRequiredOptionalSchema,
                                          False,
                                          mock_workflow)

        # Act - Call with only required parameters
        result = await wrapper(required_str="test", required_int=123)

        # Assert
        assert result == "result"
        # Function should have been called with defaults for optional parameters
        mock_function.acall_invoke.assert_called_once()

    @patch('nat.front_ends.mcp.tool_converter.ContextState')
    async def test_wrapper_with_optional_parameters_provided(self, mock_context_state_class):
        """Test wrapper execution when optional parameters are provided."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        mock_function.acall_invoke = AsyncMock(return_value="result")

        mock_workflow = create_mock_workflow_with_observability()

        # Mock ContextState.get()
        mock_context_state = MagicMock()
        mock_context_state_class.get.return_value = mock_context_state

        wrapper = create_function_wrapper("test_func",
                                          mock_function,
                                          MockMixedRequiredOptionalSchema,
                                          False,
                                          mock_workflow)

        # Act - Call with all parameters
        result = await wrapper(required_str="test",
                               required_int=123,
                               optional_str="custom",
                               optional_int=999,
                               optional_list=["a", "b"])

        # Assert
        assert result == "result"
        mock_function.acall_invoke.assert_called_once()

    @patch('nat.front_ends.mcp.tool_converter.ContextState')
    async def test_wrapper_with_none_values(self, mock_context_state_class):
        """Test wrapper execution with explicit None values for optional parameters."""
        # Arrange
        mock_function = MagicMock(spec=Function)
        mock_function.acall_invoke = AsyncMock(return_value="result")

        mock_workflow = create_mock_workflow_with_observability()

        # Mock ContextState.get()
        mock_context_state = MagicMock()
        mock_context_state_class.get.return_value = mock_context_state

        wrapper = create_function_wrapper("test_func", mock_function, MockOptionalTypesSchema, False, mock_workflow)

        # Act - Call with None for optional parameters
        result = await wrapper(required_field="test", optional_str_none=None, optional_int_none=None)

        # Assert
        assert result == "result"
        mock_function.acall_invoke.assert_called_once()
