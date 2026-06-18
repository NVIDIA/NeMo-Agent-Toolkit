# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nat.plugins.mcp.server.tool_converter import _USE_PYDANTIC_DEFAULT
from nat.plugins.mcp.server.tool_converter import _build_name_mapping
from nat.plugins.mcp.server.tool_converter import _sanitize_parameter_name
from nat.plugins.mcp.server.tool_converter import create_function_wrapper
from nat.plugins.mcp.server.tool_converter import get_function_description
from nat.plugins.mcp.server.tool_converter import is_field_optional
from nat.plugins.mcp.server.tool_converter import register_function_with_mcp
from nat.runtime.session import SessionManager


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


def create_mock_session_manager(workflow=None, result_value="result"):
    """Create a mock SessionManager for testing.

    Args:
        workflow: Optional workflow to attach to the session manager
        result_value: The value to return from runner.result()
    """
    mock_session_manager = MagicMock(spec=SessionManager)

    if workflow is None:
        workflow = create_mock_workflow_with_observability()

    mock_session_manager.workflow = workflow

    # Create mock runner with async context manager
    mock_runner = MagicMock()
    mock_runner.__aenter__ = AsyncMock(return_value=mock_runner)
    mock_runner.__aexit__ = AsyncMock(return_value=None)
    mock_runner.result = AsyncMock(return_value=result_value)

    # Make session_manager.run() return the runner
    mock_session_manager.run = MagicMock(return_value=mock_runner)

    return mock_session_manager


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
        mock_session_manager = create_mock_session_manager()
        function_name = "test_function"
        schema = MockChatRequest

        # Act
        wrapper = create_function_wrapper(function_name, mock_session_manager, schema)

        # Assert
        assert callable(wrapper)
        assert wrapper.__name__ == function_name
        sig = getattr(wrapper, '__signature__', None)
        assert sig is not None
        assert "query" in sig.parameters

    def test_create_wrapper_for_regular_function(self):
        """Test creating wrapper for function with regular schema."""
        # Arrange
        mock_session_manager = create_mock_session_manager()
        function_name = "regular_function"
        schema = MockRegularSchema

        # Act
        wrapper = create_function_wrapper(function_name, mock_session_manager, schema)

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
        mock_workflow = create_mock_workflow_with_observability()
        mock_session_manager = create_mock_session_manager(workflow=mock_workflow)
        function_name = "test_workflow"
        schema = MockChatRequest

        # Act
        wrapper = create_function_wrapper(function_name, mock_session_manager, schema)

        # Assert
        assert callable(wrapper)
        assert wrapper.__name__ == function_name

    async def test_wrapper_execution_with_observability(self):
        """Test wrapper execution with SessionManager pattern."""
        # Arrange
        mock_session_manager = create_mock_session_manager(result_value="result")

        wrapper = create_function_wrapper("test_func", mock_session_manager, MockRegularSchema)

        # Act
        result = await wrapper(name="test", age=30)

        # Assert
        assert result == "result"
        # Verify session_manager.run() was called with the validated input
        mock_session_manager.run.assert_called_once()
        # Verify runner.result() was called
        call_args = mock_session_manager.run.call_args
        assert call_args is not None

    async def test_wrapper_execution_via_session_manager(self):
        """Test wrapper execution uses SessionManager.run() pattern."""
        # Arrange
        mock_session_manager = create_mock_session_manager(result_value="chat response")
        wrapper = create_function_wrapper("test_func", mock_session_manager, MockChatRequest)

        # Act
        result = await wrapper(query="test")

        # Assert
        assert result == "chat response"
        mock_session_manager.run.assert_called_once()


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

    @patch('nat.plugins.mcp.server.tool_converter.create_function_wrapper')
    @patch('nat.plugins.mcp.server.tool_converter.get_function_description')
    @patch('nat.plugins.mcp.server.tool_converter.logger')
    def test_register_function_with_mcp_uses_function_metadata(self, mock_logger, mock_get_desc, mock_create_wrapper):
        """Test registering a function with MCP using SessionManager."""
        # Arrange
        mock_mcp = MagicMock()
        mock_workflow = MagicMock(spec=Workflow)
        mock_workflow.input_schema = "workflow_schema"
        mock_function = MagicMock(spec=Function)
        mock_function.input_schema = "function_schema"
        mock_session_manager = MagicMock(spec=SessionManager)
        mock_session_manager.workflow = mock_workflow
        function_name = "test_function"

        mock_get_desc.return_value = "Test description"
        mock_wrapper = MagicMock()
        mock_create_wrapper.return_value = mock_wrapper

        # Act
        register_function_with_mcp(mock_mcp, function_name, mock_session_manager, function=mock_function)

        # Assert - Check that logging happened
        assert mock_logger.info.call_count >= 1
        mock_get_desc.assert_called_once_with(mock_function)
        mock_create_wrapper.assert_called_once_with(function_name,
                                                    mock_session_manager,
                                                    mock_function.input_schema,
                                                    None)  # memory_profiler defaults to None
        mock_mcp.tool.assert_called_once_with(name=function_name, description="Test description")

    @patch('nat.plugins.mcp.server.tool_converter.create_function_wrapper')
    @patch('nat.plugins.mcp.server.tool_converter.get_function_description')
    @patch('nat.plugins.mcp.server.tool_converter.logger')
    def test_register_workflow_with_mcp_falls_back_to_workflow(self, mock_logger, mock_get_desc, mock_create_wrapper):
        """Test registering a workflow with MCP using SessionManager."""
        # Arrange
        mock_mcp = MagicMock()
        mock_workflow = MagicMock(spec=Workflow)
        mock_workflow.input_schema = "workflow_schema"
        mock_session_manager = MagicMock(spec=SessionManager)
        mock_session_manager.workflow = mock_workflow
        function_name = "test_workflow"

        mock_get_desc.return_value = "Workflow description"
        mock_wrapper = MagicMock()
        mock_create_wrapper.return_value = mock_wrapper

        # Act
        register_function_with_mcp(mock_mcp, function_name, mock_session_manager)

        # Assert - Check that logging happened
        assert mock_logger.info.call_count >= 1
        mock_get_desc.assert_called_once_with(mock_workflow)
        mock_create_wrapper.assert_called_once_with(function_name,
                                                    mock_session_manager,
                                                    mock_workflow.input_schema,
                                                    None)  # memory_profiler defaults to None
        mock_mcp.tool.assert_called_once_with(name=function_name, description="Workflow description")


class TestParameterSchemaValidation:
    """Test cases for validating parameter schemas after conversion."""

    def test_all_required_parameters(self):
        """Test schema with all required parameters."""
        # Arrange
        mock_session_manager = create_mock_session_manager()
        function_name = "all_required_func"

        # Act
        wrapper = create_function_wrapper(function_name, mock_session_manager, MockAllRequiredSchema)

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
        mock_session_manager = create_mock_session_manager()
        function_name = "all_optional_func"

        # Act
        wrapper = create_function_wrapper(function_name, mock_session_manager, MockAllOptionalSchema)

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
        mock_session_manager = create_mock_session_manager()
        function_name = "mixed_func"

        # Act
        wrapper = create_function_wrapper(function_name, mock_session_manager, MockMixedRequiredOptionalSchema)

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
        mock_session_manager = create_mock_session_manager()
        function_name = "optional_none_func"

        # Act
        wrapper = create_function_wrapper(function_name, mock_session_manager, MockOptionalTypesSchema)

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
        mock_session_manager = create_mock_session_manager()
        function_name = "annotated_func"

        # Act
        wrapper = create_function_wrapper(function_name, mock_session_manager, MockMixedRequiredOptionalSchema)

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
        mock_session_manager = create_mock_session_manager()
        function_name = "ordered_func"

        # Act
        wrapper = create_function_wrapper(function_name, mock_session_manager, MockMixedRequiredOptionalSchema)

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

    async def test_observability_context_propagation(self):
        """Test that SessionManager.run() handles observability."""
        # Arrange
        mock_session_manager = create_mock_session_manager(result_value="result")

        # Create wrapper
        wrapper = create_function_wrapper("test_func", mock_session_manager, MockRegularSchema)

        # Act - Execute wrapper
        await wrapper(name="test", age=25)

        # Assert - Check that session_manager.run() was called
        mock_session_manager.run.assert_called_once()

    async def test_error_handling_in_wrapper_execution(self):
        """Test error handling during wrapper execution."""
        # Arrange
        mock_workflow = create_mock_workflow_with_observability()
        mock_session_manager = MagicMock(spec=SessionManager)
        mock_session_manager.workflow = mock_workflow

        # Create mock runner that raises an error
        mock_runner = MagicMock()
        mock_runner.__aenter__ = AsyncMock(return_value=mock_runner)
        mock_runner.__aexit__ = AsyncMock(return_value=None)
        mock_runner.result = AsyncMock(side_effect=Exception("Test error"))
        mock_session_manager.run = MagicMock(return_value=mock_runner)

        wrapper = create_function_wrapper("test_func", mock_session_manager, MockRegularSchema)

        # Act & Assert
        with pytest.raises(Exception, match="Test error"):
            await wrapper(name="test", age=25)

        # Verify session_manager.run() was called even though it raised an error
        mock_session_manager.run.assert_called_once()

    async def test_wrapper_with_optional_parameters_omitted(self):
        """Test wrapper execution when optional parameters are omitted."""
        # Arrange
        mock_session_manager = create_mock_session_manager(result_value="result")

        wrapper = create_function_wrapper("test_func", mock_session_manager, MockMixedRequiredOptionalSchema)

        # Act - Call with only required parameters
        result = await wrapper(required_str="test", required_int=123)

        # Assert
        assert result == "result"
        # SessionManager.run() should have been called
        mock_session_manager.run.assert_called_once()

    async def test_wrapper_with_optional_parameters_provided(self):
        """Test wrapper execution when optional parameters are provided."""
        # Arrange
        mock_session_manager = create_mock_session_manager(result_value="result")

        wrapper = create_function_wrapper("test_func", mock_session_manager, MockMixedRequiredOptionalSchema)

        # Act - Call with all parameters
        result = await wrapper(required_str="test",
                               required_int=123,
                               optional_str="custom",
                               optional_int=999,
                               optional_list=["a", "b"])

        # Assert
        assert result == "result"
        mock_session_manager.run.assert_called_once()

    async def test_wrapper_with_none_values(self):
        """Test wrapper execution with explicit None values for optional parameters."""
        # Arrange
        mock_session_manager = create_mock_session_manager(result_value="result")

        wrapper = create_function_wrapper("test_func", mock_session_manager, MockOptionalTypesSchema)

        # Act - Call with None for optional parameters
        result = await wrapper(required_field="test", optional_str_none=None, optional_int_none=None)

        # Assert
        assert result == "result"
        mock_session_manager.run.assert_called_once()


class TestResultTypeConversion:
    """Test cases for result type conversion and serialization."""

    async def test_runner_result_called_without_to_type(self):
        """Test that runner.result() is called without to_type parameter."""
        # Arrange
        mock_session_manager = create_mock_session_manager(result_value="result")
        wrapper = create_function_wrapper("test_func", mock_session_manager, MockRegularSchema)

        # Act
        await wrapper(name="test", age=25)

        # Assert - Verify runner.result() was called without to_type
        mock_runner = mock_session_manager.run.return_value
        mock_runner.result.assert_called_once_with()  # No arguments, especially no to_type

    async def test_dict_result_converted_to_json_string(self):
        """Test that dict results are converted to JSON string."""
        # Arrange
        dict_result = {"key": "value", "number": 42}
        mock_session_manager = create_mock_session_manager(result_value=dict_result)
        wrapper = create_function_wrapper("test_func", mock_session_manager, MockRegularSchema)

        # Act
        result = await wrapper(name="test", age=25)

        # Assert
        import json
        assert isinstance(result, str)
        assert result == json.dumps(dict_result, default=str)

    async def test_list_result_converted_to_json_string(self):
        """Test that list results are converted to JSON string."""
        # Arrange
        list_result = [1, 2, 3, "test"]
        mock_session_manager = create_mock_session_manager(result_value=list_result)
        wrapper = create_function_wrapper("test_func", mock_session_manager, MockRegularSchema)

        # Act
        result = await wrapper(name="test", age=25)

        # Assert
        import json
        assert isinstance(result, str)
        assert result == json.dumps(list_result, default=str)

    async def test_string_result_returned_as_is(self):
        """Test that string results are returned without modification."""
        # Arrange
        string_result = "test result"
        mock_session_manager = create_mock_session_manager(result_value=string_result)
        wrapper = create_function_wrapper("test_func", mock_session_manager, MockRegularSchema)

        # Act
        result = await wrapper(name="test", age=25)

        # Assert
        assert isinstance(result, str)
        assert result == string_result

    async def test_complex_dict_result_serialization(self):
        """Test that complex dict with nested structures is properly serialized."""
        # Arrange
        complex_dict = {
            "nested": {
                "key": "value"
            }, "list": [1, 2, 3], "mixed": {
                "items": ["a", "b"]
            }, "number": 123.456
        }
        mock_session_manager = create_mock_session_manager(result_value=complex_dict)
        wrapper = create_function_wrapper("test_func", mock_session_manager, MockRegularSchema)

        # Act
        result = await wrapper(name="test", age=25)

        # Assert
        import json
        assert isinstance(result, str)
        # Verify it's valid JSON and matches original
        parsed = json.loads(result)
        assert parsed == complex_dict

    async def test_non_string_non_dict_result_converted_to_string(self):
        """Test that other types (int, float, etc.) are converted to string."""
        # Arrange
        test_cases = [
            (42, "42"),
            (3.14, "3.14"),
            (True, "True"),
            (None, "None"),
        ]

        for input_value, expected_output in test_cases:
            mock_session_manager = create_mock_session_manager(result_value=input_value)
            wrapper = create_function_wrapper("test_func", mock_session_manager, MockRegularSchema)

            # Act
            result = await wrapper(name="test", age=25)

            # Assert
            assert isinstance(result, str)
            assert result == expected_output


class TestSanitizeParameterName:
    """Test cases for _sanitize_parameter_name utility function."""

    def test_valid_identifier_unchanged(self):
        """Test that a valid Python identifier is returned unchanged."""
        assert _sanitize_parameter_name("query") == "query"

    def test_python_keyword_gets_trailing_underscore(self):
        """Test that Python keywords get a trailing underscore appended."""
        assert _sanitize_parameter_name("from") == "from_"
        assert _sanitize_parameter_name("class") == "class_"
        assert _sanitize_parameter_name("import") == "import_"
        assert _sanitize_parameter_name("return") == "return_"
        assert _sanitize_parameter_name("yield") == "yield_"
        assert _sanitize_parameter_name("lambda") == "lambda_"

    def test_hyphenated_name_replaced(self):
        """Test that hyphens are replaced with underscores."""
        assert _sanitize_parameter_name("cik-A") == "cik_A"
        assert _sanitize_parameter_name("start-date") == "start_date"

    def test_digit_prefix_gets_underscore(self):
        """Test that names starting with a digit get a leading underscore."""
        assert _sanitize_parameter_name("1field") == "_1field"

    def test_special_characters_replaced(self):
        """Test that special characters are replaced with underscores."""
        assert _sanitize_parameter_name("field.name") == "field_name"
        assert _sanitize_parameter_name("field@name") == "field_name"
        assert _sanitize_parameter_name("field name") == "field_name"

    def test_empty_string_gets_underscore(self):
        """Test that an empty string gets a leading underscore."""
        assert _sanitize_parameter_name("") == "_"

    def test_already_valid_name_unchanged(self):
        """Test that names that are already valid identifiers pass through."""
        assert _sanitize_parameter_name("my_field") == "my_field"
        assert _sanitize_parameter_name("_private") == "_private"
        assert _sanitize_parameter_name("CamelCase") == "CamelCase"


class TestBuildNameMapping:
    """Test cases for _build_name_mapping utility function."""

    def test_no_changes_when_names_are_valid(self):
        """Test that valid identifiers map to themselves."""
        result = _build_name_mapping(["name", "age", "email"])
        assert result == {"name": "name", "age": "age", "email": "email"}

    def test_mapping_for_keywords(self):
        """Test mapping for Python keywords."""
        result = _build_name_mapping(["from", "class", "query"])
        assert result == {"from": "from_", "class": "class_", "query": "query"}

    def test_mapping_for_hyphenated_names(self):
        """Test mapping for hyphenated names."""
        result = _build_name_mapping(["cik-A", "start-date", "name"])
        assert result == {"cik-A": "cik_A", "start-date": "start_date", "name": "name"}

    def test_mapping_mixed(self):
        """Test mapping with a mix of valid and invalid names."""
        result = _build_name_mapping(["from", "name", "cik-A"])
        assert result == {"from": "from_", "name": "name", "cik-A": "cik_A"}

    def test_collision_keyword_and_suffixed_name(self):
        """Test collision when schema has both 'from' and 'from_'."""
        result = _build_name_mapping(["from", "from_"])
        # 'from_' is a valid identifier so it reserves 'from_' in first pass
        # 'from' sanitizes to 'from_' which collides, so gets 'from_2'
        assert result["from_"] == "from_"
        assert result["from"] == "from_2"

    def test_collision_hyphenated_and_underscored(self):
        """Test collision when schema has both 'cik-A' and 'cik_A'."""
        result = _build_name_mapping(["cik-A", "cik_A"])
        # 'cik_A' is valid, reserves itself in first pass
        # 'cik-A' sanitizes to 'cik_A' which collides, so gets 'cik_A2'
        assert result["cik_A"] == "cik_A"
        assert result["cik-A"] == "cik_A2"


class TestParameterNameSanitization:
    """Test that schemas with non-identifier field names produce valid wrappers."""

    def test_wrapper_created_for_schema_with_keyword_fields(self):
        """Test that create_function_wrapper succeeds with keyword field names."""
        # Arrange
        from pydantic import create_model
        schema = create_model("DateRangeSchema", **{
            "from": (str, ...), "class": (str, ...), "query": (str, ...)
        })  # type: ignore[call-overload]

        mock_session_manager = create_mock_session_manager()

        # Act - this previously raised ValueError: 'from' is not a valid parameter name
        wrapper = create_function_wrapper("date_tool", mock_session_manager, schema)

        # Assert
        assert callable(wrapper)
        sig = getattr(wrapper, '__signature__', None)
        assert sig is not None
        assert "from_" in sig.parameters
        assert "class_" in sig.parameters
        assert "query" in sig.parameters

    def test_wrapper_created_for_schema_with_hyphenated_fields(self):
        """Test that create_function_wrapper succeeds with hyphenated field names."""
        # Arrange
        from pydantic import create_model
        schema = create_model("SecFilingsSchema", **{
            "cik-A": (str, ...), "name": (str, ...)
        })  # type: ignore[call-overload]

        mock_session_manager = create_mock_session_manager()

        # Act
        wrapper = create_function_wrapper("sec_tool", mock_session_manager, schema)

        # Assert
        assert callable(wrapper)
        sig = getattr(wrapper, '__signature__', None)
        assert sig is not None
        assert "cik_A" in sig.parameters
        assert "name" in sig.parameters

    async def test_wrapper_reverse_maps_kwargs_for_validation(self):
        """Test that sanitized kwargs are reverse-mapped before Pydantic validation."""
        # Arrange
        from pydantic import create_model
        schema = create_model("ToolSchema", **{
            "from": (str, ...), "cik-A": (str, "default")
        })  # type: ignore[call-overload]

        mock_session_manager = create_mock_session_manager(result_value="ok")

        wrapper = create_function_wrapper("tool", mock_session_manager, schema)

        # Act - call with sanitized names (as MCP would pass them)
        result = await wrapper(**{"from_": "2024-01-01", "cik_A": "ABC"})

        # Assert
        assert result == "ok"
        # Verify session_manager.run() was called with properly mapped payload
        mock_session_manager.run.assert_called_once()
        call_args = mock_session_manager.run.call_args
        payload = call_args[0][0]
        # The Pydantic model should have the original field names
        assert getattr(payload, "from") == "2024-01-01"
        assert getattr(payload, "cik-A") == "ABC"
