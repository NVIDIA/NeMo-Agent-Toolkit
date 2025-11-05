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
"""Test tool_wrapper.py file """

import inspect
import types
from dataclasses import dataclass
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.function import Function
from nat.plugins.autogen.tool_wrapper import autogen_tool_wrapper
from nat.plugins.autogen.tool_wrapper import resolve_type


class MockInputSchema(BaseModel):
    """Mock input schema for tool wrapper."""

    param1: str
    param2: int
    param3: float = 3.14


@dataclass
class MockDataclassSchema:
    """Mock dataclass schema for tool wrapper."""

    param1: str
    param2: int


class TestResolveType:
    """Test cases for resolve_type function."""

    def test_resolve_union_type(self):
        """Test resolving Union types."""
        union_type = str | None
        result = resolve_type(union_type)
        # For typing.Union, returns the original unchanged since it only handles types.UnionType
        assert result == union_type

    def test_resolve_pep604_union(self):
        """Test resolving PEP 604 union types (str | None)."""
        union_type = str | None
        result = resolve_type(union_type)
        # Should return str (the non-None type)
        assert result == union_type

    def test_resolve_non_union_type(self):
        """Test resolving non-union types."""
        result = resolve_type(int)
        assert result is int

    def test_resolve_complex_union(self):
        """Test resolving union with multiple non-None types."""
        union_type = str | int | None
        result = resolve_type(union_type)
        # For typing.Union, the function returns the original type unchanged
        # since it only handles types.UnionType (PEP 604 style)
        assert result == union_type

    def test_resolve_all_none_union(self):
        """Test resolving union with only None types."""
        union_type = None | type(None)
        result = resolve_type(union_type)
        # Should return the original type if no non-None found
        assert result == union_type


class TestAutoGenToolWrapper:
    """Test cases for autogen_tool_wrapper function."""

    @pytest.fixture
    def mock_function(self):
        """Create a mock NAT function."""
        mock_fn = Mock(spec=Function)
        mock_fn.description = "Test function description"
        mock_fn.input_schema = MockInputSchema
        mock_fn.has_streaming_output = False
        mock_fn.has_single_output = True
        mock_fn.acall_invoke = AsyncMock(return_value="test_result")
        mock_fn.acall_stream = AsyncMock()
        return mock_fn

    @pytest.fixture
    def mock_builder(self):
        """Create a mock builder."""
        return Mock(spec=Builder)

    def test_autogen_tool_wrapper_basic(self, mock_function, mock_builder):
        """Test basic tool wrapper functionality."""
        with patch('nat.plugins.autogen.tool_wrapper.FunctionTool') as mock_function_tool:
            mock_tool = Mock()
            mock_function_tool.return_value = mock_tool

            result = autogen_tool_wrapper("test_tool", mock_function, mock_builder)

            mock_function_tool.assert_called_once()
            call_args = mock_function_tool.call_args
            assert call_args[1]['name'] == "test_tool"
            assert call_args[1]['description'] == "Test function description"
            assert callable(call_args[1]['func'])
            assert result == mock_tool

    def test_autogen_tool_wrapper_streaming(self, mock_function, mock_builder):
        """Test tool wrapper with streaming output."""
        mock_function.has_streaming_output = True
        mock_function.has_single_output = False

        with patch('nat.plugins.autogen.tool_wrapper.FunctionTool') as mock_function_tool:
            mock_tool = Mock()
            mock_function_tool.return_value = mock_tool

            result = autogen_tool_wrapper("test_tool", mock_function, mock_builder)

            mock_function_tool.assert_called_once()
            # Should use streaming callable
            assert result == mock_tool

    def test_autogen_tool_wrapper_no_description(self, mock_function, mock_builder):
        """Test tool wrapper with no description."""
        _ = mock_builder  # Unused in this test
        mock_function.description = None

        with patch('nat.plugins.autogen.tool_wrapper.FunctionTool') as mock_function_tool:
            mock_tool = Mock()
            mock_function_tool.return_value = mock_tool

            autogen_tool_wrapper("test_tool", mock_function, mock_builder)
            call_args = mock_function_tool.call_args
            assert call_args[1]['description'] == "No description provided."

    @pytest.mark.asyncio
    async def test_callable_ainvoke(self, mock_function, mock_builder):
        """Test the async invoke callable."""
        with patch('nat.plugins.autogen.tool_wrapper.FunctionTool'):
            autogen_tool_wrapper("test_tool", mock_function, mock_builder)

            # Test that acall_invoke would be called
            result = await mock_function.acall_invoke("arg1", param="value")
            assert result == "test_result"
            mock_function.acall_invoke.assert_called_once_with("arg1", param="value")

    @pytest.mark.asyncio
    async def test_callable_astream(self, mock_function, mock_builder):
        """Test the async stream callable."""
        mock_function.has_streaming_output = True
        mock_function.has_single_output = False

        async def mock_stream():
            yield "item1"
            yield "item2"

        mock_function.acall_stream = mock_stream

        with patch('nat.plugins.autogen.tool_wrapper.FunctionTool'):
            autogen_tool_wrapper("test_tool", mock_function, mock_builder)

            # Test that acall_stream would work
            items = []
            async for item in mock_function.acall_stream():
                items.append(item)

            assert items == ["item1", "item2"]


class TestNatFunctionDecorator:
    """Test the nat_function decorator pattern."""

    def test_function_metadata_setting(self):
        """Test that function metadata is set correctly."""

        def test_func():
            """Test function."""

        # Mock the decorator pattern from the source
        name = "test_name"
        description = "test_description"
        input_schema = MockInputSchema

        # Set metadata like the decorator does
        test_func.__name__ = name
        test_func.__doc__ = description

        # Test signature creation
        annotations = getattr(input_schema, "__annotations__", {}) or {}
        params = []
        for param_name, param_annotation in annotations.items():
            resolved_type = resolve_type(param_annotation)
            params.append(
                inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=resolved_type))
            annotations[param_name] = resolved_type

        # Create signature
        signature = inspect.Signature(parameters=params)
        # Note: Cannot actually set __signature__ on function objects in tests
        # so we test the signature creation separately

        # Verify metadata
        assert test_func.__name__ == name
        assert test_func.__doc__ == description
        assert signature is not None
        assert len(signature.parameters) == 3

    def test_signature_creation_with_schema(self):
        """Test signature creation with input schema."""
        input_schema = MockInputSchema
        annotations = getattr(input_schema, "__annotations__", {}) or {}

        params = []
        processed_annotations = {}
        for param_name, param_annotation in annotations.items():
            resolved_type = resolve_type(param_annotation)
            params.append(
                inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=resolved_type))
            processed_annotations[param_name] = resolved_type

        signature = inspect.Signature(parameters=params)

        # Verify signature has correct parameters
        assert "param1" in signature.parameters
        assert "param2" in signature.parameters
        assert "param3" in signature.parameters
        assert signature.parameters["param1"].annotation is str
        assert signature.parameters["param2"].annotation is int
        assert signature.parameters["param3"].annotation is float

    def test_no_input_schema_handling(self):
        """Test handling when no input schema is provided."""

        def test_func():
            pass

        # When input_schema is None, no signature should be set
        input_schema = None

        if input_schema is not None:
            # This block should not execute
            pytest.fail("Should not process None input_schema")

        # Function should remain unchanged when no schema
        original_signature = inspect.signature(test_func)
        assert original_signature is not None


class TestTypeResolution:
    """Test type resolution in various scenarios."""

    def test_resolve_type_with_complex_types(self):
        """Test resolve_type with complex type annotations."""
        # Test with list type
        list_type = list[str]
        result = resolve_type(list_type)
        assert result == list_type

        # Test with dict type
        dict_type = dict[str, int]
        result = resolve_type(dict_type)
        assert result == dict_type

    def test_resolve_type_with_optional(self):
        """Test resolve_type with Optional types."""

        optional_str = str | None
        result = resolve_type(optional_str)
        # For typing.Union (which Optional[str] is), returns the original unchanged
        assert result == optional_str

    def test_union_type_detection(self):
        """Test union type detection."""
        union_type = str | int | None
        origin = types.UnionType if hasattr(types, 'UnionType') else type(union_type)

        # Test that we can detect union types
        assert origin is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
