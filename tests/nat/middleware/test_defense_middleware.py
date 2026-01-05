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
"""Tests for DefenseMiddleware base class and field extraction logic."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from jsonpath_ng import parse
from pydantic import BaseModel

from nat.middleware.defense.defense_middleware import DefenseMiddleware
from nat.middleware.defense.defense_middleware import DefenseMiddlewareConfig
from nat.middleware.defense.defense_middleware import MultipleTargetFieldMatchesError
from nat.middleware.middleware import FunctionMiddlewareContext


class _TestOutputModel(BaseModel):
    """Test output model."""
    result: float
    operation: str
    message: str


class _TestDefenseMiddleware(DefenseMiddleware):
    """Concrete implementation for testing base class methods."""

    async def function_middleware_invoke(self, value, call_next, context):
        """Test implementation that extracts fields from output and stores them for verification."""
        # Check if defense should apply
        if not self._should_apply_defense(context.name):
            return await call_next(value)

        # Call next to get output
        output = await call_next(value)

        # Extract field from output if target_field is specified
        content, field_info = self._extract_field_from_value(output)

        # Store extracted content for test verification
        self._last_extracted_content = content
        self._last_field_info = field_info

        # Return output
        return output

    async def function_middleware_stream(self, value, call_next, _context):
        """Dummy implementation."""
        async for item in call_next(value):
            yield item

    def __init__(self, config: DefenseMiddlewareConfig, builder):
        """Initialize test middleware."""
        super().__init__(config, builder)
        self._last_extracted_content = None
        self._last_field_info = None


@pytest.fixture(name="mock_builder")
def fixture_mock_builder():
    """Create a mock builder."""
    return MagicMock()


class _TestInput(BaseModel):
    """Test input model."""
    value: float


@pytest.fixture(name="middleware_context")
def fixture_middleware_context():
    """Create a test FunctionMiddlewareContext."""
    return FunctionMiddlewareContext(name="my_calculator.multiply",
                                     config=MagicMock(),
                                     description="Test function",
                                     input_schema=_TestInput,
                                     single_output_schema=_TestOutputModel,
                                     stream_output_schema=type(None))


class TestDefenseMiddlewareTargeting:
    """Test defense middleware targeting logic."""

    def test_targeting_all_functions(self, mock_builder):
        """Test that defense applies to all functions when target is None."""
        config = DefenseMiddlewareConfig(target_function_or_group=None)
        middleware = _TestDefenseMiddleware(config, mock_builder)

        assert middleware._should_apply_defense("any_function") is True
        assert middleware._should_apply_defense("my_calculator.add") is True
        assert middleware._should_apply_defense("other_group.func") is True

    def test_targeting_specific_group(self, mock_builder):
        """Test targeting a specific function group."""
        config = DefenseMiddlewareConfig(target_function_or_group="my_calculator")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        assert middleware._should_apply_defense("my_calculator.multiply") is True
        assert middleware._should_apply_defense("my_calculator.add") is True
        assert middleware._should_apply_defense("other_calculator.add") is False
        assert middleware._should_apply_defense("my_calculator") is True

    def test_targeting_specific_function(self, mock_builder):
        """Test targeting a specific function."""
        config = DefenseMiddlewareConfig(target_function_or_group="my_calculator.multiply")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        assert middleware._should_apply_defense("my_calculator.multiply") is True
        assert middleware._should_apply_defense("my_calculator.add") is False
        assert middleware._should_apply_defense("other_calculator.multiply") is False

    def test_targeting_workflow(self, mock_builder):
        """Test targeting workflow-level functions."""
        config = DefenseMiddlewareConfig(target_function_or_group="<workflow>")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        assert middleware._should_apply_defense("<workflow>") is True
        assert middleware._should_apply_defense("my_calculator.multiply") is False

        # Also test "workflow" as target
        config2 = DefenseMiddlewareConfig(target_function_or_group="workflow")
        middleware2 = _TestDefenseMiddleware(config2, mock_builder)

        assert middleware2._should_apply_defense("<workflow>") is True
        assert middleware2._should_apply_defense("my_calculator.multiply") is False


class TestDefenseMiddlewareFieldExtraction:
    """Test field extraction logic with different value types and JSONPath expressions."""

    def test_extract_simple_type_no_target_field(self, mock_builder):
        """Test extracting from simple type without target_field."""
        config = DefenseMiddlewareConfig(target_field=None)
        middleware = _TestDefenseMiddleware(config, mock_builder)

        value = "simple string"
        content, field_info = middleware._extract_field_from_value(value)
        assert content == "simple string"
        assert field_info is None

    def test_extract_simple_type_with_target_field(self, mock_builder):
        """Test that target_field is ignored for simple types."""
        config = DefenseMiddlewareConfig(target_field="$.result")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        value = 42.0
        content, field_info = middleware._extract_field_from_value(value)
        assert content == 42.0
        assert field_info is None  # Simple types don't support field extraction

    def test_extract_dict_no_target_field(self, mock_builder):
        """Test extracting from dict without target_field."""
        config = DefenseMiddlewareConfig(target_field=None)
        middleware = _TestDefenseMiddleware(config, mock_builder)

        value = {"result": 42.0, "operation": "multiply"}
        content, field_info = middleware._extract_field_from_value(value)
        assert content == value
        assert field_info is None

    def test_extract_dict_simple_field(self, mock_builder):
        """Test extracting simple field from dict."""
        config = DefenseMiddlewareConfig(target_field="$.result")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        value = {"result": 42.0, "operation": "multiply", "message": "Success"}
        content, field_info = middleware._extract_field_from_value(value)
        assert content == 42.0
        assert field_info is not None
        assert field_info["target_field"] == "$.result"
        assert field_info["original_value"] == value

    def test_extract_dict_nested_field(self, mock_builder):
        """Test extracting nested field from dict."""
        config = DefenseMiddlewareConfig(target_field="$.data.message")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        value = {"data": {"message": "Hello", "status": "ok"}, "result": 42.0}
        content, field_info = middleware._extract_field_from_value(value)
        assert content == "Hello"
        assert field_info is not None
        assert field_info["target_field"] == "$.data.message"

    def test_extract_list_index(self, mock_builder):
        """Test extracting list element by index."""
        config = DefenseMiddlewareConfig(target_field="[0]")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        value = ["first", "second", "third"]
        content, field_info = middleware._extract_field_from_value(value)
        assert content == "first"
        assert field_info is not None

    def test_extract_list_field(self, mock_builder):
        """Test extracting field from list element."""
        config = DefenseMiddlewareConfig(target_field="$.numbers[0]")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        value = {"numbers": [10, 20, 30], "operation": "sum"}
        content, field_info = middleware._extract_field_from_value(value)
        assert content == 10
        assert field_info is not None

    def test_extract_basemodel_field(self, mock_builder):
        """Test extracting field from BaseModel."""
        config = DefenseMiddlewareConfig(target_field="$.result")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        value = _TestOutputModel(result=42.0, operation="multiply", message="Success")
        content, field_info = middleware._extract_field_from_value(value)
        assert content == 42.0
        assert field_info is not None
        assert field_info["is_basemodel"] is True
        assert field_info["original_type"] == _TestOutputModel

    def test_extract_no_match(self, mock_builder):
        """Test extracting field that doesn't exist."""
        config = DefenseMiddlewareConfig(target_field="$.nonexistent")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        value = {"result": 42.0, "operation": "multiply"}
        content, field_info = middleware._extract_field_from_value(value)
        # Should return original value when no match found
        assert content == value
        assert field_info is None


class TestDefenseMiddlewareFieldResolutionStrategy:
    """Test multiple field match resolution strategies."""

    def test_resolution_strategy_error(self, mock_builder):
        """Test error strategy raises ValueError on multiple matches."""
        config = DefenseMiddlewareConfig(target_field="$.result", target_field_resolution_strategy="error")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        # Create mock matches
        match1 = MagicMock()
        match1.value = "first"
        match2 = MagicMock()
        match2.value = "second"
        matches = [match1, match2]

        with pytest.raises(MultipleTargetFieldMatchesError, match="Multiple matches found"):
            middleware._resolve_multiple_field_matches(matches)

    def test_resolution_strategy_first(self, mock_builder):
        """Test first strategy returns first match."""
        config = DefenseMiddlewareConfig(target_field="$.result", target_field_resolution_strategy="first")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        match1 = MagicMock()
        match1.value = "first"
        match2 = MagicMock()
        match2.value = "second"
        matches = [match1, match2]

        result = middleware._resolve_multiple_field_matches(matches)
        assert len(result) == 1
        assert result[0].value == "first"

    def test_resolution_strategy_last(self, mock_builder):
        """Test last strategy returns last match."""
        config = DefenseMiddlewareConfig(target_field="$.result", target_field_resolution_strategy="last")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        match1 = MagicMock()
        match1.value = "first"
        match2 = MagicMock()
        match2.value = "second"
        matches = [match1, match2]

        result = middleware._resolve_multiple_field_matches(matches)
        assert len(result) == 1
        assert result[0].value == "second"

    def test_resolution_strategy_random(self, mock_builder):
        """Test random strategy returns one random match."""
        config = DefenseMiddlewareConfig(target_field="$.result", target_field_resolution_strategy="random")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        match1 = MagicMock()
        match1.value = "first"
        match2 = MagicMock()
        match2.value = "second"
        matches = [match1, match2]

        result = middleware._resolve_multiple_field_matches(matches)
        assert len(result) == 1
        assert result[0].value in ["first", "second"]

    def test_resolution_strategy_all(self, mock_builder):
        """Test all strategy returns all matches."""
        config = DefenseMiddlewareConfig(target_field="$.result", target_field_resolution_strategy="all")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        match1 = MagicMock()
        match1.value = "first"
        match2 = MagicMock()
        match2.value = "second"
        matches = [match1, match2]

        result = middleware._resolve_multiple_field_matches(matches)
        assert len(result) == 2
        assert result[0].value == "first"
        assert result[1].value == "second"


class TestDefenseMiddlewareFieldApplication:
    """Test applying analysis results back to original values."""

    def test_apply_result_single_match(self, mock_builder):
        """Test applying result to single field match."""
        config = DefenseMiddlewareConfig(target_field="$.result")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        original_value = {"result": 42.0, "operation": "multiply"}
        # Use real JSONPath matches
        jsonpath_expr = parse("$.result")
        matches = jsonpath_expr.find(original_value)

        field_info = {
            "target_field": "$.result",
            "matches": matches,
            "original_value": original_value,
            "is_basemodel": False,
            "original_type": dict
        }

        # Apply sanitized result
        sanitized_result = 4.0
        result = middleware._apply_field_result_to_value(original_value, field_info, sanitized_result)

        assert result == {"result": 4.0, "operation": "multiply"}

    def test_apply_result_multiple_matches_all_strategy(self, mock_builder):
        """Test applying result to multiple matches with all strategy."""
        config = DefenseMiddlewareConfig(target_field="$.results[*]", target_field_resolution_strategy="all")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        original_value = {"results": [42.0, 43.0], "operation": "multiply"}
        # Use real JSONPath matches
        jsonpath_expr = parse("$.results[*]")
        matches = jsonpath_expr.find(original_value)

        field_info = {
            "target_field": "$.results[*]",
            "matches": matches,
            "original_value": original_value,
            "is_basemodel": False,
            "original_type": dict
        }

        # Apply sanitized results (list for multiple matches)
        sanitized_results = [4.0, 5.0]
        result = middleware._apply_field_result_to_value(original_value, field_info, sanitized_results)

        assert result == {"results": [4.0, 5.0], "operation": "multiply"}

    def test_apply_result_basemodel(self, mock_builder):
        """Test applying result to BaseModel."""
        config = DefenseMiddlewareConfig(target_field="$.result")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        original_value = _TestOutputModel(result=42.0, operation="multiply", message="Success")
        # Use real JSONPath matches on the dict representation
        value_dict = original_value.model_dump()
        jsonpath_expr = parse("$.result")
        matches = jsonpath_expr.find(value_dict)

        field_info = {
            "target_field": "$.result",
            "matches": matches,
            "original_value": original_value,
            "is_basemodel": True,
            "original_type": _TestOutputModel
        }

        sanitized_result = 4.0
        result = middleware._apply_field_result_to_value(original_value, field_info, sanitized_result)

        # Should return BaseModel instance
        assert isinstance(result, _TestOutputModel)
        assert result.result == 4.0
        assert result.operation == "multiply"
        assert result.message == "Success"

    def test_apply_result_no_field_info(self, mock_builder):
        """Test applying result when no field_info (no targeting)."""
        config = DefenseMiddlewareConfig(target_field=None)
        middleware = _TestDefenseMiddleware(config, mock_builder)

        original_value = {"result": 42.0}
        sanitized_result = {"result": 4.0}

        # When no field_info, should return sanitized_result directly
        # Note: _apply_field_result_to_value expects field_info to be dict or None
        # Passing None is valid - it means no field extraction was done
        result = middleware._apply_field_result_to_value(original_value, None,
                                                         sanitized_result)  # type: ignore[arg-type]

        assert result == sanitized_result


class TestDefenseMiddlewareEndToEnd:
    """End-to-end tests for defense middleware field extraction and targeting."""

    async def test_extract_nested_output_field(self, mock_builder):
        """Test extracting nested field from output in actual invoke scenario."""
        config = DefenseMiddlewareConfig(target_field="$.result", target_function_or_group="my_calculator.multiply")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        output_value = _TestOutputModel(result=42.0, operation="multiply", message="Success")
        mock_call_next = AsyncMock(return_value=output_value)

        context = FunctionMiddlewareContext(name="my_calculator.multiply",
                                            config=MagicMock(),
                                            description="Multiply",
                                            input_schema=_TestInput,
                                            single_output_schema=_TestOutputModel,
                                            stream_output_schema=type(None))

        await middleware.function_middleware_invoke(10.0, mock_call_next, context)

        # Verify field was extracted
        assert middleware._last_extracted_content == 42.0
        assert middleware._last_field_info is not None
        assert middleware._last_field_info["target_field"] == "$.result"

    async def test_extract_deeply_nested_field(self, mock_builder):
        """Test extracting deeply nested field using JSONPath."""

        class NestedOutput(BaseModel):
            data: dict
            status: str

        config = DefenseMiddlewareConfig(target_field="$.data.message.text", target_function_or_group="service.process")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        output_value = NestedOutput(data={"message": {"text": "Hello world", "metadata": "ignored"}}, status="ok")
        mock_call_next = AsyncMock(return_value=output_value)

        context = FunctionMiddlewareContext(name="service.process",
                                            config=MagicMock(),
                                            description="Process",
                                            input_schema=_TestInput,
                                            single_output_schema=NestedOutput,
                                            stream_output_schema=type(None))

        await middleware.function_middleware_invoke({}, mock_call_next, context)

        # Verify deeply nested field was extracted
        assert middleware._last_extracted_content == "Hello world"
        assert middleware._last_field_info is not None

    async def test_target_function_filtering(self, mock_builder):
        """Test that defense skips non-targeted functions."""
        config = DefenseMiddlewareConfig(target_field="$.result", target_function_or_group="other_function")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        output_value = _TestOutputModel(result=42.0, operation="multiply", message="Success")
        mock_call_next = AsyncMock(return_value=output_value)

        context = FunctionMiddlewareContext(name="my_calculator.multiply",
                                            config=MagicMock(),
                                            description="Multiply",
                                            input_schema=_TestInput,
                                            single_output_schema=_TestOutputModel,
                                            stream_output_schema=type(None))

        await middleware.function_middleware_invoke(10.0, mock_call_next, context)

        # Defense should not apply, so no field extraction should occur
        assert middleware._last_extracted_content is None
        assert middleware._last_field_info is None

    async def test_multiple_field_matches_with_all_strategy(self, mock_builder):
        """Test resolution strategy 'all' extracts all matching fields."""

        class MultiFieldOutput(BaseModel):
            results: list[float]
            status: str

        config = DefenseMiddlewareConfig(target_field="$.results[*]",
                                         target_field_resolution_strategy="all",
                                         target_function_or_group="processor.batch")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        output_value = MultiFieldOutput(results=[10.0, 20.0, 30.0], status="ok")
        mock_call_next = AsyncMock(return_value=output_value)

        context = FunctionMiddlewareContext(name="processor.batch",
                                            config=MagicMock(),
                                            description="Batch process",
                                            input_schema=_TestInput,
                                            single_output_schema=MultiFieldOutput,
                                            stream_output_schema=type(None))

        await middleware.function_middleware_invoke({}, mock_call_next, context)

        # Verify all fields were extracted as a list
        assert middleware._last_extracted_content == [10.0, 20.0, 30.0]
        assert middleware._last_field_info is not None
        assert len(middleware._last_field_info["matches"]) == 3

    async def test_multiple_field_matches_with_first_strategy(self, mock_builder):
        """Test resolution strategy 'first' extracts only first match."""

        class MultiFieldOutput(BaseModel):
            results: list[float]
            status: str

        config = DefenseMiddlewareConfig(target_field="$.results[*]",
                                         target_field_resolution_strategy="first",
                                         target_function_or_group="processor.batch")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        output_value = MultiFieldOutput(results=[10.0, 20.0, 30.0], status="ok")
        mock_call_next = AsyncMock(return_value=output_value)

        context = FunctionMiddlewareContext(name="processor.batch",
                                            config=MagicMock(),
                                            description="Batch process",
                                            input_schema=_TestInput,
                                            single_output_schema=MultiFieldOutput,
                                            stream_output_schema=type(None))

        await middleware.function_middleware_invoke({}, mock_call_next, context)

        # Verify only first field was extracted
        assert middleware._last_extracted_content == 10.0
        assert middleware._last_field_info is not None
        assert len(middleware._last_field_info["matches"]) == 1

    async def test_multiple_field_matches_with_error_strategy(self, mock_builder):
        """Test resolution strategy 'error' logs warning and analyzes entire value on multiple matches."""

        class MultiFieldOutput(BaseModel):
            results: list[float]
            status: str

        config = DefenseMiddlewareConfig(target_field="$.results[*]",
                                         target_field_resolution_strategy="error",
                                         target_function_or_group="processor.batch")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        output_value = MultiFieldOutput(results=[10.0, 20.0, 30.0], status="ok")
        mock_call_next = AsyncMock(return_value=output_value)

        context = FunctionMiddlewareContext(name="processor.batch",
                                            config=MagicMock(),
                                            description="Batch process",
                                            input_schema=_TestInput,
                                            single_output_schema=MultiFieldOutput,
                                            stream_output_schema=type(None))

        # Defense middleware catches ValueError and analyzes entire value instead
        # (unlike red teaming which raises the error)
        await middleware.function_middleware_invoke({}, mock_call_next, context)

        # Should fall back to analyzing entire value when error strategy encounters multiple matches
        assert middleware._last_extracted_content == output_value
        assert middleware._last_field_info is None

    async def test_extract_simple_type_no_target_field(self, mock_builder):
        """Test that simple types without target_field extract entire value."""
        config = DefenseMiddlewareConfig(target_field=None, target_function_or_group="simple_function")
        middleware = _TestDefenseMiddleware(config, mock_builder)

        output_value = "simple string output"
        mock_call_next = AsyncMock(return_value=output_value)

        context = FunctionMiddlewareContext(name="simple_function",
                                            config=MagicMock(),
                                            description="Simple function",
                                            input_schema=_TestInput,
                                            single_output_schema=str,
                                            stream_output_schema=type(None))

        await middleware.function_middleware_invoke({}, mock_call_next, context)

        # Should extract entire value when no target_field
        assert middleware._last_extracted_content == "simple string output"
        assert middleware._last_field_info is None
