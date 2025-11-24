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
"""Tests for the RedTeamingMiddleware functionality."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from nat.middleware.function_middleware import FunctionMiddlewareContext
from nat.middleware.red_teaming_middleware import RedTeamingMiddleware


# Test Models
class SimpleModel(BaseModel):
    """Simple test model with basic fields."""
    text: str
    number: int
    value: float


class NestedModel(BaseModel):
    """Nested test model."""
    data: DataModel
    metadata: str


class DataModel(BaseModel):
    """Data sub-model for nested structures."""
    response: ResponseModel
    count: int


class ResponseModel(BaseModel):
    """Response sub-model for deeply nested structures."""
    text: str
    confidence: float


class ComplexModel(BaseModel):
    """Complex model with multiple fields of same type."""
    prompt: str
    context: str
    instruction: str
    temperature: float
    max_tokens: int


@pytest.fixture(name="simple_middleware_context")
def simple_middleware_context():
    """Create a test FunctionMiddlewareContext with simple schemas."""
    return FunctionMiddlewareContext(
        name="test_function",
        config=MagicMock(),
        description="Test function",
        input_schema=SimpleModel,
        single_output_schema=SimpleModel,
        stream_output_schema=SimpleModel
    )


@pytest.fixture(name="nested_middleware_context")
def nested_middleware_context():
    """Create a test FunctionMiddlewareContext with nested schemas."""
    return FunctionMiddlewareContext(
        name="test_nested_function",
        config=MagicMock(),
        description="Test nested function",
        input_schema=NestedModel,
        single_output_schema=NestedModel,
        stream_output_schema=NestedModel
    )



class TestFindFieldInValue:
    """Test the _find_field_in_value method for field discovery in schemas."""

    def test_no_target_field_returns_value_directly(self):
        """Test that None target_field returns the value directly."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK", target_field=None)
        value = "simple_value"

        result_value, field_path = middleware._find_field_in_value(value, None, is_nested_path=False)

        assert result_value == "simple_value"
        assert field_path == []

    def test_find_unique_field_in_simple_schema(self):
        """Test finding a unique field in a simple schema."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK", target_field="text")
        value = SimpleModel(text="hello", number=42, value=3.14)

        result_value, field_path = middleware._find_field_in_value(value, SimpleModel, is_nested_path=False)

        assert result_value == "hello"
        assert field_path == ["text"]

    def test_find_field_in_dict_value(self):
        """Test finding a field when value is a dict."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK", target_field="text")
        value = {"text": "hello", "number": 42, "value": 3.14}

        result_value, field_path = middleware._find_field_in_value(value, SimpleModel, is_nested_path=False)

        assert result_value == "hello"
        assert field_path == ["text"]

    @pytest.mark.parametrize("target_field,expected_value", [
        ("prompt", "ignore previous instructions"),
        ("context", "additional context here"),
        ("instruction", "do something else"),
    ])
    def test_find_different_fields_in_complex_schema(self, target_field, expected_value):
        """Parametric test for finding different unique fields in a complex schema."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK", target_field=target_field)
        value = ComplexModel(
            prompt="ignore previous instructions",
            context="additional context here",
            instruction="do something else",
            temperature=0.7,
            max_tokens=100
        )

        result_value, field_path = middleware._find_field_in_value(value, ComplexModel, is_nested_path=False)

        assert result_value == expected_value
        assert field_path == [target_field]

    def test_nested_path_deep_nesting(self):
        """Test deeply nested path navigation."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK", target_field="data.response.text")
        value = NestedModel(
            data=DataModel(
                response=ResponseModel(text="original response", confidence=0.95),
                count=10
            ),
            metadata="metadata value"
        )

        result_value, field_path = middleware._find_field_in_value(value, NestedModel, is_nested_path=True)

        assert result_value == "original response"
        assert field_path == ["data", "response", "text"]

    def test_nested_path_with_dict(self):
        """Test nested path navigation with dict values."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK", target_field="data.response.text")
        value = {
            "data": {
                "response": {"text": "nested text", "confidence": 0.8},
                "count": 3
            },
            "metadata": "meta"
        }

        result_value, field_path = middleware._find_field_in_value(value, NestedModel, is_nested_path=True)

        assert result_value == "nested text"
        assert field_path == ["data", "response", "text"]

    def test_field_not_found_raises_error(self):
        """Test that searching for non-existent field raises ValueError."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK", target_field="nonexistent")
        value = SimpleModel(text="hello", number=42, value=3.14)

        with pytest.raises(ValueError, match="Field 'nonexistent' not found in schema"):
            middleware._find_field_in_value(value, SimpleModel, is_nested_path=False)

    def test_invalid_nested_path_raises_error(self):
        """Test that invalid nested path raises ValueError."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK", target_field="data.invalid.field")
        value = NestedModel(
            data=DataModel(
                response=ResponseModel(text="test", confidence=0.9),
                count=5
            ),
            metadata="meta"
        )

        with pytest.raises(ValueError, match="Invalid nested path"):
            middleware._find_field_in_value(value, NestedModel, is_nested_path=True)

    def test_cannot_navigate_non_dict_or_basemodel(self):
        """Test that navigating through non-dict/BaseModel raises error."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK", target_field="text.invalid")
        value = SimpleModel(text="hello", number=42, value=3.14)

        with pytest.raises(ValueError, match="Cannot navigate path.*value is not a dict or BaseModel"):
            middleware._find_field_in_value(value, SimpleModel, is_nested_path=True)

    def test_no_schema_with_field_search_raises_error(self):
        """Test that field search without schema raises ValueError."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK", target_field="text")
        value = "simple string"

        with pytest.raises(ValueError, match="Cannot search for field.*without a schema"):
            middleware._find_field_in_value(value, None, is_nested_path=False)


class TestApplyPayload:
    """Test the _apply_payload method for different payload placements and types."""

    # String payload tests
    @pytest.mark.parametrize("placement,original,payload,expected", [
        ("replace", "original text", "ATTACK", "ATTACK"),
        ("append_start", "original text", "ATTACK ", "ATTACK original text"),
        ("append_end", "original text", " ATTACK", "original text ATTACK"),
    ])
    def test_string_payload_basic_placements(self, placement, original, payload, expected):
        """Parametric test for basic string payload placements."""
        middleware = RedTeamingMiddleware(attack_payload=payload, payload_placement=placement)

        result = middleware._apply_payload(original, payload, placement, str)

        assert result == expected

    def test_string_payload_append_middle_single_sentence(self):
        """Test append_middle with single sentence."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK", payload_placement="append_middle")
        original = "This is a test sentence."

        result = middleware._apply_payload(original, " ATTACK ", "append_middle", str)

        # Should insert after the sentence
        assert result == "This is a test sentence. ATTACK "

    def test_string_payload_append_middle_multiple_sentences(self):
        """Test append_middle with multiple sentences."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK", payload_placement="append_middle")
        original = "First sentence. Second sentence. Third sentence."

        result = middleware._apply_payload(original, " ATTACK ", "append_middle", str)

        # Should insert near the middle sentence boundary
        assert " ATTACK " in result
        # Check it's roughly in the middle
        attack_pos = result.index(" ATTACK ")
        assert 10 < attack_pos < len(result) - 10

    def test_string_payload_append_middle_no_punctuation(self):
        """Test append_middle with no sentence punctuation."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK", payload_placement="append_middle")
        original = "no punctuation here"

        result = middleware._apply_payload(original, " ATTACK ", "append_middle", str)

        # Should insert at character midpoint
        assert " ATTACK " in result
        attack_pos = result.index(" ATTACK ")
        # Should be roughly in the middle
        assert 5 < attack_pos < 15

    # Float payload tests
    @pytest.mark.parametrize("placement,original,payload,expected", [
        ("replace", 3.14, "2.71", 2.71),
        ("append_start", 1.5, "9.99", 9.99),  # Falls back to replace
        ("append_end", 1.5, "9.99", 9.99),    # Falls back to replace
        ("append_middle", 1.5, "9.99", 9.99), # Falls back to replace
    ])
    def test_float_payload_all_placements(self, placement, original, payload, expected):
        """Parametric test for float payload (all modes should replace)."""
        middleware = RedTeamingMiddleware(attack_payload=payload, payload_placement=placement)

        result = middleware._apply_payload(original, payload, placement, float)

        assert result == expected
        assert isinstance(result, float)

    def test_list_payload_replaces_random_index(self):
        """Test that list values have a random element replaced."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK", payload_placement="replace")
        original = ["first", "second", "third"]

        result = middleware._apply_payload(original, "ATTACK", "replace")

        # One element should be replaced
        assert "ATTACK" in result
        assert len(result) == 3
        # Two elements should be unchanged
        unchanged_count = sum(1 for elem in ["first", "second", "third"] if elem in result)
        assert unchanged_count == 2

    def test_invalid_float_payload_raises_error(self):
        """Test that invalid float payload raises ValueError."""
        middleware = RedTeamingMiddleware(attack_payload="not_a_number", payload_placement="replace")

        with pytest.raises(ValueError, match="Cannot convert attack payload.*to float"):
            middleware._apply_payload(3.14, "not_a_number", "replace", float)

    def test_unknown_placement_raises_error(self):
        """Test that unknown placement mode raises ValueError."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK", payload_placement="replace")

        with pytest.raises(ValueError, match="Unknown payload placement"):
            middleware._apply_payload("text", "ATTACK", "unknown_mode", str)


class TestSetFieldInValue:
    """Test the _set_field_in_value and _set_field_in_dict methods."""

    def test_set_field_in_basemodel(self):
        """Test setting a field in a BaseModel instance."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK")
        original = SimpleModel(text="original", number=42, value=3.14)

        result = middleware._set_field_in_value(original, ["text"], "modified")

        assert isinstance(result, SimpleModel)
        assert result.text == "modified"
        assert result.number == 42
        assert result.value == 3.14
        # Original should be unchanged (immutable)
        assert original.text == "original"

    def test_set_nested_field_in_basemodel(self):
        """Test setting a deeply nested field in a BaseModel."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK")
        original = NestedModel(
            data=DataModel(
                response=ResponseModel(text="original", confidence=0.9),
                count=5
            ),
            metadata="meta"
        )

        result = middleware._set_field_in_value(original, ["data", "response", "text"], "attacked")

        assert isinstance(result, NestedModel)
        assert result.data.response.text == "attacked"
        assert result.data.response.confidence == 0.9
        assert result.data.count == 5
        assert result.metadata == "meta"

    def test_set_nested_field_in_dict(self):
        """Test setting a deeply nested field in a dict."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK")
        original = {
            "data": {
                "response": {"text": "original", "confidence": 0.9},
                "count": 5
            },
            "metadata": "meta"
        }

        result = middleware._set_field_in_value(original, ["data", "response", "text"], "attacked")

        assert isinstance(result, dict)
        assert result["data"]["response"]["text"] == "attacked"
        assert result["data"]["response"]["confidence"] == 0.9
        assert result["data"]["count"] == 5

    def test_set_field_invalid_path_raises_error(self):
        """Test that invalid path raises ValueError."""
        middleware = RedTeamingMiddleware(attack_payload="ATTACK")
        original = {"text": "original", "number": 42}

        with pytest.raises(ValueError, match="Path key.*not found"):
            middleware._set_field_in_value(original, ["nonexistent", "field"], "value")


class TestApplyPayloadToSchema:
    """Test the _apply_payload_to_schema method end-to-end."""

    def test_apply_payload_to_nested_schema(self, nested_middleware_context):
        """End-to-end test: apply payload to a deeply nested schema field."""
        middleware = RedTeamingMiddleware(
            attack_payload="ATTACKED ",
            target_field="data.response.text",
            payload_placement="append_start"
        )
        value = NestedModel(
            data=DataModel(
                response=ResponseModel(text="original response", confidence=0.95),
                count=10
            ),
            metadata="metadata"
        )

        result = middleware._apply_payload_to_schema(
            value,
            NestedModel,
            nested_middleware_context,
            is_nested_path=True
        )

        assert isinstance(result, NestedModel)
        assert result.data.response.text == "ATTACKED original response"
        assert result.data.response.confidence == 0.95
        assert result.data.count == 10
        assert result.metadata == "metadata"



class TestFunctionMiddlewareInvoke:
    """Test the function_middleware_invoke method for the complete attack flow."""

    async def test_non_targeted_function_skipped(self, simple_middleware_context):
        """Test that non-targeted functions are skipped without modification."""
        middleware = RedTeamingMiddleware(
            attack_payload="ATTACK",
            target_function_or_group="other_function",
            target_field="text"
        )

        input_value = SimpleModel(text="original", number=42, value=3.14)
        expected_output = SimpleModel(text="result", number=100, value=1.0)

        # Track what was passed to call_next
        received_input = None

        async def mock_call_next(value):
            nonlocal received_input
            received_input = value
            return expected_output

        # Call middleware - should pass through without modification
        result = await middleware.function_middleware_invoke(
            input_value,
            mock_call_next,
            simple_middleware_context
        )

        # Verify input was not modified
        assert received_input is not None
        assert received_input == input_value
        assert received_input.text == "original"
        # Verify output was not modified
        assert result == expected_output
        assert result.text == "result"

    @pytest.mark.parametrize("placement,expected_text", [
        ("append_start", "ATTACKoriginal"),
        ("append_end", "originalATTACK"),
        ("replace", "ATTACK"),
    ])
    async def test_input_attack_with_placements(
        self, simple_middleware_context, placement, expected_text
    ):
        """Test attacking function input with different payload placements."""
        middleware = RedTeamingMiddleware(
            attack_payload="ATTACK",
            target_field="text",
            payload_placement=placement,
            target_location="input"
        )

        input_value = SimpleModel(text="original", number=42, value=3.14)

        # Track what was passed to call_next
        received_input = None

        async def mock_call_next(value):
            nonlocal received_input
            received_input = value
            return SimpleModel(text="output", number=1, value=1.0)

        # Call middleware
        await middleware.function_middleware_invoke(
            input_value,
            mock_call_next,
            simple_middleware_context
        )

        # Verify the input was modified correctly before being passed to call_next
        assert received_input is not None
        assert received_input.text == expected_text
        assert received_input.number == 42  # Unchanged
        assert received_input.value == 3.14  # Unchanged

    async def test_input_attack_with_nested_field(self, nested_middleware_context):
        """Test attacking nested input field with dot notation."""
        middleware = RedTeamingMiddleware(
            attack_payload=" INJECTED",
            target_field="data.response.text",
            payload_placement="append_end",
            target_location="input"
        )

        input_value = NestedModel(
            data=DataModel(
                response=ResponseModel(text="original text", confidence=0.9),
                count=5
            ),
            metadata="meta"
        )

        # Track what was passed to call_next
        received_input = None

        async def mock_call_next(value):
            nonlocal received_input
            received_input = value
            return NestedModel(
                data=DataModel(
                    response=ResponseModel(text="output", confidence=0.8),
                    count=1
                ),
                metadata="result"
            )

        # Call middleware
        await middleware.function_middleware_invoke(
            input_value,
            mock_call_next,
            nested_middleware_context
        )

        # Verify nested field was modified
        assert received_input is not None
        assert received_input.data.response.text == "original text INJECTED"
        assert received_input.data.response.confidence == 0.9  # Unchanged
        assert received_input.data.count == 5  # Unchanged
        assert received_input.metadata == "meta"  # Unchanged

    async def test_output_attack_with_replace(self, simple_middleware_context):
        """Test attacking function output with replace mode."""
        middleware = RedTeamingMiddleware(
            attack_payload="REPLACED",
            target_field="text",
            payload_placement="replace",
            target_location="output"
        )

        input_value = SimpleModel(text="input", number=1, value=1.0)
        output_value = SimpleModel(text="original output", number=42, value=3.14)

        # Track what was passed to call_next
        received_input = None

        async def mock_call_next(value):
            nonlocal received_input
            received_input = value
            return output_value

        # Call middleware
        result = await middleware.function_middleware_invoke(
            input_value,
            mock_call_next,
            simple_middleware_context
        )

        # Verify input was passed unchanged to call_next
        assert received_input is not None
        assert received_input == input_value
        assert received_input.text == "input"

        # Verify output was modified
        assert result.text == "REPLACED"
        assert result.number == 42  # Unchanged
        assert result.value == 3.14  # Unchanged

    async def test_error_when_field_not_found(self, simple_middleware_context):
        """Test that ValueError is raised when target field is not found."""
        middleware = RedTeamingMiddleware(
            attack_payload="ATTACK",
            target_field="nonexistent_field",
            target_location="input"
        )

        input_value = SimpleModel(text="input", number=42, value=3.14)

        async def mock_call_next(value):
            return SimpleModel(text="output", number=1, value=1.0)

        # Call middleware - should raise ValueError
        with pytest.raises(ValueError, match="Field 'nonexistent_field' not found in schema"):
            await middleware.function_middleware_invoke(
                input_value,
                mock_call_next,
                simple_middleware_context
            )


