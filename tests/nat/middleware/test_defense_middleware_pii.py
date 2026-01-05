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
"""Tests for PIIDefenseMiddleware field targeting and analysis."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.middleware.defense.defense_middleware_pii import PIIDefenseMiddleware
from nat.middleware.defense.defense_middleware_pii import PIIDefenseMiddlewareConfig
from nat.middleware.middleware import FunctionMiddlewareContext


class _TestInput(BaseModel):
    """Test input model."""
    request: dict


class _TestOutputModel(BaseModel):
    """Test output model."""
    text: str
    metadata: str


@pytest.fixture(name="mock_builder")
def fixture_mock_builder():
    """Create a mock builder."""
    return MagicMock()


@pytest.fixture(name="middleware_context")
def fixture_middleware_context():
    """Create a test FunctionMiddlewareContext."""
    return FunctionMiddlewareContext(name="my_calculator.get_random_string",
                                     config=MagicMock(),
                                     description="Get random string",
                                     input_schema=_TestInput,
                                     single_output_schema=_TestOutputModel,
                                     stream_output_schema=type(None))


class TestPIIDefenseInvoke:
    """Test PII Defense invoke behavior."""

    async def test_simple_output_no_target_field(self, mock_builder, middleware_context):
        """Test analyzing simple string output without target_field."""
        config = PIIDefenseMiddlewareConfig(target_field=None, action="partial_compliance")
        middleware = PIIDefenseMiddleware(config, mock_builder)

        # Mock Presidio analyzer and anonymizer
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="<EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return "Contact john.doe@example.com"

        await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        # Should analyze the entire output string
        assert mock_analyzer.analyze.called
        call_args = mock_analyzer.analyze.call_args
        assert "john.doe@example.com" in str(call_args)

    async def test_dict_output_with_target_field(self, mock_builder, middleware_context):
        """Test analyzing dict output with target_field."""
        config = PIIDefenseMiddlewareConfig(target_field="$.text", action="partial_compliance")
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="<EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return {"text": "Contact john.doe@example.com", "status": "ok"}

        await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert mock_analyzer.analyze.called
        # Should analyze only the text field
        call_args = mock_analyzer.analyze.call_args
        assert "john.doe@example.com" in str(call_args)

    async def test_basemodel_output_with_target_field(self, mock_builder, middleware_context):
        """Test analyzing BaseModel output with target_field."""
        config = PIIDefenseMiddlewareConfig(target_field="$.text", action="partial_compliance")
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="<EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return _TestOutputModel(text="Contact john.doe@example.com", metadata="ok")

        await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert mock_analyzer.analyze.called
        # Should analyze only the text field
        call_args = mock_analyzer.analyze.call_args
        assert "john.doe@example.com" in str(call_args)

    async def test_nested_field_targeting(self, mock_builder, middleware_context):
        """Test analyzing nested field in output."""
        config = PIIDefenseMiddlewareConfig(target_field="$.data.content.message", action="partial_compliance")
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="<EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return {"data": {"content": {"message": "Contact john.doe@example.com", "metadata": "ignored"}}}

        await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert mock_analyzer.analyze.called
        # Should analyze only the nested message field
        call_args = mock_analyzer.analyze.call_args
        assert "john.doe@example.com" in str(call_args)

    async def test_complex_nested_structure_with_field_targeting(self, mock_builder, middleware_context):
        """Test field targeting on complex nested structure with lists and dicts."""
        config = PIIDefenseMiddlewareConfig(
            target_field="$.results[0].user.email",
            action="redirection"  # Use redirection to verify anonymization works
        )
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="<EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return {
                "results": [{
                    "user": {
                        "email": "john.doe@example.com", "id": 123
                    }, "metadata": {
                        "ignored": True
                    }
                }, {
                    "user": {
                        "email": "jane.smith@example.com", "id": 456
                    }
                }],
                "total": 2
            }

        result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert mock_analyzer.analyze.called
        # Should analyze only the first result's user email
        call_args = mock_analyzer.analyze.call_args
        assert "john.doe@example.com" in str(call_args)
        # Verify structure is preserved and email is anonymized
        assert result["results"][0]["user"]["email"] == "<EMAIL_ADDRESS>"
        assert result["results"][1]["user"]["email"] == "jane.smith@example.com"
        assert result["total"] == 2

    async def test_field_resolution_strategy_all(self, mock_builder, middleware_context):
        """Test field resolution strategy 'all' analyzes all matching fields."""
        config = PIIDefenseMiddlewareConfig(target_field="$.items[*].email",
                                            target_field_resolution_strategy="all",
                                            action="partial_compliance")
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="<EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return {
                "items": [{
                    "email": "first@example.com", "id": 1
                }, {
                    "email": "second@example.com", "id": 2
                }, {
                    "email": "third@example.com", "id": 3
                }]
            }

        with patch('nat.middleware.defense.defense_middleware_pii.logger'):
            result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
            assert mock_analyzer.analyze.called

            # For partial_compliance, middleware should return original structure unchanged
            assert result == {
                "items": [{
                    "email": "first@example.com", "id": 1
                }, {
                    "email": "second@example.com", "id": 2
                }, {
                    "email": "third@example.com", "id": 3
                }]
            }

            # call_args is a unittest.mock._Call object
            # Presidio's analyze method signature: analyze(text=..., language='en', entities=...)
            call_args = mock_analyzer.analyze.call_args
            # Extract text from kwargs (Presidio uses keyword arguments)
            text_analyzed = call_args.kwargs.get(
                'text', '') if call_args.kwargs else (call_args.args[0] if call_args.args else '')

            # When strategy="all", extracted_value is a list:
            # ["first@example.com", "second@example.com", "third@example.com"]
            # This gets converted to string for Presidio analysis
            assert "first@example.com" in text_analyzed, (
                f"Expected 'first@example.com' in analyzed text: {text_analyzed}")
            assert "second@example.com" in text_analyzed, (
                f"Expected 'second@example.com' in analyzed text: {text_analyzed}")
            assert "third@example.com" in text_analyzed, (
                f"Expected 'third@example.com' in analyzed text: {text_analyzed}")

    async def test_action_partial_compliance(self, mock_builder, middleware_context):
        """Test partial_compliance action logs but allows output."""
        config = PIIDefenseMiddlewareConfig(action="partial_compliance")
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="<EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return "Contact john.doe@example.com"

        with patch('nat.middleware.defense.defense_middleware_pii.logger') as mock_logger:
            await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
            # Should log warning but return original output
            mock_logger.warning.assert_called()

    async def test_action_refusal(self, mock_builder, middleware_context):
        """Test refusal action raises ValueError."""
        config = PIIDefenseMiddlewareConfig(action="refusal")
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)]
        middleware._analyzer = mock_analyzer

        # Anonymizer is needed even for refusal action (it's called during analysis)
        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="<EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return "Contact john.doe@example.com"

        with pytest.raises(ValueError, match="PII detected"):
            await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)

    async def test_action_redirection(self, mock_builder, middleware_context):
        """Test redirection action anonymizes PII."""
        config = PIIDefenseMiddlewareConfig(action="redirection")
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="Contact <EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer
        # Ensure analyzer is also set
        if middleware._analyzer is None:
            middleware._analyzer = mock_analyzer

        async def mock_next(_value):
            return "Contact john.doe@example.com"

        result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        # Should return anonymized output
        assert "<EMAIL_ADDRESS>" in result
        assert "john.doe@example.com" not in result

    async def test_multiple_entity_types(self, mock_builder, middleware_context):
        """Test detecting multiple PII entity types."""
        config = PIIDefenseMiddlewareConfig(action="partial_compliance")
        middleware = PIIDefenseMiddleware(config, mock_builder)

        # Mock multiple entity types
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [
            MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9),
            MagicMock(entity_type="PERSON", start=21, end=26, score=0.95),
            MagicMock(entity_type="PHONE_NUMBER", start=27, end=39, score=0.85)
        ]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="<EMAIL_ADDRESS> <PERSON> <PHONE_NUMBER>")
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return "Contact john.doe@example.com John 555-123-4567"

        with patch('nat.middleware.defense.defense_middleware_pii.logger') as mock_logger:
            await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
            # Should detect all three entity types
            assert mock_analyzer.analyze.called
            mock_logger.warning.assert_called()

    async def test_no_pii_detected(self, mock_builder, middleware_context):
        """Test when no PII is detected."""
        config = PIIDefenseMiddlewareConfig(action="partial_compliance")
        middleware = PIIDefenseMiddleware(config, mock_builder)

        # Mock no PII detected
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []  # No entities
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return "Safe content with no PII"

        result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert mock_analyzer.analyze.called
        assert result == "Safe content with no PII"

    async def test_targeting_configuration(self, mock_builder, middleware_context):
        """Test targeting configuration (function/group targeting and target_location)."""
        # Test None target applies to all functions
        config = PIIDefenseMiddlewareConfig(target_function_or_group=None, action="partial_compliance")
        middleware = PIIDefenseMiddleware(config, mock_builder)
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        middleware._analyzer = mock_analyzer
        middleware._anonymizer = MagicMock()

        async def mock_next(_value):
            return "content"

        result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert mock_analyzer.analyze.called
        assert result == "content"

        # Test specific function targeting
        config = PIIDefenseMiddlewareConfig(target_function_or_group="my_calculator.get_random_string",
                                            action="partial_compliance")
        middleware = PIIDefenseMiddleware(config, mock_builder)
        middleware._analyzer = mock_analyzer
        middleware._anonymizer = MagicMock()
        mock_analyzer.analyze.reset_mock()

        result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert mock_analyzer.analyze.called
        assert result == "content"

        # Test non-targeted function skips defense
        config = PIIDefenseMiddlewareConfig(target_function_or_group="calculator.invalid_func",
                                            action="partial_compliance")
        middleware = PIIDefenseMiddleware(config, mock_builder)
        mock_analyzer.analyze.reset_mock()

        result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert not mock_analyzer.analyze.called  # Defense should not run
        assert result == "content"

    async def test_target_location_validation(self, mock_builder, middleware_context):
        """Test target_location validation and default behavior."""
        # Test that target_location='input' raises ValidationError
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="Input should be 'output'"):
            PIIDefenseMiddlewareConfig(
                target_location="input",  # type: ignore[arg-type]
                action="partial_compliance")

        # Test default is 'output'
        config = PIIDefenseMiddlewareConfig(action="partial_compliance")
        assert config.target_location == "output"

        # Test explicit 'output' works
        config = PIIDefenseMiddlewareConfig(target_location="output", action="partial_compliance")
        middleware = PIIDefenseMiddleware(config, mock_builder)
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        middleware._analyzer = mock_analyzer
        middleware._anonymizer = MagicMock()

        async def mock_next(_value):
            return "content"

        result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert mock_analyzer.analyze.called
        assert result == "content"

    async def test_non_string_output_converts_to_string(self, mock_builder, middleware_context):
        """Test that non-string outputs (int, float, dict, list) are converted to strings for analysis."""
        config = PIIDefenseMiddlewareConfig(target_field=None, action="partial_compliance")
        middleware = PIIDefenseMiddleware(config, mock_builder)

        # Test int
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        middleware._analyzer = mock_analyzer
        middleware._anonymizer = MagicMock()

        async def mock_next_int(_value):
            return 42

        result = await middleware.function_middleware_invoke({}, call_next=mock_next_int, context=middleware_context)
        assert mock_analyzer.analyze.called
        call_args = mock_analyzer.analyze.call_args
        # Verify int was converted to string for Presidio analysis
        assert "42" in str(call_args) or '"42"' in str(call_args)
        assert result == 42

        # Test float
        mock_analyzer.analyze.reset_mock()

        async def mock_next_float(_value):
            return 3.14

        result = await middleware.function_middleware_invoke({}, call_next=mock_next_float, context=middleware_context)
        assert mock_analyzer.analyze.called
        call_args = mock_analyzer.analyze.call_args
        assert "3.14" in str(call_args) or '"3.14"' in str(call_args)
        assert result == 3.14

        # Test dict
        mock_analyzer.analyze.reset_mock()

        async def mock_next_dict(_value):
            return {"key": "value"}

        result = await middleware.function_middleware_invoke({}, call_next=mock_next_dict, context=middleware_context)
        assert mock_analyzer.analyze.called
        call_args = mock_analyzer.analyze.call_args
        # Dict should be converted to string representation
        assert "key" in str(call_args) or "value" in str(call_args)
        assert result == {"key": "value"}


class TestPIIDefenseStreaming:
    """Test PII Defense streaming behavior."""

    async def test_streaming_no_pii_detected(self, mock_builder, middleware_context):
        """Test streaming with no PII yields original chunks."""
        config = PIIDefenseMiddlewareConfig(action="redirection")
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        middleware._analyzer = mock_analyzer
        middleware._anonymizer = MagicMock()

        async def mock_stream(_value):
            yield "Hello "
            yield "world"

        chunks = []
        async for chunk in middleware.function_middleware_stream({}, call_next=mock_stream, context=middleware_context):
            chunks.append(chunk)

        assert chunks == ["Hello ", "world"]
        assert mock_analyzer.analyze.called

    async def test_streaming_refusal_action(self, mock_builder, middleware_context):
        """Test streaming refusal action raises exception."""
        config = PIIDefenseMiddlewareConfig(action="refusal")
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="Contact <EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_stream(_value):
            yield "Contact "
            yield "john.doe@example.com"

        with pytest.raises(ValueError, match="PII detected in output"):
            async for _ in middleware.function_middleware_stream({}, call_next=mock_stream, context=middleware_context):
                pass

    async def test_streaming_redirection_action(self, mock_builder, middleware_context):
        """Test streaming redirection action yields anonymized content."""
        config = PIIDefenseMiddlewareConfig(action="redirection")
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="Contact <EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_stream(_value):
            yield "Contact "
            yield "john.doe@example.com"

        chunks = []
        async for chunk in middleware.function_middleware_stream({}, call_next=mock_stream, context=middleware_context):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == "Contact <EMAIL_ADDRESS>"

    async def test_streaming_partial_compliance(self, mock_builder, middleware_context):
        """Test streaming partial_compliance yields original chunks."""
        config = PIIDefenseMiddlewareConfig(action="partial_compliance")
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="Contact <EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_stream(_value):
            yield "Contact "
            yield "john.doe@example.com"

        with patch('nat.middleware.defense.defense_middleware_pii.logger') as mock_logger:
            chunks = []
            async for chunk in middleware.function_middleware_stream({},
                                                                     call_next=mock_stream,
                                                                     context=middleware_context):
                chunks.append(chunk)

            assert chunks == ["Contact ", "john.doe@example.com"]
            mock_logger.warning.assert_called()

    async def test_streaming_skips_when_not_targeted(self, mock_builder, middleware_context):
        """Test streaming skips when function not targeted."""
        config = PIIDefenseMiddlewareConfig(target_function_or_group="other_function", action="refusal")
        middleware = PIIDefenseMiddleware(config, mock_builder)

        async def mock_stream(_value):
            yield "chunk1"
            yield "chunk2"

        chunks = []
        async for chunk in middleware.function_middleware_stream({}, call_next=mock_stream, context=middleware_context):
            chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2"]
        assert middleware._analyzer is None
