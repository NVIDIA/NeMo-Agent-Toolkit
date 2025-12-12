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
"""Tests for PIIDefenseMiddleware field targeting and analysis."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.middleware.defense_middleware_pii import PIIDefenseMiddleware
from nat.middleware.defense_middleware_pii import PIIDefenseMiddlewareConfig
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
    return FunctionMiddlewareContext(
        name="my_calculator.get_random_string",
        config=MagicMock(),
        description="Get random string",
        input_schema=_TestInput,
        single_output_schema=_TestOutputModel,
        stream_output_schema=type(None)
    )


class TestPIIDefenseFieldTargeting:
    """Test PII Defense with different field targeting scenarios."""

    @pytest.mark.asyncio
    async def test_simple_output_no_target_field(self, mock_builder, middleware_context):
        """Test analyzing simple string output without target_field."""
        config = PIIDefenseMiddlewareConfig(
            target_field=None,
            action="partial_compliance"
        )
        middleware = PIIDefenseMiddleware(config, mock_builder)

        # Mock Presidio analyzer and anonymizer
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [
            MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)
        ]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="<EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return "Contact john.doe@example.com"

        await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        # Should analyze the entire output string
        assert mock_analyzer.analyze.called
        call_args = mock_analyzer.analyze.call_args
        assert "john.doe@example.com" in str(call_args)

    @pytest.mark.asyncio
    async def test_dict_output_with_target_field(self, mock_builder, middleware_context):
        """Test analyzing dict output with target_field."""
        config = PIIDefenseMiddlewareConfig(
            target_field="$.text",
            action="partial_compliance"
        )
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [
            MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)
        ]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="<EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return {"text": "Contact john.doe@example.com", "status": "ok"}

        await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_analyzer.analyze.called
        # Should analyze only the text field
        call_args = mock_analyzer.analyze.call_args
        assert "john.doe@example.com" in str(call_args)

    @pytest.mark.asyncio
    async def test_basemodel_output_with_target_field(self, mock_builder, middleware_context):
        """Test analyzing BaseModel output with target_field."""
        config = PIIDefenseMiddlewareConfig(
            target_field="$.text",
            action="partial_compliance"
        )
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [
            MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)
        ]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="<EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return _TestOutputModel(text="Contact john.doe@example.com", metadata="ok")

        await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_analyzer.analyze.called
        # Should analyze only the text field
        call_args = mock_analyzer.analyze.call_args
        assert "john.doe@example.com" in str(call_args)

    @pytest.mark.asyncio
    async def test_nested_field_targeting(self, mock_builder, middleware_context):
        """Test analyzing nested field in output."""
        config = PIIDefenseMiddlewareConfig(
            target_field="$.data.content.message",
            action="partial_compliance"
        )
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [
            MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)
        ]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="<EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return {
                "data": {
                    "content": {
                        "message": "Contact john.doe@example.com",
                        "metadata": "ignored"
                    }
                }
            }

        await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_analyzer.analyze.called
        # Should analyze only the nested message field
        call_args = mock_analyzer.analyze.call_args
        assert "john.doe@example.com" in str(call_args)


class TestPIIDefenseActions:
    """Test PII Defense actions."""

    @pytest.mark.asyncio
    async def test_action_partial_compliance(self, mock_builder, middleware_context):
        """Test partial_compliance action logs but allows output."""
        config = PIIDefenseMiddlewareConfig(
            action="partial_compliance"
        )
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [
            MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)
        ]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="<EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return "Contact john.doe@example.com"

        with patch('nat.middleware.defense_middleware_pii.logger') as mock_logger:
            await middleware.function_middleware_invoke({}, mock_next, middleware_context)
            # Should log warning but return original output
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_action_refusal(self, mock_builder, middleware_context):
        """Test refusal action raises ValueError."""
        config = PIIDefenseMiddlewareConfig(
            action="refusal"
        )
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [
            MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)
        ]
        middleware._analyzer = mock_analyzer

        # Anonymizer is needed even for refusal action (it's called during analysis)
        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="<EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return "Contact john.doe@example.com"

        with pytest.raises(ValueError, match="PII detected"):
            await middleware.function_middleware_invoke({}, mock_next, middleware_context)

    @pytest.mark.asyncio
    async def test_action_redirection(self, mock_builder, middleware_context):
        """Test redirection action anonymizes PII."""
        config = PIIDefenseMiddlewareConfig(
            action="redirection"
        )
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [
            MagicMock(entity_type="EMAIL_ADDRESS", start=0, end=20, score=0.9)
        ]
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="Contact <EMAIL_ADDRESS>")
        middleware._anonymizer = mock_anonymizer
        # Ensure analyzer is also set
        if middleware._analyzer is None:
            middleware._analyzer = mock_analyzer

        async def mock_next(_value):
            return "Contact john.doe@example.com"

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        # Should return anonymized output
        assert "<EMAIL_ADDRESS>" in result
        assert "john.doe@example.com" not in result


class TestPIIDefenseEntityTypes:
    """Test PII Defense with different entity types."""

    @pytest.mark.asyncio
    async def test_multiple_entity_types(self, mock_builder, middleware_context):
        """Test detecting multiple PII entity types."""
        config = PIIDefenseMiddlewareConfig(
            action="partial_compliance"
        )
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

        with patch('nat.middleware.defense_middleware_pii.logger') as mock_logger:
            await middleware.function_middleware_invoke({}, mock_next, middleware_context)
            # Should detect all three entity types
            assert mock_analyzer.analyze.called
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_no_pii_detected(self, mock_builder, middleware_context):
        """Test when no PII is detected."""
        config = PIIDefenseMiddlewareConfig(
            action="partial_compliance"
        )
        middleware = PIIDefenseMiddleware(config, mock_builder)

        # Mock no PII detected
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []  # No entities
        middleware._analyzer = mock_analyzer

        mock_anonymizer = MagicMock()
        middleware._anonymizer = mock_anonymizer

        async def mock_next(_value):
            return "Safe content with no PII"

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_analyzer.analyze.called
        assert result == "Safe content with no PII"


class TestPIIDefenseTargeting:
    """Test PII Defense targeting configuration."""

    @pytest.mark.asyncio
    async def test_target_function_or_group_none(self, mock_builder, middleware_context):
        """Test that None target applies to all functions."""
        config = PIIDefenseMiddlewareConfig(
            target_function_or_group=None,
            action="partial_compliance"
        )
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        middleware._analyzer = mock_analyzer
        middleware._anonymizer = MagicMock()

        async def mock_next(_value):
            return "content"

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_analyzer.analyze.called
        assert result == "content"

    @pytest.mark.asyncio
    async def test_target_function_or_group_valid(self, mock_builder, middleware_context):
        """Test targeting a valid function."""
        config = PIIDefenseMiddlewareConfig(
            target_function_or_group="my_calculator.get_random_string",
            action="partial_compliance"
        )
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        middleware._analyzer = mock_analyzer
        middleware._anonymizer = MagicMock()

        async def mock_next(_value):
            return "content"

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_analyzer.analyze.called
        assert result == "content"

    @pytest.mark.asyncio
    async def test_target_function_or_group_non_existent(self, mock_builder, middleware_context):
        """Test targeting a non-existent function skips defense."""
        config = PIIDefenseMiddlewareConfig(
            target_function_or_group="calculator.invalid_func",
            action="partial_compliance"
        )
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        middleware._analyzer = mock_analyzer
        middleware._anonymizer = MagicMock()

        async def mock_next(_value):
            return "content"

        # Should NOT apply to non-targeted function
        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert not mock_analyzer.analyze.called  # Defense should not run
        assert result == "content"

    @pytest.mark.asyncio
    async def test_target_location_input_error(self, mock_builder):
        """Test that target_location='input' raises ValidationError at config creation."""
        # Pydantic validates at config creation time, so we can't create a config with "input"
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="Input should be 'output'"):
            PIIDefenseMiddlewareConfig(
                target_location="input",  # type: ignore[arg-type]
                action="partial_compliance"
            )

    @pytest.mark.asyncio
    async def test_target_location_default_output(self, mock_builder, middleware_context):
        """Test that default target_location is 'output'."""
        config = PIIDefenseMiddlewareConfig(
            action="partial_compliance"
        )
        # target_location not specified, should default to "output"
        assert config.target_location == "output"

        middleware = PIIDefenseMiddleware(config, mock_builder)
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        middleware._analyzer = mock_analyzer
        middleware._anonymizer = MagicMock()

        async def mock_next(_value):
            return "content"

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_analyzer.analyze.called
        assert result == "content"

    @pytest.mark.asyncio
    async def test_target_location_explicit_output(self, mock_builder, middleware_context):
        """Test that explicit target_location='output' works."""
        config = PIIDefenseMiddlewareConfig(
            target_location="output",
            action="partial_compliance"
        )
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        middleware._analyzer = mock_analyzer
        middleware._anonymizer = MagicMock()

        async def mock_next(_value):
            return "content"

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_analyzer.analyze.called
        assert result == "content"


class TestPIIDefenseSimpleOutputs:
    """Test PII Defense with simple output formats."""

    @pytest.mark.asyncio
    async def test_simple_string_output(self, mock_builder, middleware_context):
        """Test analyzing simple string output."""
        config = PIIDefenseMiddlewareConfig(
            target_field=None,
            action="partial_compliance"
        )
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        middleware._analyzer = mock_analyzer
        middleware._anonymizer = MagicMock()

        async def mock_next(_value):
            return "simple string output"

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_analyzer.analyze.called
        assert result == "simple string output"

    @pytest.mark.asyncio
    async def test_simple_int_output(self, mock_builder, middleware_context):
        """Test analyzing simple int output."""
        config = PIIDefenseMiddlewareConfig(
            target_field=None,
            action="partial_compliance"
        )
        middleware = PIIDefenseMiddleware(config, mock_builder)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        middleware._analyzer = mock_analyzer
        middleware._anonymizer = MagicMock()

        async def mock_next(_value):
            return 42

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_analyzer.analyze.called
        assert result == 42

