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


@pytest.fixture
def mock_builder():
    """Create a mock builder."""
    return MagicMock()


@pytest.fixture
def middleware_context():
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

        async def mock_next(value):
            return "Contact john.doe@example.com"

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
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

        async def mock_next(value):
            return {"text": "Contact john.doe@example.com", "status": "ok"}

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
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

        async def mock_next(value):
            return _TestOutputModel(text="Contact john.doe@example.com", metadata="ok")

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
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

        async def mock_next(value):
            return {
                "data": {
                    "content": {
                        "message": "Contact john.doe@example.com",
                        "metadata": "ignored"
                    }
                }
            }

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
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

        async def mock_next(value):
            return "Contact john.doe@example.com"

        with patch('nat.middleware.defense_middleware_pii.logger') as mock_logger:
            result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
            # Should log warning but return original output
            mock_logger.warning.assert_called()
            assert result == "Contact john.doe@example.com"

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

        async def mock_next(value):
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

        async def mock_next(value):
            return "Contact john.doe@example.com"

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        # Should return anonymized output
        assert "<EMAIL_ADDRESS>" in result
        assert "john.doe@example.com" not in result

