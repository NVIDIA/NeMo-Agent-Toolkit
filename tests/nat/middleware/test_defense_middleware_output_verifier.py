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
"""Tests for OutputVerifierMiddleware field targeting and analysis."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.middleware.defense_middleware_output_verifier import OutputVerifierMiddleware
from nat.middleware.defense_middleware_output_verifier import OutputVerifierMiddlewareConfig
from nat.middleware.middleware import FunctionMiddlewareContext


class _TestInput(BaseModel):
    """Test input model."""
    value: float


class _TestOutputModel(BaseModel):
    """Test output model."""
    result: float
    operation: str


@pytest.fixture(name="mock_builder")
def fixture_mock_builder():
    """Create a mock builder."""
    builder = MagicMock()
    return builder


@pytest.fixture(name="middleware_context")
def fixture_middleware_context():
    """Create a test FunctionMiddlewareContext."""
    return FunctionMiddlewareContext(
        name="my_calculator.multiply",
        config=MagicMock(),
        description="Multiply function",
        input_schema=_TestInput,
        single_output_schema=_TestOutputModel,
        stream_output_schema=type(None)
    )


class TestOutputVerifierFieldTargeting:
    """Test Output Verifier with different field targeting scenarios."""

    @pytest.mark.asyncio
    async def test_simple_output_no_target_field(self, mock_builder, middleware_context):
        """Test analyzing simple output without target_field."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            target_field=None,
            action="partial_compliance"
        )
        middleware = OutputVerifierMiddleware(config, mock_builder)

        # Mock LLM response
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 42.0

        # Should analyze the entire output (42.0)
        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        # Check that the LLM was called with the output value
        call_args = mock_llm.ainvoke.call_args
        assert "42.0" in str(call_args) or "42" in str(call_args)
        assert result == 42.0

    @pytest.mark.asyncio
    async def test_dict_output_with_target_field(self, mock_builder, middleware_context):
        """Test analyzing dict output with target_field."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            target_field="$.result",
            action="partial_compliance"
        )
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return {"result": 42.0, "operation": "multiply"}

        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the result field (42.0)
        call_args = mock_llm.ainvoke.call_args
        assert "42.0" in str(call_args) or "42" in str(call_args)
        assert result == {"result": 42.0, "operation": "multiply"}

    @pytest.mark.asyncio
    async def test_basemodel_output_with_target_field(self, mock_builder, middleware_context):
        """Test analyzing BaseModel output with target_field."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            target_field="$.result",
            action="partial_compliance"
        )
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return _TestOutputModel(result=42.0, operation="multiply")

        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the result field
        call_args = mock_llm.ainvoke.call_args
        assert "42.0" in str(call_args) or "42" in str(call_args)
        assert isinstance(result, _TestOutputModel)
        assert result.result == 42.0

    @pytest.mark.asyncio
    async def test_nested_field_targeting(self, mock_builder, middleware_context):
        """Test analyzing nested field in output."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            target_field="$.data.message.result",
            action="partial_compliance"
        )
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return {
                "data": {
                    "message": {
                        "result": 42.0,
                        "status": "ok"
                    }
                },
                "metadata": "ignored"
            }

        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the nested result field
        call_args = mock_llm.ainvoke.call_args
        assert "42.0" in str(call_args) or "42" in str(call_args)
        assert result["data"]["message"]["result"] == 42.0

    @pytest.mark.asyncio
    async def test_list_field_targeting(self, mock_builder, middleware_context):
        """Test analyzing list element with target_field."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            target_field="results[0]",
            action="partial_compliance"
        )
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return {"results": [42.0, 43.0, 44.0], "count": 3}

        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the first result
        call_args = mock_llm.ainvoke.call_args
        assert "42.0" in str(call_args) or "42" in str(call_args)
        assert result == {"results": [42.0, 43.0, 44.0], "count": 3}


class TestOutputVerifierActions:
    """Test Output Verifier actions (partial_compliance, refusal, redirection)."""

    @pytest.mark.asyncio
    async def test_action_partial_compliance(self, mock_builder, middleware_context):
        """Test partial_compliance action logs but allows output."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            action="partial_compliance",
            threshold=0.7
        )
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = (
            '{"threat_detected": true, "confidence": 0.8, '
            '"correct_answer": 4.0, "reason": "Incorrect result"}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 999.0  # Incorrect result

        with patch('nat.middleware.defense_middleware_output_verifier.logger') as mock_logger:
            result = await middleware.function_middleware_invoke(2.0, mock_next, middleware_context)
            # Should log warning but return original output
            mock_logger.warning.assert_called()
            assert result == 999.0

    @pytest.mark.asyncio
    async def test_action_refusal(self, mock_builder, middleware_context):
        """Test refusal action raises ValueError."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            action="refusal",
            threshold=0.7
        )
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = (
            '{"threat_detected": true, "confidence": 0.9, "correct_answer": 4.0, "reason": "Incorrect result"}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 999.0  # Incorrect result

        with pytest.raises(ValueError, match="Content blocked by security policy"):
            await middleware.function_middleware_invoke(2.0, mock_next, middleware_context)

    @pytest.mark.asyncio
    async def test_action_redirection(self, mock_builder, middleware_context):
        """Test redirection action replaces output with correct answer."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            action="redirection",
            threshold=0.7,
            tool_description="Multiplies numbers"
        )
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = (
            '{"threat_detected": true, "confidence": 0.9, "correct_answer": 4.0, "reason": "Incorrect result"}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 999.0  # Incorrect result

        result = await middleware.function_middleware_invoke(2.0, mock_next, middleware_context)
        # Should return corrected value
        assert result == 4.0


class TestOutputVerifierTargeting:
    """Test Output Verifier targeting configuration."""

    @pytest.mark.asyncio
    async def test_target_function_or_group_none(self, mock_builder, middleware_context):
        """Test that None target applies to all functions."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            target_function_or_group=None,
            action="partial_compliance"
        )
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 42.0

        # Should apply to any function
        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        assert result == 42.0

    @pytest.mark.asyncio
    async def test_target_function_or_group_valid(self, mock_builder, middleware_context):
        """Test targeting a valid function."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            target_function_or_group="my_calculator.multiply",
            action="partial_compliance"
        )
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 42.0

        # Should apply to targeted function
        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        assert result == 42.0

    @pytest.mark.asyncio
    async def test_target_function_or_group_non_existent(self, mock_builder, middleware_context):
        """Test targeting a non-existent function skips defense."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            target_function_or_group="calculator.invalid_func",
            action="partial_compliance"
        )
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 42.0

        # Should NOT apply to non-targeted function
        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert not mock_llm.ainvoke.called  # Defense should not run
        assert result == 42.0

    @pytest.mark.asyncio
    async def test_target_location_input_error(self, mock_builder, middleware_context):
        """Test that target_location='input' raises ValidationError at config creation."""
        # Pydantic validates at config creation time, so we can't create a config with "input"
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="Input should be 'output'"):
            OutputVerifierMiddlewareConfig(
                llm_name="test_llm",
                target_location="input",  # type: ignore[arg-type]
                action="partial_compliance"
            )

    @pytest.mark.asyncio
    async def test_target_location_default_output(self, mock_builder, middleware_context):
        """Test that default target_location is 'output'."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            action="partial_compliance"
        )
        # target_location not specified, should default to "output"
        assert config.target_location == "output"

        middleware = OutputVerifierMiddleware(config, mock_builder)
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 42.0

        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        assert result == 42.0

    @pytest.mark.asyncio
    async def test_target_location_explicit_output(self, mock_builder, middleware_context):
        """Test that explicit target_location='output' works."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            target_location="output",
            action="partial_compliance"
        )
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 42.0

        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        assert result == 42.0


class TestOutputVerifierSimpleOutputs:
    """Test Output Verifier with simple output formats."""

    @pytest.mark.asyncio
    async def test_simple_string_output(self, mock_builder, middleware_context):
        """Test analyzing simple string output."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            target_field=None,
            action="partial_compliance"
        )
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "simple string output"

        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        assert result == "simple string output"

    @pytest.mark.asyncio
    async def test_simple_int_output(self, mock_builder, middleware_context):
        """Test analyzing simple int output."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            target_field=None,
            action="partial_compliance"
        )
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 42

        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        assert result == 42

    @pytest.mark.asyncio
    async def test_simple_output_with_target_field_ignored(self, mock_builder, middleware_context):
        """Test that target_field is ignored for simple types."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            target_field="$.result",  # Should be ignored for simple types
            action="partial_compliance"
        )
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 42.0  # Simple float

        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze entire value, not try to extract field
        call_args = mock_llm.ainvoke.call_args
        assert "42.0" in str(call_args) or "42" in str(call_args)
        assert result == 42.0

