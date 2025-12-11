# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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


@pytest.fixture
def mock_builder():
    """Create a mock builder."""
    builder = MagicMock()
    return builder


@pytest.fixture
def middleware_context():
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

        async def mock_next(value):
            return 42.0

        # Should analyze the entire output (42.0)
        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        # Check that the LLM was called with the output value
        call_args = mock_llm.ainvoke.call_args
        assert "42.0" in str(call_args) or "42" in str(call_args)

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

        async def mock_next(value):
            return {"result": 42.0, "operation": "multiply"}

        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the result field (42.0)
        call_args = mock_llm.ainvoke.call_args
        assert "42.0" in str(call_args) or "42" in str(call_args)

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

        async def mock_next(value):
            return _TestOutputModel(result=42.0, operation="multiply")

        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the result field
        call_args = mock_llm.ainvoke.call_args
        assert "42.0" in str(call_args) or "42" in str(call_args)

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

        async def mock_next(value):
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

        async def mock_next(value):
            return {"results": [42.0, 43.0, 44.0], "count": 3}

        result = await middleware.function_middleware_invoke(10.0, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the first result
        call_args = mock_llm.ainvoke.call_args
        assert "42.0" in str(call_args) or "42" in str(call_args)


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
        mock_response.content = '{"threat_detected": true, "confidence": 0.8, "correct_answer": 4.0, "reason": "Incorrect result"}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(value):
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

        async def mock_next(value):
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

        async def mock_next(value):
            return 999.0  # Incorrect result

        result = await middleware.function_middleware_invoke(2.0, mock_next, middleware_context)
        # Should return corrected value
        assert result == 4.0

