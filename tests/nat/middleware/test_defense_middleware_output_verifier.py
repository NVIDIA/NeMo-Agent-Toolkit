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
"""Tests for OutputVerifierMiddleware field targeting and analysis."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.middleware.defense.defense_middleware_output_verifier import OutputVerifierMiddleware
from nat.middleware.defense.defense_middleware_output_verifier import OutputVerifierMiddlewareConfig
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
    return FunctionMiddlewareContext(name="my_calculator.multiply",
                                     config=MagicMock(),
                                     description="Multiply function",
                                     input_schema=_TestInput,
                                     single_output_schema=_TestOutputModel,
                                     stream_output_schema=type(None))


class TestOutputVerifierInvoke:
    """Test Output Verifier invoke behavior."""

    async def test_simple_output_no_target_field(self, mock_builder, middleware_context):
        """Test analyzing simple output without target_field."""
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm", target_field=None, action="partial_compliance")
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
        result = await middleware.function_middleware_invoke(10.0, call_next=mock_next, context=middleware_context)
        assert mock_llm.ainvoke.called
        # Check that the LLM was called with the output value
        call_args = mock_llm.ainvoke.call_args
        assert "42.0" in str(call_args) or "42" in str(call_args)
        assert result == 42.0

    async def test_dict_output_with_target_field(self, mock_builder, middleware_context):
        """Test analyzing dict output with target_field."""
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm",
                                                target_field="$.result",
                                                action="partial_compliance")
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return {"result": 42.0, "operation": "multiply"}

        result = await middleware.function_middleware_invoke(10.0, call_next=mock_next, context=middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the result field (42.0)
        call_args = mock_llm.ainvoke.call_args
        assert "42.0" in str(call_args) or "42" in str(call_args)
        assert result == {"result": 42.0, "operation": "multiply"}

    async def test_basemodel_output_with_target_field(self, mock_builder, middleware_context):
        """Test analyzing BaseModel output with target_field."""
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm",
                                                target_field="$.result",
                                                action="partial_compliance")
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return _TestOutputModel(result=42.0, operation="multiply")

        result = await middleware.function_middleware_invoke(10.0, call_next=mock_next, context=middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the result field
        call_args = mock_llm.ainvoke.call_args
        assert "42.0" in str(call_args) or "42" in str(call_args)
        assert isinstance(result, _TestOutputModel)
        assert result.result == 42.0

    async def test_nested_field_targeting(self, mock_builder, middleware_context):
        """Test analyzing nested field in output."""
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm",
                                                target_field="$.data.message.result",
                                                action="partial_compliance")
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return {"data": {"message": {"result": 42.0, "status": "ok"}}, "metadata": "ignored"}

        result = await middleware.function_middleware_invoke(10.0, call_next=mock_next, context=middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the nested result field
        call_args = mock_llm.ainvoke.call_args
        assert "42.0" in str(call_args) or "42" in str(call_args)
        assert result["data"]["message"]["result"] == 42.0

    async def test_list_field_targeting(self, mock_builder, middleware_context):
        """Test analyzing list element with target_field."""
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm",
                                                target_field="$.results[0]",
                                                action="partial_compliance")
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return {"results": [42.0, 43.0, 44.0], "count": 3}

        result = await middleware.function_middleware_invoke(10.0, call_next=mock_next, context=middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the first result
        call_args = mock_llm.ainvoke.call_args
        assert "42.0" in str(call_args) or "42" in str(call_args)
        assert result == {"results": [42.0, 43.0, 44.0], "count": 3}

    async def test_complex_nested_structure_with_field_targeting(self, mock_builder, middleware_context):
        """Test field targeting on complex nested structure with lists and dicts."""
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm",
                                                target_field="$.results[0].calculation.result",
                                                action="partial_compliance")
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return {
                "results": [{
                    "calculation": {
                        "result": 42.0, "operation": "multiply"
                    }, "metadata": {
                        "ignored": True
                    }
                }, {
                    "calculation": {
                        "result": 10.0, "operation": "add"
                    }
                }],
                "total": 2
            }

        result = await middleware.function_middleware_invoke({
            "a": 2, "b": 3
        },
                                                             call_next=mock_next,
                                                             context=middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the first result's calculation result
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]
        user_content = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        assert "42.0" in user_content or "42" in user_content
        # Verify structure is preserved
        assert result["results"][0]["calculation"]["result"] == 42.0
        assert result["results"][1]["calculation"]["result"] == 10.0
        assert result["total"] == 2

    async def test_field_resolution_strategy_all(self, mock_builder, middleware_context):
        """Test field resolution strategy 'all' analyzes all matching fields."""
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm",
                                                target_field="$.items[*].result",
                                                target_field_resolution_strategy="all",
                                                action="partial_compliance")
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": true, "confidence": 0.8, "reason": "Incorrect result"}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return {"items": [{"result": 1.0, "id": 1}, {"result": 2.0, "id": 2}, {"result": 3.0, "id": 3}]}

        with patch('nat.middleware.defense.defense_middleware_output_verifier.logger'):
            result = await middleware.function_middleware_invoke({
                "a": 2, "b": 3
            },
                                                                 call_next=mock_next,
                                                                 context=middleware_context)
            assert mock_llm.ainvoke.called

            # call_args is a unittest.mock._Call object: call(args, kwargs)
            # call_args[0] is the args tuple, call_args[0][0] is the first positional argument (messages list)
            call_args = mock_llm.ainvoke.call_args
            messages = call_args[0][
                0]  # Extract messages list: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            user_content = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]

            # When strategy="all", extracted_value is a list: [1.0, 2.0, 3.0]
            # This gets converted to string for analysis: "[1.0, 2.0, 3.0]"
            # Verify all three fields are present in the content string sent to the verifier
            assert "1.0" in user_content or "1" in user_content, f"Expected '1.0' in content: {user_content}"
            assert "2.0" in user_content or "2" in user_content, f"Expected '2.0' in content: {user_content}"
            assert "3.0" in user_content or "3" in user_content, f"Expected '3.0' in content: {user_content}"

            # For partial_compliance action, result should be unchanged (original structure)
            assert result == {"items": [{"result": 1.0, "id": 1}, {"result": 2.0, "id": 2}, {"result": 3.0, "id": 3}]}

    async def test_action_partial_compliance(self, mock_builder, middleware_context):
        """Test partial_compliance action logs but allows output."""
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm", action="partial_compliance", threshold=0.7)
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = ('{"threat_detected": true, "confidence": 0.8, '
                                 '"correct_answer": 4.0, "reason": "Incorrect result"}')
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 999.0  # Incorrect result

        with patch('nat.middleware.defense.defense_middleware_output_verifier.logger') as mock_logger:
            result = await middleware.function_middleware_invoke(2.0, call_next=mock_next, context=middleware_context)
            # Should log warning but return original output
            mock_logger.warning.assert_called()
            assert result == 999.0

    async def test_action_refusal(self, mock_builder, middleware_context):
        """Test refusal action raises ValueError."""
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm", action="refusal", threshold=0.7)
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = (
            '{"threat_detected": true, "confidence": 0.9, "correct_answer": 4.0, "reason": "Incorrect result"}')
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 999.0  # Incorrect result

        with pytest.raises(ValueError, match="Content blocked by security policy"):
            await middleware.function_middleware_invoke(2.0, call_next=mock_next, context=middleware_context)

    async def test_action_redirection(self, mock_builder, middleware_context):
        """Test redirection action replaces output with correct answer."""
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm",
                                                action="redirection",
                                                threshold=0.7,
                                                tool_description="Multiplies numbers")
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = (
            '{"threat_detected": true, "confidence": 0.9, "correct_answer": 4.0, "reason": "Incorrect result"}')
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 999.0  # Incorrect result

        result = await middleware.function_middleware_invoke(2.0, call_next=mock_next, context=middleware_context)
        # Should return corrected value
        assert result == 4.0

    async def test_targeting_configuration(self, mock_builder, middleware_context):
        """Test targeting configuration (function/group targeting and target_location)."""
        # Test None target applies to all functions
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm",
                                                target_function_or_group=None,
                                                action="partial_compliance")
        middleware = OutputVerifierMiddleware(config, mock_builder)
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 42.0

        result = await middleware.function_middleware_invoke(10.0, call_next=mock_next, context=middleware_context)
        assert mock_llm.ainvoke.called
        assert result == 42.0

        # Test specific function targeting
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm",
                                                target_function_or_group="my_calculator.multiply",
                                                action="partial_compliance")
        middleware = OutputVerifierMiddleware(config, mock_builder)
        middleware._llm = mock_llm
        mock_llm.ainvoke.reset_mock()

        result = await middleware.function_middleware_invoke(10.0, call_next=mock_next, context=middleware_context)
        assert mock_llm.ainvoke.called
        assert result == 42.0

        # Test non-targeted function skips defense
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm",
                                                target_function_or_group="calculator.invalid_func",
                                                action="partial_compliance")
        middleware = OutputVerifierMiddleware(config, mock_builder)
        mock_llm.ainvoke.reset_mock()

        result = await middleware.function_middleware_invoke(10.0, call_next=mock_next, context=middleware_context)
        assert not mock_llm.ainvoke.called  # Defense should not run
        assert result == 42.0

    async def test_target_location_validation(self, mock_builder, middleware_context):
        """Test target_location validation and default behavior."""
        # Test that target_location='input' raises ValidationError
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="Input should be 'output'"):
            OutputVerifierMiddlewareConfig(
                llm_name="test_llm",
                target_location="input",  # type: ignore[arg-type]
                action="partial_compliance")

        # Test default is 'output'
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm", action="partial_compliance")
        assert config.target_location == "output"

        # Test explicit 'output' works
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm",
                                                target_location="output",
                                                action="partial_compliance")
        middleware = OutputVerifierMiddleware(config, mock_builder)
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 42.0

        result = await middleware.function_middleware_invoke(10.0, call_next=mock_next, context=middleware_context)
        assert mock_llm.ainvoke.called
        assert result == 42.0

    async def test_non_string_output_converts_to_string(self, mock_builder, middleware_context):
        """Test that non-string outputs (int, float, dict, list) are converted to strings for analysis."""
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm", target_field=None, action="partial_compliance")
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        # Test int
        async def mock_next_int(_value):
            return 42

        result = await middleware.function_middleware_invoke(10.0, call_next=mock_next_int, context=middleware_context)
        assert mock_llm.ainvoke.called
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]  # Extract messages list
        # Output Verifier uses [system, user] format, user message contains the output
        user_content = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        # Verify int was converted to string for analysis (check in user content)
        assert "42" in user_content or '"42"' in user_content
        assert result == 42

        # Test float
        mock_llm.ainvoke.reset_mock()

        async def mock_next_float(_value):
            return 3.14

        result = await middleware.function_middleware_invoke(10.0,
                                                             call_next=mock_next_float,
                                                             context=middleware_context)
        assert mock_llm.ainvoke.called
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]
        user_content = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        assert "3.14" in user_content or '"3.14"' in user_content
        assert result == 3.14

        # Test dict
        mock_llm.ainvoke.reset_mock()

        async def mock_next_dict(_value):
            return {"key": "value"}

        result = await middleware.function_middleware_invoke(10.0, call_next=mock_next_dict, context=middleware_context)
        assert mock_llm.ainvoke.called
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]
        user_content = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        # Dict should be converted to string representation
        assert "key" in user_content or "value" in user_content
        assert result == {"key": "value"}

    async def test_simple_output_with_target_field_ignored(self, mock_builder, middleware_context):
        """Test that target_field is ignored for simple types."""
        config = OutputVerifierMiddlewareConfig(
            llm_name="test_llm",
            target_field="$.result",  # Should be ignored for simple types
            action="partial_compliance")
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 42.0  # Simple float

        result = await middleware.function_middleware_invoke(10.0, call_next=mock_next, context=middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze entire value, not try to extract field
        call_args = mock_llm.ainvoke.call_args
        assert "42.0" in str(call_args) or "42" in str(call_args)
        assert result == 42.0


class TestOutputVerifierStreaming:
    """Test Output Verifier streaming behavior."""

    async def test_streaming_correct_output(self, mock_builder, middleware_context):
        """Test streaming correct output yields original chunks."""
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm", action="refusal")
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": false, "confidence": 0.9}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_stream(_value):
            yield "6.0"

        chunks = []
        async for chunk in middleware.function_middleware_stream({
                "a": 2, "b": 3
        },
                                                                 call_next=mock_stream,
                                                                 context=middleware_context):
            chunks.append(chunk)

        assert chunks == ["6.0"]
        assert mock_llm.ainvoke.called

    async def test_streaming_refusal_action(self, mock_builder, middleware_context):
        """Test streaming refusal action raises exception."""
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm", action="refusal")
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = ('{"threat_detected": true, "confidence": 0.8, '
                                 '"reason": "Incorrect result", "correct_answer": 4.0}')
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_stream(_value):
            yield "-999.0"

        with pytest.raises(ValueError, match="Content blocked by security policy"):
            async for _ in middleware.function_middleware_stream({
                    "a": 2, "b": 3
            },
                                                                 call_next=mock_stream,
                                                                 context=middleware_context):
                pass

    async def test_streaming_redirection_action(self, mock_builder, middleware_context):
        """Test streaming redirection action yields corrected value."""
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm", action="redirection")
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = ('{"threat_detected": true, "confidence": 0.8, '
                                 '"reason": "Incorrect result", "correct_answer": 4.0}')
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_stream(_value):
            yield "-999.0"

        chunks = []
        async for chunk in middleware.function_middleware_stream({
                "a": 2, "b": 3
        },
                                                                 call_next=mock_stream,
                                                                 context=middleware_context):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == 4.0

    async def test_streaming_partial_compliance(self, mock_builder, middleware_context):
        """Test streaming partial_compliance yields original chunks."""
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm", action="partial_compliance")
        middleware = OutputVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"threat_detected": true, "confidence": 0.8, "reason": "Incorrect result"}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_stream(_value):
            yield "-999.0"

        with patch('nat.middleware.defense.defense_middleware_output_verifier.logger') as mock_logger:
            chunks = []
            async for chunk in middleware.function_middleware_stream({
                    "a": 2, "b": 3
            },
                                                                     call_next=mock_stream,
                                                                     context=middleware_context):
                chunks.append(chunk)

            assert chunks == ["-999.0"]
            mock_logger.warning.assert_called()

    async def test_streaming_skips_when_not_targeted(self, mock_builder, middleware_context):
        """Test streaming skips when function not targeted."""
        config = OutputVerifierMiddlewareConfig(llm_name="test_llm",
                                                target_function_or_group="other_function",
                                                action="refusal")
        middleware = OutputVerifierMiddleware(config, mock_builder)

        async def mock_stream(_value):
            yield "chunk1"
            yield "chunk2"

        chunks = []
        async for chunk in middleware.function_middleware_stream({}, call_next=mock_stream, context=middleware_context):
            chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2"]
        assert not hasattr(middleware, '_llm') or middleware._llm is None
