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
"""Tests for ContentSafetyGuardMiddleware field targeting and analysis."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.middleware.defense.defense_middleware_content_guard import ContentSafetyGuardMiddleware
from nat.middleware.defense.defense_middleware_content_guard import ContentSafetyGuardMiddlewareConfig
from nat.middleware.middleware import FunctionMiddlewareContext


class _TestInput(BaseModel):
    """Test input model."""
    request: dict


class _TestOutputModel(BaseModel):
    """Test output model."""
    message: str
    status: str


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


class TestContentSafetyGuardInvoke:
    """Test Content Safety Guard invoke behavior."""

    async def test_simple_output_no_target_field(self, mock_builder, middleware_context):
        """Test analyzing simple string output without target_field."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm", target_field=None, action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "Hello world"

        await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze the entire output string
        call_args = mock_llm.ainvoke.call_args
        assert "Hello world" in str(call_args)

    async def test_dict_output_with_target_field(self, mock_builder, middleware_context):
        """Test analyzing dict output with target_field."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm",
                                                    target_field="$.message",
                                                    action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return {"message": "Hello world", "status": "ok"}

        result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the message field
        call_args = mock_llm.ainvoke.call_args
        assert "Hello world" in str(call_args)
        assert result == {"message": "Hello world", "status": "ok"}

    async def test_basemodel_output_with_target_field(self, mock_builder, middleware_context):
        """Test analyzing BaseModel output with target_field."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm",
                                                    target_field="$.message",
                                                    action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return _TestOutputModel(message="harmful content", status="ok")

        with patch('nat.middleware.defense.defense_middleware_content_guard.logger'):
            result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
            assert mock_llm.ainvoke.called
            # Should analyze only the message field
            call_args = mock_llm.ainvoke.call_args
            assert "harmful content" in str(call_args)
            assert isinstance(result, _TestOutputModel)
            assert result.message == "harmful content"

    async def test_nested_field_targeting(self, mock_builder, middleware_context):
        """Test analyzing nested field in output."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm",
                                                    target_field="$.data.content.text",
                                                    action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return {"data": {"content": {"text": "Hello world", "metadata": "ignored"}}}

        result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the nested text field
        call_args = mock_llm.ainvoke.call_args
        assert "Hello world" in str(call_args)
        assert result["data"]["content"]["text"] == "Hello world"

    async def test_complex_nested_structure_with_field_targeting(self, mock_builder, middleware_context):
        """Test field targeting on complex nested structure with lists and dicts."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm",
                                                    target_field="$.results[0].user.message",
                                                    action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return {
                "results": [{
                    "user": {
                        "message": "harmful content", "id": 123
                    }, "metadata": {
                        "ignored": True
                    }
                }, {
                    "user": {
                        "message": "safe content", "id": 456
                    }
                }],
                "total": 2
            }

        with patch('nat.middleware.defense.defense_middleware_content_guard.logger'):
            result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
            assert mock_llm.ainvoke.called
            # Should analyze only the first result's user message
            call_args = mock_llm.ainvoke.call_args
            assert "harmful content" in str(call_args)
            # Verify structure is preserved
            assert result["results"][0]["user"]["message"] == "harmful content"
            assert result["results"][1]["user"]["message"] == "safe content"
            assert result["total"] == 2

    async def test_field_resolution_strategy_all(self, mock_builder, middleware_context):
        """Test field resolution strategy 'all' analyzes all matching fields."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm",
                                                    target_field="$.items[*].text",
                                                    target_field_resolution_strategy="all",
                                                    action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return {
                "items": [{
                    "text": "first harmful", "id": 1
                }, {
                    "text": "second harmful", "id": 2
                }, {
                    "text": "third harmful", "id": 3
                }]
            }

        with patch('nat.middleware.defense.defense_middleware_content_guard.logger'):
            result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
            assert mock_llm.ainvoke.called

            # call_args is a unittest.mock._Call object: call(args, kwargs)
            # call_args[0] is the args tuple, call_args[0][0] is the first positional argument (messages list)
            call_args = mock_llm.ainvoke.call_args
            messages = call_args[0][0]  # Extract messages list: [{"role": "user", "content": "..."}]
            content_str = messages[0]["content"]  # Extract content string sent to LLM

            # When strategy="all", extracted_value is a list: ["first harmful", "second harmful", "third harmful"]
            # This gets converted to string for analysis: "['first harmful', 'second harmful', 'third harmful']"
            # Verify all three fields are present in the content string sent to the guard model
            assert "first harmful" in content_str, f"Expected 'first harmful' in content: {content_str}"
            assert "second harmful" in content_str, f"Expected 'second harmful' in content: {content_str}"
            assert "third harmful" in content_str, f"Expected 'third harmful' in content: {content_str}"

            # Verify the defense processed all fields (logger.warning should be called for unsafe content)
            # Verify structure is preserved after processing
            assert result["items"][0]["text"] == "first harmful"
            assert result["items"][1]["text"] == "second harmful"
            assert result["items"][2]["text"] == "third harmful"

    async def test_action_partial_compliance(self, mock_builder, middleware_context):
        """Test partial_compliance action logs but allows output."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm", action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "harmful content"

        with patch('nat.middleware.defense.defense_middleware_content_guard.logger') as mock_logger:
            result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
            # Should log warning but return original output
            mock_logger.warning.assert_called()
            assert result == "harmful content"

    async def test_action_refusal(self, mock_builder, middleware_context):
        """Test refusal action raises ValueError."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm", action="refusal")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "harmful content"

        with pytest.raises(ValueError, match="Content blocked by safety policy"):
            await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)

    async def test_action_redirection(self, mock_builder, middleware_context):
        """Test redirection action replaces output with safe message."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm", action="redirection")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "harmful content"

        result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        # Should return safe refusal message
        assert "cannot" in result.lower() or "sorry" in result.lower() or "cannot assist" in result.lower()

    async def test_nemoguard_json_format(self, mock_builder, middleware_context):
        """Test parsing Nemoguard JSON response format."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm", action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        # Nemoguard format: JSON with "User Safety" and "Safety Categories"
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"User Safety": "unsafe", "Safety Categories": "Violent, Hate"}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "harmful content"

        with patch('nat.middleware.defense.defense_middleware_content_guard.logger') as mock_logger:
            result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
            # Should detect unsafe and extract categories
            mock_logger.warning.assert_called()
            # Check that categories are in the warning message
            # The warning is called with format string and args:
            # ("Content Safety Guard detected unsafe content in %s (categories: %s)",
            #  context.name, categories)
            call_args_list = mock_logger.warning.call_args_list
            # Find the call that contains categories
            found_categories = False
            for call in call_args_list:
                call_str = str(call)
                if "Violent" in call_str or "Hate" in call_str:
                    found_categories = True
                    break
            assert found_categories, f"Categories not found in warning calls: {call_args_list}"
            assert result == "harmful content"

    async def test_qwen_guard_text_format(self, mock_builder, middleware_context):
        """Test parsing Qwen Guard text response format."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm", action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        # Qwen Guard format: Text with "Categories:"
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe\nCategories: Violent, Hate"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "harmful content"

        with patch('nat.middleware.defense.defense_middleware_content_guard.logger') as mock_logger:
            result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
            # Should detect unsafe and extract categories
            mock_logger.warning.assert_called()
            assert result == "harmful content"

    async def test_plain_safe_response(self, mock_builder, middleware_context):
        """Test parsing plain "Safe" response."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm", action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "safe content"

        result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert mock_llm.ainvoke.called
        assert result == "safe content"

    async def test_plain_unsafe_response(self, mock_builder, middleware_context):
        """Test parsing plain "Unsafe" response."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm", action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "harmful content"

        with patch('nat.middleware.defense.defense_middleware_content_guard.logger') as mock_logger:
            result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
            mock_logger.warning.assert_called()
            assert result == "harmful content"

    async def test_targeting_configuration(self, mock_builder, middleware_context):
        """Test targeting configuration (function/group targeting and target_location)."""
        # Test None target applies to all functions
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm",
                                                    target_function_or_group=None,
                                                    action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "content"

        result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert mock_llm.ainvoke.called
        assert result == "content"

        # Test specific function targeting
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm",
                                                    target_function_or_group="my_calculator.get_random_string",
                                                    action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)
        middleware._llm = mock_llm
        mock_llm.ainvoke.reset_mock()

        result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert mock_llm.ainvoke.called
        assert result == "content"

        # Test non-targeted function skips defense
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm",
                                                    target_function_or_group="calculator.invalid_func",
                                                    action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)
        mock_llm.ainvoke.reset_mock()

        result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert not mock_llm.ainvoke.called  # Defense should not run
        assert result == "content"

    async def test_target_location_validation(self, mock_builder, middleware_context):
        """Test target_location validation and default behavior."""
        # Test that target_location='input' raises ValidationError
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="Input should be 'output'"):
            ContentSafetyGuardMiddlewareConfig(
                llm_name="test_llm",
                target_location="input",  # type: ignore[arg-type]
                action="partial_compliance")

        # Test default is 'output'
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm", action="partial_compliance")
        assert config.target_location == "output"

        # Test explicit 'output' works
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm",
                                                    target_location="output",
                                                    action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "content"

        result = await middleware.function_middleware_invoke({}, call_next=mock_next, context=middleware_context)
        assert mock_llm.ainvoke.called
        assert result == "content"

    async def test_non_string_output_converts_to_string(self, mock_builder, middleware_context):
        """Test that non-string outputs (int, float, dict, list) are converted to strings for analysis."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm", target_field=None, action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        # Test int
        async def mock_next_int(_value):
            return 42

        result = await middleware.function_middleware_invoke({}, call_next=mock_next_int, context=middleware_context)
        assert mock_llm.ainvoke.called
        call_args = mock_llm.ainvoke.call_args
        # Verify int was converted to string for analysis
        assert "42" in str(call_args) or '"42"' in str(call_args)
        assert result == 42

        # Test float
        mock_llm.ainvoke.reset_mock()

        async def mock_next_float(_value):
            return 3.14

        result = await middleware.function_middleware_invoke({}, call_next=mock_next_float, context=middleware_context)
        assert mock_llm.ainvoke.called
        call_args = mock_llm.ainvoke.call_args
        assert "3.14" in str(call_args) or '"3.14"' in str(call_args)
        assert result == 3.14

        # Test dict
        mock_llm.ainvoke.reset_mock()

        async def mock_next_dict(_value):
            return {"key": "value"}

        result = await middleware.function_middleware_invoke({}, call_next=mock_next_dict, context=middleware_context)
        assert mock_llm.ainvoke.called
        call_args = mock_llm.ainvoke.call_args
        # Dict should be converted to string representation
        assert "key" in str(call_args) or "value" in str(call_args)
        assert result == {"key": "value"}


class TestContentSafetyGuardStreaming:
    """Test Content Safety Guard streaming behavior."""

    async def test_streaming_safe_content(self, mock_builder, middleware_context):
        """Test streaming safe content yields original chunks."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm", action="refusal")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_stream(_value):
            yield "Hello "
            yield "world"

        chunks = []
        async for chunk in middleware.function_middleware_stream({}, call_next=mock_stream, context=middleware_context):
            chunks.append(chunk)

        assert chunks == ["Hello ", "world"]
        assert mock_llm.ainvoke.called

    async def test_streaming_refusal_action(self, mock_builder, middleware_context):
        """Test streaming refusal action raises exception."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm", action="refusal")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_stream(_value):
            yield "harmful "
            yield "content"

        with pytest.raises(ValueError, match="Content blocked by safety policy"):
            async for _ in middleware.function_middleware_stream({}, call_next=mock_stream, context=middleware_context):
                pass

    async def test_streaming_redirection_action(self, mock_builder, middleware_context):
        """Test streaming redirection action yields single redirected chunk."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm", action="redirection")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_stream(_value):
            yield "harmful "
            yield "content"

        chunks = []
        async for chunk in middleware.function_middleware_stream({}, call_next=mock_stream, context=middleware_context):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == "I'm sorry, I cannot help you with that request."

    async def test_streaming_partial_compliance(self, mock_builder, middleware_context):
        """Test streaming partial_compliance yields original chunks."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm", action="partial_compliance")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_stream(_value):
            yield "harmful "
            yield "content"

        with patch('nat.middleware.defense.defense_middleware_content_guard.logger') as mock_logger:
            chunks = []
            async for chunk in middleware.function_middleware_stream({},
                                                                     call_next=mock_stream,
                                                                     context=middleware_context):
                chunks.append(chunk)

            assert chunks == ["harmful ", "content"]
            mock_logger.warning.assert_called()

    async def test_streaming_skips_when_not_targeted(self, mock_builder, middleware_context):
        """Test streaming skips when function not targeted."""
        config = ContentSafetyGuardMiddlewareConfig(llm_name="test_llm",
                                                    target_function_or_group="other_function",
                                                    action="refusal")
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        async def mock_stream(_value):
            yield "chunk1"
            yield "chunk2"

        chunks = []
        async for chunk in middleware.function_middleware_stream({}, call_next=mock_stream, context=middleware_context):
            chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2"]
        assert not hasattr(middleware, '_llm') or middleware._llm is None
