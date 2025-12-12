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
"""Tests for ContentSafetyGuardMiddleware field targeting and analysis."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.middleware.defense_middleware_content_guard import ContentSafetyGuardMiddleware
from nat.middleware.defense_middleware_content_guard import ContentSafetyGuardMiddlewareConfig
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
    return FunctionMiddlewareContext(
        name="my_calculator.get_random_string",
        config=MagicMock(),
        description="Get random string",
        input_schema=_TestInput,
        single_output_schema=_TestOutputModel,
        stream_output_schema=type(None)
    )


class TestContentSafetyGuardFieldTargeting:
    """Test Content Safety Guard with different field targeting scenarios."""

    @pytest.mark.asyncio
    async def test_simple_output_no_target_field(self, mock_builder, middleware_context):
        """Test analyzing simple string output without target_field."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            target_field=None,
            action="partial_compliance"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "Hello world"

        await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze the entire output string
        call_args = mock_llm.ainvoke.call_args
        assert "Hello world" in str(call_args)

    @pytest.mark.asyncio
    async def test_dict_output_with_target_field(self, mock_builder, middleware_context):
        """Test analyzing dict output with target_field."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            target_field="$.message",
            action="partial_compliance"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return {"message": "Hello world", "status": "ok"}

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the message field
        call_args = mock_llm.ainvoke.call_args
        assert "Hello world" in str(call_args)
        assert result == {"message": "Hello world", "status": "ok"}

    @pytest.mark.asyncio
    async def test_basemodel_output_with_target_field(self, mock_builder, middleware_context):
        """Test analyzing BaseModel output with target_field."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            target_field="$.message",
            action="partial_compliance"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return _TestOutputModel(message="harmful content", status="ok")

        with patch('nat.middleware.defense_middleware_content_guard.logger'):
            result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
            assert mock_llm.ainvoke.called
            # Should analyze only the message field
            call_args = mock_llm.ainvoke.call_args
            assert "harmful content" in str(call_args)
            assert isinstance(result, _TestOutputModel)
            assert result.message == "harmful content"

    @pytest.mark.asyncio
    async def test_nested_field_targeting(self, mock_builder, middleware_context):
        """Test analyzing nested field in output."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            target_field="$.data.content.text",
            action="partial_compliance"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return {
                "data": {
                    "content": {
                        "text": "Hello world",
                        "metadata": "ignored"
                    }
                }
            }

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the nested text field
        call_args = mock_llm.ainvoke.call_args
        assert "Hello world" in str(call_args)
        assert result["data"]["content"]["text"] == "Hello world"


class TestContentSafetyGuardActions:
    """Test Content Safety Guard actions."""

    @pytest.mark.asyncio
    async def test_action_partial_compliance(self, mock_builder, middleware_context):
        """Test partial_compliance action logs but allows output."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            action="partial_compliance"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "harmful content"

        with patch('nat.middleware.defense_middleware_content_guard.logger') as mock_logger:
            result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
            # Should log warning but return original output
            mock_logger.warning.assert_called()
            assert result == "harmful content"

    @pytest.mark.asyncio
    async def test_action_refusal(self, mock_builder, middleware_context):
        """Test refusal action raises ValueError."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            action="refusal"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "harmful content"

        with pytest.raises(ValueError, match="Content blocked by safety policy"):
            await middleware.function_middleware_invoke({}, mock_next, middleware_context)

    @pytest.mark.asyncio
    async def test_action_redirection(self, mock_builder, middleware_context):
        """Test redirection action replaces output with safe message."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            action="redirection"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "harmful content"

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        # Should return safe refusal message
        assert "cannot" in result.lower() or "sorry" in result.lower() or "cannot assist" in result.lower()


class TestContentSafetyGuardResponseFormats:
    """Test Content Safety Guard with different guard model response formats."""

    @pytest.mark.asyncio
    async def test_nemoguard_json_format(self, mock_builder, middleware_context):
        """Test parsing Nemoguard JSON response format."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            action="partial_compliance"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        # Nemoguard format: JSON with "User Safety" and "Safety Categories"
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"User Safety": "unsafe", "Safety Categories": "Violent, Hate"}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "harmful content"

        with patch('nat.middleware.defense_middleware_content_guard.logger') as mock_logger:
            result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
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

    @pytest.mark.asyncio
    async def test_qwen_guard_text_format(self, mock_builder, middleware_context):
        """Test parsing Qwen Guard text response format."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            action="partial_compliance"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        # Qwen Guard format: Text with "Categories:"
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe\nCategories: Violent, Hate"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "harmful content"

        with patch('nat.middleware.defense_middleware_content_guard.logger') as mock_logger:
            result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
            # Should detect unsafe and extract categories
            mock_logger.warning.assert_called()
            assert result == "harmful content"

    @pytest.mark.asyncio
    async def test_plain_safe_response(self, mock_builder, middleware_context):
        """Test parsing plain "Safe" response."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            action="partial_compliance"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "safe content"

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        assert result == "safe content"

    @pytest.mark.asyncio
    async def test_plain_unsafe_response(self, mock_builder, middleware_context):
        """Test parsing plain "Unsafe" response."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            action="partial_compliance"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Unsafe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "harmful content"

        with patch('nat.middleware.defense_middleware_content_guard.logger') as mock_logger:
            result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
            mock_logger.warning.assert_called()
            assert result == "harmful content"


class TestContentSafetyGuardTargeting:
    """Test Content Safety Guard targeting configuration."""

    @pytest.mark.asyncio
    async def test_target_function_or_group_none(self, mock_builder, middleware_context):
        """Test that None target applies to all functions."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            target_function_or_group=None,
            action="partial_compliance"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "content"

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        assert result == "content"

    @pytest.mark.asyncio
    async def test_target_function_or_group_valid(self, mock_builder, middleware_context):
        """Test targeting a valid function."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            target_function_or_group="my_calculator.get_random_string",
            action="partial_compliance"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "content"

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        assert result == "content"

    @pytest.mark.asyncio
    async def test_target_function_or_group_non_existent(self, mock_builder, middleware_context):
        """Test targeting a non-existent function skips defense."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            target_function_or_group="calculator.invalid_func",
            action="partial_compliance"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "content"

        # Should NOT apply to non-targeted function
        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert not mock_llm.ainvoke.called  # Defense should not run
        assert result == "content"

    @pytest.mark.asyncio
    async def test_target_location_input_error(self, mock_builder):
        """Test that target_location='input' raises ValidationError at config creation."""
        # Pydantic validates at config creation time, so we can't create a config with "input"
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="Input should be 'output'"):
            ContentSafetyGuardMiddlewareConfig(
                llm_name="test_llm",
                target_location="input",  # type: ignore[arg-type]
                action="partial_compliance"
            )

    @pytest.mark.asyncio
    async def test_target_location_default_output(self, mock_builder, middleware_context):
        """Test that default target_location is 'output'."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            action="partial_compliance"
        )
        # target_location not specified, should default to "output"
        assert config.target_location == "output"

        middleware = ContentSafetyGuardMiddleware(config, mock_builder)
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "content"

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        assert result == "content"

    @pytest.mark.asyncio
    async def test_target_location_explicit_output(self, mock_builder, middleware_context):
        """Test that explicit target_location='output' works."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            target_location="output",
            action="partial_compliance"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "content"

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        assert result == "content"


class TestContentSafetyGuardSimpleOutputs:
    """Test Content Safety Guard with simple output formats."""

    @pytest.mark.asyncio
    async def test_simple_string_output(self, mock_builder, middleware_context):
        """Test analyzing simple string output."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            target_field=None,
            action="partial_compliance"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return "simple string output"

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        assert result == "simple string output"

    @pytest.mark.asyncio
    async def test_simple_int_output(self, mock_builder, middleware_context):
        """Test analyzing simple int output."""
        config = ContentSafetyGuardMiddlewareConfig(
            llm_name="test_llm",
            target_field=None,
            action="partial_compliance"
        )
        middleware = ContentSafetyGuardMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Safe"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        middleware._llm = mock_llm

        async def mock_next(_value):
            return 42

        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        assert result == 42

