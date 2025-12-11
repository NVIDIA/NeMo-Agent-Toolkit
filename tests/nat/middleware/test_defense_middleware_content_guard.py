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
        
        async def mock_next(value):
            return "Hello world"
        
        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
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
        
        async def mock_next(value):
            return {"message": "Hello world", "status": "ok"}
        
        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        assert mock_llm.ainvoke.called
        # Should analyze only the message field
        call_args = mock_llm.ainvoke.call_args
        assert "Hello world" in str(call_args)
    
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
        
        async def mock_next(value):
            return _TestOutputModel(message="harmful content", status="ok")
        
        with patch('nat.middleware.defense_middleware_content_guard.logger') as mock_logger:
            result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
            assert mock_llm.ainvoke.called
            # Should analyze only the message field
            call_args = mock_llm.ainvoke.call_args
            assert "harmful content" in str(call_args)
    
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
        
        async def mock_next(value):
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
        
        async def mock_next(value):
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
        
        async def mock_next(value):
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
        
        async def mock_next(value):
            return "harmful content"
        
        result = await middleware.function_middleware_invoke({}, mock_next, middleware_context)
        # Should return safe refusal message
        assert "cannot" in result.lower() or "sorry" in result.lower() or "cannot assist" in result.lower()

