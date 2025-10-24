# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.adk.llm import openai_adk

# ----------------------------
# Test Fixtures and Helpers
# ----------------------------


@pytest.fixture
def mock_builder():
    """Mock builder fixture."""
    return MagicMock()


@pytest.fixture
def litellm_config():
    """Sample LiteLLM configuration for testing."""
    return OpenAIModelConfig(model_name="gpt-3.5-turbo",
                             temperature=0.7,
                             api_key="test-api-key",
                             base_url="https://api.openai.com/v1")


@pytest.fixture
def minimal_litellm_config():
    """Minimal LiteLLM configuration for testing."""
    return OpenAIModelConfig(model_name="gpt-4")


# ----------------------------
# Pytest Unit Tests
# ----------------------------


@patch('google.adk.models.lite_llm.LiteLlm')
@pytest.mark.asyncio
async def test_litellm_adk_with_full_config(mock_litellm_class, litellm_config, mock_builder):
    """Test litellm_adk function with full configuration."""
    mock_llm_instance = MagicMock()
    mock_litellm_class.return_value = mock_llm_instance

    # Use async context manager (not async for)
    async with openai_adk(litellm_config, mock_builder) as llm:
        result_llm = llm

    # Verify LiteLlm was instantiated with correct parameters
    mock_litellm_class.assert_called_once_with('gpt-3.5-turbo',
                                               top_p=1.0,
                                               temperature=0.7,
                                               api_key='test-api-key',
                                               api_base='https://api.openai.com/v1')

    # Verify the returned LLM instance
    assert result_llm == mock_llm_instance


@patch('google.adk.models.lite_llm.LiteLlm')
@pytest.mark.asyncio
async def test_litellm_adk_with_minimal_config(mock_litellm_class, minimal_litellm_config, mock_builder):
    """Test litellm_adk function with minimal configuration."""
    mock_llm_instance = MagicMock()
    mock_litellm_class.return_value = mock_llm_instance

    # Use async context manager (not async for)
    async with openai_adk(minimal_litellm_config, mock_builder) as llm:
        result_llm = llm

    # Verify LiteLlm was instantiated with default values for missing fields
    mock_litellm_class.assert_called_once_with('gpt-4', top_p=1.0, temperature=0.0)

    # Verify the returned LLM instance
    assert result_llm == mock_llm_instance


@patch('google.adk.models.lite_llm.LiteLlm')
@pytest.mark.asyncio
async def test_litellm_adk_config_exclusion(mock_litellm_class, mock_builder):
    """Test that 'type' field is excluded from config when creating LiteLlm."""
    config_with_type = OpenAIModelConfig(model_name="gpt-3.5-turbo", temperature=0.5)
    # Manually add a 'type' field to test exclusion
    config_with_type.__dict__['type'] = 'test_type'

    mock_llm_instance = MagicMock()
    mock_litellm_class.return_value = mock_llm_instance

    # Use async context manager (not async for)
    async with openai_adk(config_with_type, mock_builder) as llm:
        result_llm = llm

    # Verify LiteLlm was called (the exact parameters depend on model_dump implementation)
    mock_litellm_class.assert_called_once()
    call_args = mock_litellm_class.call_args[0]
    call_kwargs = mock_litellm_class.call_args[1]

    # Verify that 'type' is not passed to LiteLlm constructor
    assert 'type' not in call_kwargs

    # Verify expected parameters are present
    assert call_args[0] == "gpt-3.5-turbo"  # model name as first positional arg
    assert call_kwargs['temperature'] == 0.5

    # Verify the returned LLM instance
    assert result_llm == mock_llm_instance


@patch('google.adk.models.lite_llm.LiteLlm')
@pytest.mark.asyncio
async def test_litellm_adk_is_generator(mock_litellm_class, litellm_config, mock_builder):
    """Test that litellm_adk returns an async context manager."""
    mock_llm_instance = MagicMock()
    mock_litellm_class.return_value = mock_llm_instance

    # Get the context manager
    context_manager = openai_adk(litellm_config, mock_builder)

    # Verify it's an async context manager
    assert hasattr(context_manager, '__aenter__')
    assert hasattr(context_manager, '__aexit__')

    # Use the context manager to get the LLM instance
    async with context_manager as llm:
        result_llm = llm

    # Should return exactly one LLM instance
    assert result_llm == mock_llm_instance


@pytest.mark.asyncio
async def test_litellm_adk_decorator_registration():
    """Test that the litellm_adk function is properly decorated."""
    from nat.plugins.adk.llm import openai_adk

    # Verify the function has the expected attributes from the decorator
    # Note: This test verifies the decorator was applied, but the exact attributes
    # depend on the implementation of register_llm_client decorator
    assert callable(openai_adk)

    # The function should return a context manager when called (due to decorator)
    from unittest.mock import MagicMock

    from nat.llm.openai_llm import OpenAIModelConfig

    config = OpenAIModelConfig(model_name="test")
    builder = MagicMock()
    result = openai_adk(config, builder)

    # It should be an async context manager
    assert hasattr(result, '__aenter__')
    assert hasattr(result, '__aexit__')
