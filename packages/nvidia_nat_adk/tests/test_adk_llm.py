# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nat.llm.dynamo_llm import DynamoModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.adk.llm import dynamo_adk
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
    mock_litellm_class.assert_called_once_with('gpt-4')

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


# ----------------------------
# Dynamo ADK Tests
# ----------------------------


class TestDynamoAdk:
    """Tests for the dynamo_adk wrapper."""

    @pytest.fixture
    def mock_builder(self):
        return MagicMock()

    @pytest.fixture
    def dynamo_cfg_no_prefix(self):
        """Dynamo config without prefix template (no header injection)."""
        return DynamoModelConfig(
            model_name="test-model",
            base_url="http://localhost:8000/v1",
            prefix_template=None,
        )

    @pytest.fixture
    def dynamo_cfg_with_prefix(self):
        """Dynamo config with prefix template (enables header injection)."""
        return DynamoModelConfig(
            model_name="test-model",
            base_url="http://localhost:8000/v1",
            prefix_template="session-{uuid}",
            prefix_total_requests=15,
            prefix_osl="HIGH",
            prefix_iat="LOW",
        )

    @patch('google.adk.models.lite_llm.LiteLlm')
    @pytest.mark.asyncio
    async def test_basic_creation_without_prefix(self, mock_litellm_class, dynamo_cfg_no_prefix, mock_builder):
        """Wrapper should create LiteLlm without extra_headers when no prefix template."""
        mock_llm_instance = MagicMock()
        mock_litellm_class.return_value = mock_llm_instance

        async with dynamo_adk(dynamo_cfg_no_prefix, mock_builder) as client:
            mock_litellm_class.assert_called_once()
            kwargs = mock_litellm_class.call_args.kwargs

            assert mock_litellm_class.call_args.args[0] == "test-model"
            assert kwargs["api_base"] == "http://localhost:8000/v1"
            # Should NOT have extra_headers
            assert "extra_headers" not in kwargs
            assert client is mock_llm_instance

    @patch('google.adk.models.lite_llm.LiteLlm')
    @pytest.mark.asyncio
    async def test_creation_with_prefix_template(self, mock_litellm_class, dynamo_cfg_with_prefix, mock_builder):
        """Wrapper should create LiteLlm with extra_headers when prefix template is set."""
        mock_llm_instance = MagicMock()
        mock_litellm_class.return_value = mock_llm_instance

        async with dynamo_adk(dynamo_cfg_with_prefix, mock_builder) as client:
            mock_litellm_class.assert_called_once()
            kwargs = mock_litellm_class.call_args.kwargs

            # Should have extra_headers with Dynamo headers
            assert "extra_headers" in kwargs
            headers = kwargs["extra_headers"]

            assert "x-prefix-id" in headers
            assert headers["x-prefix-id"].startswith("session-")
            assert headers["x-prefix-total-requests"] == "15"
            assert headers["x-prefix-osl"] == "HIGH"
            assert headers["x-prefix-iat"] == "LOW"

            assert client is mock_llm_instance

    @patch('google.adk.models.lite_llm.LiteLlm')
    @pytest.mark.asyncio
    async def test_excludes_dynamo_specific_fields(self, mock_litellm_class, dynamo_cfg_with_prefix, mock_builder):
        """Dynamo-specific fields should be excluded from LiteLlm kwargs.

        DynamoModelConfig has fields (prefix_template, prefix_total_requests, prefix_osl,
        prefix_iat, request_timeout) that are only used internally by NAT to configure
        the Dynamo headers. These fields must NOT be passed directly to LiteLlm because
        LiteLlm doesn't understand them - they're NAT-specific configuration.

        This test ensures the `exclude` set in model_dump() properly filters these fields.
        """
        mock_llm_instance = MagicMock()
        mock_litellm_class.return_value = mock_llm_instance

        async with dynamo_adk(dynamo_cfg_with_prefix, mock_builder):
            pass

        kwargs = mock_litellm_class.call_args.kwargs

        # These Dynamo-specific fields should NOT be passed to LiteLlm
        assert "prefix_template" not in kwargs
        assert "prefix_total_requests" not in kwargs
        assert "prefix_osl" not in kwargs
        assert "prefix_iat" not in kwargs
        assert "request_timeout" not in kwargs

    @patch('google.adk.models.lite_llm.LiteLlm')
    @pytest.mark.asyncio
    async def test_prefix_id_is_unique_per_instance(self, mock_litellm_class, mock_builder):
        """Each LiteLlm instance should get a unique prefix ID."""
        mock_llm_instance = MagicMock()
        mock_litellm_class.return_value = mock_llm_instance

        config = DynamoModelConfig(
            model_name="test-model",
            prefix_template="session-{uuid}",
        )

        prefix_ids = set()

        # Create multiple instances
        for _ in range(5):
            async with dynamo_adk(config, mock_builder):
                pass
            headers = mock_litellm_class.call_args.kwargs.get("extra_headers", {})
            if "x-prefix-id" in headers:
                prefix_ids.add(headers["x-prefix-id"])

        # All IDs should be unique
        assert len(prefix_ids) == 5

    @pytest.mark.asyncio
    async def test_dynamo_adk_decorator_registration(self):
        """Test that the dynamo_adk function is properly decorated."""
        from nat.plugins.adk.llm import dynamo_adk

        assert callable(dynamo_adk)

        config = DynamoModelConfig(model_name="test")
        builder = MagicMock()
        result = dynamo_adk(config, builder)

        # It should be an async context manager
        assert hasattr(result, '__aenter__')
        assert hasattr(result, '__aexit__')
