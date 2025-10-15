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
"""Tests for SSL verification feature in LangChain LLM plugin."""

from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest

from nat.builder.builder import Builder
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.langchain.llm import openai_langchain


class TestOpenAILangChainSSLVerification:
    """Tests for SSL verification in openai_langchain function."""

    @pytest.fixture
    def mock_builder(self):
        """Create a mock Builder object."""
        return MagicMock(spec=Builder)

    @pytest.fixture
    def openai_config(self):
        """Create an OpenAIModelConfig instance."""
        return OpenAIModelConfig(model_name="gpt-4", api_key="test-key")

    @patch("langchain_openai.ChatOpenAI")
    async def test_ssl_verification_enabled_by_default(self, mock_chat_openai, openai_config, mock_builder):
        """Test that SSL verification is enabled by default."""
        async with openai_langchain(openai_config, mock_builder):
            # Verify ChatOpenAI was called
            mock_chat_openai.assert_called_once()
            call_kwargs = mock_chat_openai.call_args[1]

            # When verify_ssl is True (default), no custom http_client should be passed
            assert "http_client" not in call_kwargs
            assert "http_async_client" not in call_kwargs

    @patch("langchain_openai.ChatOpenAI")
    async def test_ssl_verification_disabled(self, mock_chat_openai, openai_config, mock_builder):
        """Test that SSL verification can be disabled."""
        # Disable SSL verification
        openai_config.verify_ssl = False

        async with openai_langchain(openai_config, mock_builder):
            # Verify ChatOpenAI was called
            mock_chat_openai.assert_called_once()
            call_kwargs = mock_chat_openai.call_args[1]

            # When verify_ssl is False, custom http clients should be provided
            assert "http_client" in call_kwargs
            assert "http_async_client" in call_kwargs

            # Verify the clients are httpx clients with verify=False
            sync_client = call_kwargs["http_client"]
            async_client = call_kwargs["http_async_client"]

            assert isinstance(sync_client, httpx.Client)
            assert isinstance(async_client, httpx.AsyncClient)

            # Check that verify is False (httpx clients don't expose this directly,
            # but we can verify they were created with the right config)
            ssl_context = sync_client._transport._pool._ssl_context
            assert ssl_context is None or not ssl_context.check_hostname

    @patch("langchain_openai.ChatOpenAI")
    async def test_verify_ssl_field_excluded_from_config(self, mock_chat_openai, openai_config, mock_builder):
        """Test that verify_ssl field is not passed to ChatOpenAI."""
        openai_config.verify_ssl = False

        async with openai_langchain(openai_config, mock_builder):
            # Verify ChatOpenAI was called
            mock_chat_openai.assert_called_once()
            call_kwargs = mock_chat_openai.call_args[1]

            # verify_ssl should not be in the kwargs passed to ChatOpenAI
            assert "verify_ssl" not in call_kwargs
            assert "ssl_verify" not in call_kwargs

    @patch("langchain_openai.ChatOpenAI")
    async def test_httpx_timeout_configuration(self, mock_chat_openai, openai_config, mock_builder):
        """Test that httpx clients are configured with appropriate timeouts."""
        openai_config.verify_ssl = False

        async with openai_langchain(openai_config, mock_builder):
            call_kwargs = mock_chat_openai.call_args[1]

            sync_client = call_kwargs["http_client"]
            async_client = call_kwargs["http_async_client"]

            # Verify timeout configuration exists
            assert sync_client.timeout is not None
            assert async_client.timeout is not None

    @patch("langchain_openai.ChatOpenAI")
    async def test_all_config_params_passed_correctly(self, mock_chat_openai, openai_config, mock_builder):
        """Test that all config parameters are passed correctly when SSL is disabled."""
        # Configure with multiple parameters
        openai_config.verify_ssl = False
        openai_config.base_url = "https://custom.endpoint.com/v1"
        openai_config.temperature = 0.7
        openai_config.seed = 42

        async with openai_langchain(openai_config, mock_builder):
            call_kwargs = mock_chat_openai.call_args[1]

            # Check that standard params are passed
            assert call_kwargs["model"] == "gpt-4"
            assert call_kwargs["api_key"] == "test-key"
            assert call_kwargs["base_url"] == "https://custom.endpoint.com/v1"
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["seed"] == 42

            # Check that SSL verification params are excluded but httpx clients are included
            assert "verify_ssl" not in call_kwargs
            assert "ssl_verify" not in call_kwargs
            assert "http_client" in call_kwargs
            assert "http_async_client" in call_kwargs

    @patch("langchain_openai.ChatOpenAI")
    async def test_openai_model_config_verify_ssl_field_exists(self, mock_chat_openai, mock_builder):
        """Test that OpenAIModelConfig has the verify_ssl field."""
        # Test with explicit True
        config = OpenAIModelConfig(model_name="gpt-4", api_key="test", verify_ssl=True)
        assert hasattr(config, "verify_ssl")
        assert config.verify_ssl is True

        # Test with explicit False
        config = OpenAIModelConfig(model_name="gpt-4", api_key="test", verify_ssl=False)
        assert config.verify_ssl is False

        # Test default value (should be True)
        config = OpenAIModelConfig(model_name="gpt-4", api_key="test")
        assert config.verify_ssl is True
