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
"""Tests for SSL verification feature in LangChain Embedder plugin."""

from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest

from nat.builder.builder import Builder
from nat.embedder.openai_embedder import OpenAIEmbedderModelConfig
from nat.plugins.langchain.embedder import openai_langchain


class TestOpenAILangChainEmbedderSSLVerification:
    """Tests for SSL verification in openai_langchain embedder function."""

    @pytest.fixture
    def mock_builder(self):
        """Create a mock Builder object."""
        return MagicMock(spec=Builder)

    @pytest.fixture
    def embedder_config(self):
        """Create an OpenAIEmbedderModelConfig instance."""
        return OpenAIEmbedderModelConfig(model_name="text-embedding-ada-002", api_key="test-key")

    @patch("langchain_openai.OpenAIEmbeddings")
    async def test_ssl_verification_enabled_by_default(self, mock_openai_embeddings, embedder_config, mock_builder):
        """Test that SSL verification is enabled by default for embedders."""
        async with openai_langchain(embedder_config, mock_builder):
            # Verify OpenAIEmbeddings was called
            mock_openai_embeddings.assert_called_once()
            call_kwargs = mock_openai_embeddings.call_args[1]

            # When verify_ssl is True (default), no custom http_client should be passed
            assert "http_client" not in call_kwargs

    @patch("langchain_openai.OpenAIEmbeddings")
    async def test_ssl_verification_disabled(self, mock_openai_embeddings, embedder_config, mock_builder):
        """Test that SSL verification can be disabled for embedders."""
        # Disable SSL verification
        embedder_config.verify_ssl = False

        async with openai_langchain(embedder_config, mock_builder):
            # Verify OpenAIEmbeddings was called
            mock_openai_embeddings.assert_called_once()
            call_kwargs = mock_openai_embeddings.call_args[1]

            # When verify_ssl is False, custom http client should be provided
            assert "http_client" in call_kwargs

            # Verify the client is an httpx client with verify=False
            sync_client = call_kwargs["http_client"]
            assert isinstance(sync_client, httpx.Client)

    @patch("langchain_openai.OpenAIEmbeddings")
    async def test_verify_ssl_field_excluded_from_config(self, mock_openai_embeddings, embedder_config, mock_builder):
        """Test that verify_ssl field is not passed to OpenAIEmbeddings."""
        embedder_config.verify_ssl = False

        async with openai_langchain(embedder_config, mock_builder):
            # Verify OpenAIEmbeddings was called
            mock_openai_embeddings.assert_called_once()
            call_kwargs = mock_openai_embeddings.call_args[1]

            # verify_ssl should not be in the kwargs passed to OpenAIEmbeddings
            assert "verify_ssl" not in call_kwargs

    @patch("langchain_openai.OpenAIEmbeddings")
    async def test_embedder_config_params_passed_correctly(self, mock_openai_embeddings, embedder_config, mock_builder):
        """Test that all embedder config parameters are passed correctly when SSL is disabled."""
        # Configure with multiple parameters
        embedder_config.verify_ssl = False
        embedder_config.base_url = "https://custom.endpoint.com/v1"

        async with openai_langchain(embedder_config, mock_builder):
            call_kwargs = mock_openai_embeddings.call_args[1]

            # Check that standard params are passed
            assert call_kwargs["model"] == "text-embedding-ada-002"
            assert call_kwargs["api_key"] == "test-key"
            assert call_kwargs["base_url"] == "https://custom.endpoint.com/v1"

            # Check that SSL verification param is excluded but httpx client is included
            assert "verify_ssl" not in call_kwargs
            assert "http_client" in call_kwargs

    @patch("langchain_openai.OpenAIEmbeddings")
    async def test_openai_embedder_config_verify_ssl_field_exists(self, mock_openai_embeddings, mock_builder):
        """Test that OpenAIEmbedderModelConfig has the verify_ssl field."""
        # Test with explicit True
        config = OpenAIEmbedderModelConfig(model_name="text-embedding-ada-002", api_key="test", verify_ssl=True)
        assert hasattr(config, "verify_ssl")
        assert config.verify_ssl is True

        # Test with explicit False
        config = OpenAIEmbedderModelConfig(model_name="text-embedding-ada-002", api_key="test", verify_ssl=False)
        assert config.verify_ssl is False

        # Test default value (should be True)
        config = OpenAIEmbedderModelConfig(model_name="text-embedding-ada-002", api_key="test")
        assert config.verify_ssl is True
