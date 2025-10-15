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
"""Tests for SSL verification feature in LlamaIndex Embedder plugin."""

from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest

from nat.builder.builder import Builder
from nat.embedder.openai_embedder import OpenAIEmbedderModelConfig
from nat.plugins.llama_index.embedder import openai_llama_index


class TestOpenAILlamaIndexEmbedderSSLVerification:
    """Tests for SSL verification in openai_llama_index embedder function."""

    @pytest.fixture
    def mock_builder(self):
        """Create a mock Builder object."""
        return MagicMock(spec=Builder)

    @pytest.fixture
    def embedder_config(self):
        """Create an OpenAIEmbedderModelConfig instance."""
        return OpenAIEmbedderModelConfig(model_name="text-embedding-ada-002", api_key="test-key")

    @patch("llama_index.embeddings.openai.OpenAIEmbedding")
    async def test_ssl_verification_enabled_by_default(self, mock_openai_embedding, embedder_config, mock_builder):
        """Test that SSL verification is enabled by default for embedders."""
        async with openai_llama_index(embedder_config, mock_builder):
            # Verify OpenAIEmbedding was called
            mock_openai_embedding.assert_called_once()
            call_kwargs = mock_openai_embedding.call_args[1]

            # When verify_ssl is True (default), no custom http_client should be passed
            assert "http_client" not in call_kwargs

    @patch("llama_index.embeddings.openai.OpenAIEmbedding")
    async def test_ssl_verification_disabled(self, mock_openai_embedding, embedder_config, mock_builder):
        """Test that SSL verification can be disabled for embedders."""
        # Disable SSL verification
        embedder_config.verify_ssl = False

        async with openai_llama_index(embedder_config, mock_builder):
            # Verify OpenAIEmbedding was called
            mock_openai_embedding.assert_called_once()
            call_kwargs = mock_openai_embedding.call_args[1]

            # When verify_ssl is False, custom http client should be provided
            assert "http_client" in call_kwargs

            # Verify the client is an httpx client with verify=False
            http_client = call_kwargs["http_client"]
            assert isinstance(http_client, httpx.Client)

    @patch("llama_index.embeddings.openai.OpenAIEmbedding")
    async def test_verify_ssl_field_excluded_from_config(self, mock_openai_embedding, embedder_config, mock_builder):
        """Test that verify_ssl field is not passed to OpenAIEmbedding."""
        embedder_config.verify_ssl = False

        async with openai_llama_index(embedder_config, mock_builder):
            # Verify OpenAIEmbedding was called
            mock_openai_embedding.assert_called_once()
            call_kwargs = mock_openai_embedding.call_args[1]

            # verify_ssl should not be in the kwargs passed to OpenAIEmbedding
            assert "verify_ssl" not in call_kwargs
