# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# pylint: disable=unused-argument

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.embedder.azure_openai_embedder import AzureOpenAIEmbedderModelConfig
from nat.embedder.nim_embedder import NIMEmbedderModelConfig
from nat.embedder.openai_embedder import OpenAIEmbedderModelConfig
from nat.plugins.llama_index.embedder import azure_openai_llama_index
from nat.plugins.llama_index.embedder import nim_llama_index
from nat.plugins.llama_index.embedder import openai_llama_index

# ---------------------------------------------------------------------------
# OpenAI embedder → Llama-Index
# ---------------------------------------------------------------------------


class TestOpenAIEmbedderLlamaIndex:
    """Tests for the openai_llama_index embedder wrapper."""

    @pytest.fixture
    def openai_embedder_config(self):
        return OpenAIEmbedderModelConfig(model_name="text-embedding-3-small")

    @pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
    @patch("llama_index.embeddings.openai.OpenAIEmbedding")
    async def test_verify_ssl_passed_to_client(self,
                                               mock_embedding,
                                               openai_embedder_config,
                                               mock_builder,
                                               mock_httpx_async_client,
                                               mock_httpx_sync_client,
                                               verify_ssl):
        """Test that verify_ssl is passed to both sync and async httpx clients as verify."""
        openai_embedder_config.verify_ssl = verify_ssl
        async with openai_llama_index(openai_embedder_config, mock_builder):
            mock_httpx_async_client.assert_called_once()
            assert mock_httpx_async_client.call_args.kwargs["verify"] is verify_ssl
            mock_httpx_sync_client.assert_called_once()
            assert mock_httpx_sync_client.call_args.kwargs["verify"] is verify_ssl


# ---------------------------------------------------------------------------
# Azure OpenAI embedder → Llama-Index
# ---------------------------------------------------------------------------


class TestAzureOpenAIEmbedderLlamaIndex:
    """Tests for the azure_openai_llama_index embedder wrapper."""

    @pytest.fixture
    def azure_embedder_config(self):
        return AzureOpenAIEmbedderModelConfig(
            azure_deployment="text-embedding-3-small",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-01",
        )

    @pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
    @patch("llama_index.embeddings.azure_openai.AzureOpenAIEmbedding")
    async def test_verify_ssl_passed_to_client(self,
                                               mock_embedding,
                                               azure_embedder_config,
                                               mock_builder,
                                               mock_httpx_async_client,
                                               mock_httpx_sync_client,
                                               verify_ssl):
        """Test that verify_ssl is passed to both sync and async httpx clients as verify."""
        azure_embedder_config.verify_ssl = verify_ssl
        async with azure_openai_llama_index(azure_embedder_config, mock_builder):
            mock_httpx_async_client.assert_called_once()
            assert mock_httpx_async_client.call_args.kwargs["verify"] is verify_ssl
            mock_httpx_sync_client.assert_called_once()
            assert mock_httpx_sync_client.call_args.kwargs["verify"] is verify_ssl


# ---------------------------------------------------------------------------
# NIM embedder → Llama-Index
# ---------------------------------------------------------------------------


class TestNIMEmbedderLlamaIndex:
    """Tests for the nim_llama_index embedder wrapper."""

    @pytest.fixture
    def nim_embedder_config(self):
        return NIMEmbedderModelConfig(model_name="nvidia/nv-embed-qa-4")

    @patch("llama_index.embeddings.nvidia.NVIDIAEmbedding")
    async def test_verify_ssl_true_functions(self, mock_embedding, nim_embedder_config, mock_builder):
        """When verify_ssl is True, nim_llama_index creates NVIDIAEmbedding."""
        nim_embedder_config.verify_ssl = True
        mock_embedding.return_value = MagicMock()
        async with nim_llama_index(nim_embedder_config, mock_builder):
            mock_embedding.assert_called_once()

    async def test_verify_ssl_false_raises_value_error(self, nim_embedder_config, mock_builder):
        """When verify_ssl is False, nim_llama_index raises ValueError."""
        nim_embedder_config.verify_ssl = False
        with pytest.raises(ValueError, match="verify_ssl is currently not supported for NVIDIAEmbedding"):
            async with nim_llama_index(nim_embedder_config, mock_builder):
                pass
