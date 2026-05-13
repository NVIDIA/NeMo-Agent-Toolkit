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

from unittest.mock import patch

import pytest

from nat.embedder.azure_openai_embedder import AzureOpenAIEmbedderModelConfig
from nat.embedder.nim_embedder import NIMEmbedderModelConfig
from nat.embedder.openai_embedder import OpenAIEmbedderModelConfig
from nat.plugins.langchain.embedder import azure_openai_langchain
from nat.plugins.langchain.embedder import nim_langchain
from nat.plugins.langchain.embedder import openai_langchain

# ---------------------------------------------------------------------------
# OpenAI embedder → LangChain
# ---------------------------------------------------------------------------


class TestOpenAIEmbedderLangChain:
    """Tests for the openai_langchain embedder wrapper."""

    @pytest.fixture
    def openai_embedder_config(self):
        return OpenAIEmbedderModelConfig(model_name="text-embedding-3-small")

    @pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
    @patch("langchain_openai.OpenAIEmbeddings")
    async def test_verify_ssl_passed_to_client(self,
                                               mock_embeddings,
                                               openai_embedder_config,
                                               mock_builder,
                                               mock_httpx_async_client,
                                               mock_httpx_sync_client,
                                               verify_ssl):
        """Test that verify_ssl is passed to both sync and async httpx clients as verify."""
        openai_embedder_config.verify_ssl = verify_ssl
        async with openai_langchain(openai_embedder_config, mock_builder):
            mock_httpx_async_client.assert_called_once()
            assert mock_httpx_async_client.call_args.kwargs["verify"] is verify_ssl
            mock_httpx_sync_client.assert_called_once()
            assert mock_httpx_sync_client.call_args.kwargs["verify"] is verify_ssl


# ---------------------------------------------------------------------------
# Azure OpenAI embedder → LangChain
# ---------------------------------------------------------------------------


class TestAzureOpenAIEmbedderLangChain:
    """Tests for the azure_openai_langchain embedder wrapper."""

    @pytest.fixture
    def azure_embedder_config(self):
        return AzureOpenAIEmbedderModelConfig(
            azure_deployment="text-embedding-3-small",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-01",
        )

    @pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
    @patch("langchain_openai.AzureOpenAIEmbeddings")
    async def test_verify_ssl_passed_to_client(self,
                                               mock_embeddings,
                                               azure_embedder_config,
                                               mock_builder,
                                               mock_httpx_async_client,
                                               mock_httpx_sync_client,
                                               verify_ssl):
        """Test that verify_ssl is passed to both sync and async httpx clients as verify."""
        azure_embedder_config.verify_ssl = verify_ssl
        async with azure_openai_langchain(azure_embedder_config, mock_builder):
            mock_httpx_async_client.assert_called_once()
            assert mock_httpx_async_client.call_args.kwargs["verify"] is verify_ssl
            mock_httpx_sync_client.assert_called_once()
            assert mock_httpx_sync_client.call_args.kwargs["verify"] is verify_ssl


# ---------------------------------------------------------------------------
# NIM embedder → LangChain
# ---------------------------------------------------------------------------


class TestNIMEmbedderLangChain:
    """Tests for the nim_langchain embedder wrapper."""

    @pytest.fixture
    def nim_embedder_config(self):
        return NIMEmbedderModelConfig(model_name="nvidia/nv-embed-qa-4")

    @pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
    @patch("langchain_nvidia_ai_endpoints.NVIDIAEmbeddings")
    async def test_verify_ssl_passed_to_nvidia_embeddings(self,
                                                          mock_embeddings,
                                                          nim_embedder_config,
                                                          mock_builder,
                                                          verify_ssl):
        """Test that verify_ssl is passed to NVIDIAEmbeddings as a keyword argument."""
        nim_embedder_config.verify_ssl = verify_ssl
        async with nim_langchain(nim_embedder_config, mock_builder):
            mock_embeddings.assert_called_once()
            assert mock_embeddings.call_args.kwargs["verify_ssl"] is verify_ssl
