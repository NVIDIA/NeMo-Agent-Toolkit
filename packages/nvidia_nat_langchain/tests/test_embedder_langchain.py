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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.embedder.azure_openai_embedder import AzureOpenAIEmbedderModelConfig
from nat.embedder.huggingface_embedder import HuggingFaceEmbedderConfig
from nat.embedder.nim_embedder import NIMEmbedderModelConfig
from nat.embedder.openai_embedder import OpenAIEmbedderModelConfig
from nat.plugins.langchain.embedder import azure_openai_langchain
from nat.plugins.langchain.embedder import huggingface_langchain
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


class TestHuggingFaceLangChain:

    @patch("langchain_huggingface.HuggingFaceEmbeddings")
    async def test_auto_device_is_left_to_sentence_transformers(self, mock_embeddings):
        config = HuggingFaceEmbedderConfig(model_name="BAAI/bge-small-en-v1.5", do_auto_retry=False)

        async with huggingface_langchain(config, MagicMock()):
            pass

        assert "device" not in mock_embeddings.call_args.kwargs["model_kwargs"]

    @patch("langchain_huggingface.HuggingFaceEmbeddings")
    async def test_explicit_device_is_passed(self, mock_embeddings):
        config = HuggingFaceEmbedderConfig(model_name="BAAI/bge-small-en-v1.5", device="cpu", do_auto_retry=False)

        async with huggingface_langchain(config, MagicMock()):
            pass

        assert mock_embeddings.call_args.kwargs["model_kwargs"]["device"] == "cpu"

    @patch("langchain_huggingface.HuggingFaceEmbeddings")
    async def test_max_seq_length_is_set_on_the_model(self, mock_embeddings):
        config = HuggingFaceEmbedderConfig(model_name="BAAI/bge-small-en-v1.5", max_seq_length=128, do_auto_retry=False)

        async with huggingface_langchain(config, MagicMock()):
            pass

        assert mock_embeddings.return_value._client.max_seq_length == 128

    @patch("huggingface_hub.AsyncInferenceClient")
    @patch("huggingface_hub.InferenceClient")
    @patch("langchain_huggingface.HuggingFaceEndpointEmbeddings")
    async def test_remote_endpoint_url_configures_inference_clients(self,
                                                                    mock_endpoint_embeddings: MagicMock,
                                                                    mock_sync_client: MagicMock,
                                                                    mock_async_client: MagicMock):
        config = HuggingFaceEmbedderConfig(
            endpoint_url="http://localhost:8080",
            api_key="test-token",
            timeout=45.0,
            do_auto_retry=False,
        )

        mock_instance = MagicMock()
        mock_endpoint_embeddings.return_value = mock_instance

        async with huggingface_langchain(config, MagicMock()) as client:
            assert client is mock_instance

        mock_endpoint_embeddings.assert_called_once_with(
            model=None,
            huggingfacehub_api_token="test-token",
        )
        mock_sync_client.assert_called_once_with(
            model="http://localhost:8080",
            token="test-token",
            timeout=45.0,
        )
        mock_async_client.assert_called_once_with(
            model="http://localhost:8080",
            token="test-token",
            timeout=45.0,
        )
        assert mock_instance.client is mock_sync_client.return_value
        assert mock_instance.async_client is mock_async_client.return_value

    @patch("huggingface_hub.AsyncInferenceClient")
    @patch("huggingface_hub.InferenceClient")
    @patch("langchain_huggingface.HuggingFaceEndpointEmbeddings")
    async def test_remote_endpoint_url_with_explicit_model_name(self,
                                                                mock_endpoint_embeddings: MagicMock,
                                                                mock_sync_client: MagicMock,
                                                                mock_async_client: MagicMock):
        config = HuggingFaceEmbedderConfig(
            model_name="BAAI/bge-large-en-v1.5",
            endpoint_url="https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5",
            api_key="test-token",
            timeout=90.0,
            do_auto_retry=False,
        )

        mock_instance = MagicMock()
        mock_endpoint_embeddings.return_value = mock_instance

        async with huggingface_langchain(config, MagicMock()):
            pass

        mock_endpoint_embeddings.assert_called_once_with(
            model="BAAI/bge-large-en-v1.5",
            huggingfacehub_api_token="test-token",
        )
        mock_sync_client.assert_called_once_with(
            model="https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5",
            token="test-token",
            timeout=90.0,
        )
        mock_async_client.assert_called_once_with(
            model="https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5",
            token="test-token",
            timeout=90.0,
        )
