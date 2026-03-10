# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# pylint: disable=unused-argument, not-async-context-manager

import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.llm import APITypeEnum
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.dynamo_llm import DynamoModelConfig
from nat.llm.litellm_llm import LiteLlmModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.langchain.llm import aws_bedrock_langchain
from nat.plugins.langchain.llm import azure_openai_langchain
from nat.plugins.langchain.llm import dynamo_langchain
from nat.plugins.langchain.llm import litellm_langchain
from nat.plugins.langchain.llm import nim_langchain
from nat.plugins.langchain.llm import openai_langchain

# ---------------------------------------------------------------------------
# NIM → LangChain wrapper tests
# ---------------------------------------------------------------------------


class TestNimLangChain:
    """Tests for the nim_langchain wrapper."""

    @pytest.fixture
    def nim_cfg(self):
        # Default API type is CHAT_COMPLETION
        return NIMModelConfig(model_name="nemotron-3b-chat")

    @pytest.fixture
    def nim_cfg_wrong_api(self):
        # Purposely create a config that violates the API-type requirement
        return NIMModelConfig(model_name="nemotron-3b-chat", api_type=APITypeEnum.RESPONSES)

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    async def test_basic_creation(self, mock_chat, nim_cfg, mock_builder):
        """Wrapper should yield a ChatNVIDIA client with the dumped kwargs."""
        async with nim_langchain(nim_cfg, mock_builder) as client:
            mock_chat.assert_called_once()
            kwargs = mock_chat.call_args.kwargs
            print(kwargs)
            assert kwargs["model"] == "nemotron-3b-chat"
            assert client is mock_chat.return_value

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    async def test_api_type_validation(self, mock_chat, nim_cfg_wrong_api, mock_builder):
        """Non-chat-completion API types must raise a ValueError."""
        with pytest.raises(ValueError):
            async with nim_langchain(nim_cfg_wrong_api, mock_builder):
                pass
        mock_chat.assert_not_called()

    @pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    async def test_verify_ssl_passed_to_chat_nvidia(self, mock_chat, nim_cfg, mock_builder, verify_ssl):
        """Test that verify_ssl is passed to ChatNVIDIA."""
        nim_cfg.verify_ssl = verify_ssl
        async with nim_langchain(nim_cfg, mock_builder):
            pass
        mock_chat.assert_called_once()
        assert mock_chat.call_args.kwargs["verify_ssl"] is verify_ssl


# ---------------------------------------------------------------------------
# OpenAI → LangChain wrapper tests
# ---------------------------------------------------------------------------


class TestOpenAILangChain:
    """Tests for the openai_langchain wrapper."""

    @pytest.fixture
    def oa_cfg(self):
        return OpenAIModelConfig(model_name="gpt-4o-mini")

    @pytest.fixture
    def oa_cfg_responses(self):
        # Explicitly set RESPONSES API and stream=True to test the branch logic.
        return OpenAIModelConfig(
            model_name="gpt-4o-mini",
            api_type=APITypeEnum.RESPONSES,
            stream=True,
            temperature=0.2,
        )

    @patch("langchain_openai.ChatOpenAI")
    async def test_basic_creation(self, mock_chat, oa_cfg, mock_builder):
        """Default kwargs (stream_usage=True) and config kwargs must reach ChatOpenAI."""
        async with openai_langchain(oa_cfg, mock_builder) as client:
            mock_chat.assert_called_once()
            kwargs = mock_chat.call_args.kwargs
            assert kwargs["model"] == "gpt-4o-mini"
            # default injected by wrapper:
            assert kwargs["stream_usage"] is True
            assert client is mock_chat.return_value

    @patch("langchain_openai.ChatOpenAI")
    async def test_responses_branch(self, mock_chat, oa_cfg_responses, mock_builder):
        """When APIType==RESPONSES, special flags are added and stream is forced False."""
        # Silence the warning that the wrapper logs when it toggles stream.
        with patch.object(logging.getLogger("nat.plugins.langchain.llm"), "warning"):
            async with openai_langchain(oa_cfg_responses, mock_builder):
                pass

        kwargs = mock_chat.call_args.kwargs
        assert kwargs["use_responses_api"] is True
        assert kwargs["use_previous_response_id"] is True
        # Other original kwargs remain unchanged
        assert kwargs["temperature"] == 0.2
        assert kwargs["stream_usage"] is True

    @pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
    @patch("langchain_openai.ChatOpenAI")
    async def test_verify_ssl_passed_to_client(self,
                                               mock_chat,
                                               oa_cfg,
                                               mock_builder,
                                               mock_httpx_async_client,
                                               verify_ssl):
        """Test that verify_ssl is passed to the underlying httpx.AsyncClient as verify."""
        mock_httpx_async_client.aclose = AsyncMock()
        oa_cfg.verify_ssl = verify_ssl
        async with openai_langchain(oa_cfg, mock_builder):
            pass
        mock_httpx_async_client.assert_called_once()
        assert mock_httpx_async_client.call_args.kwargs["verify"] is verify_ssl


# ---------------------------------------------------------------------------
# Azure OpenAI → LangChain wrapper tests
# ---------------------------------------------------------------------------


class TestAzureOpenAILangChain:
    """Tests for the azure_openai_langchain wrapper."""

    @pytest.fixture
    def azure_cfg(self):
        return AzureOpenAIModelConfig(
            azure_deployment="gpt-4",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-01",
        )

    @pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
    @patch("langchain_openai.AzureChatOpenAI")
    async def test_verify_ssl_passed_to_client(self,
                                               mock_chat,
                                               azure_cfg,
                                               mock_builder,
                                               mock_httpx_async_client,
                                               verify_ssl):
        """Test that verify_ssl is passed to the underlying httpx.AsyncClient as verify."""
        mock_httpx_async_client.aclose = AsyncMock()
        azure_cfg.verify_ssl = verify_ssl
        async with azure_openai_langchain(azure_cfg, mock_builder):
            pass
        mock_httpx_async_client.assert_called_once()
        assert mock_httpx_async_client.call_args.kwargs["verify"] is verify_ssl


# ---------------------------------------------------------------------------
# AWS Bedrock → LangChain wrapper tests
# ---------------------------------------------------------------------------


class TestBedrockLangChain:
    """Tests for the aws_bedrock_langchain wrapper."""

    @pytest.fixture
    def bedrock_cfg(self):
        return AWSBedrockModelConfig(model_name="ai21.j2-ultra")

    @pytest.fixture
    def bedrock_cfg_wrong_api(self):
        return AWSBedrockModelConfig(model_name="ai21.j2-ultra", api_type=APITypeEnum.RESPONSES)

    @patch("langchain_aws.ChatBedrockConverse")
    async def test_basic_creation(self, mock_chat, bedrock_cfg, mock_builder):
        async with aws_bedrock_langchain(bedrock_cfg, mock_builder) as client:
            mock_chat.assert_called_once()
            kwargs = mock_chat.call_args.kwargs
            assert kwargs["model"] == "ai21.j2-ultra"
            assert client is mock_chat.return_value

    @patch("langchain_aws.ChatBedrockConverse")
    async def test_api_type_validation(self, mock_chat, bedrock_cfg_wrong_api, mock_builder):
        with pytest.raises(ValueError):
            async with aws_bedrock_langchain(bedrock_cfg_wrong_api, mock_builder):
                pass
        mock_chat.assert_not_called()


# ---------------------------------------------------------------------------
# Dynamo → LangChain wrapper tests
# ---------------------------------------------------------------------------


class TestDynamoLangChain:
    """Tests for the dynamo_langchain wrapper."""

    @pytest.fixture
    def dynamo_cfg_no_prefix(self):
        """Dynamo config with nvext hints disabled (no nvext request-body injection)."""
        return DynamoModelConfig(
            model_name="test-model",
            base_url="http://localhost:8000/v1",
        )

    @pytest.fixture
    def dynamo_cfg_with_prefix(self):
        """Dynamo config with nvext hints enabled (injects nvext fields into the JSON request body)."""
        return DynamoModelConfig(
            model_name="test-model",
            base_url="http://localhost:8000/v1",
            enable_nvext_hints=True,
            nvext_prefix_id_template="session-{uuid}",
            nvext_prefix_total_requests=15,
            nvext_prefix_osl=2048,
            nvext_prefix_iat=50,
            request_timeout=300.0,
        )

    @pytest.fixture
    def dynamo_cfg_responses_api(self):
        """Dynamo config with RESPONSES API type."""
        return DynamoModelConfig(
            model_name="test-model",
            base_url="http://localhost:8000/v1",
            api_type=APITypeEnum.RESPONSES,
            enable_nvext_hints=True,
            nvext_prefix_id_template="session-{uuid}",
        )

    @patch("langchain_openai.ChatOpenAI")
    async def test_basic_creation_without_prefix(self, mock_chat, dynamo_cfg_no_prefix, mock_builder):
        """Wrapper should create ChatOpenAI with httpx client (no Dynamo transport when nvext hints disabled)."""
        async with dynamo_langchain(dynamo_cfg_no_prefix, mock_builder) as client:
            mock_chat.assert_called_once()
            kwargs = mock_chat.call_args.kwargs

            assert kwargs["model"] == "test-model"
            assert kwargs["base_url"] == "http://localhost:8000/v1"
            assert kwargs["stream_usage"] is True
            # Always passes an httpx client; when enable_nvext_hints=False it has no _DynamoTransport
            assert "http_async_client" in kwargs
            assert client is mock_chat.return_value

    @patch("nat.plugins.langchain.llm._create_httpx_client_with_dynamo_hooks")
    @patch("langchain_openai.ChatOpenAI")
    async def test_creation_with_prefix_template(self,
                                                 mock_chat,
                                                 mock_create_client,
                                                 dynamo_cfg_with_prefix,
                                                 mock_builder,
                                                 mock_httpx_async_client):
        """Wrapper should create ChatOpenAI with custom httpx client when nvext hints enabled."""

        async def _aexit(*a, **k):
            await mock_httpx_async_client.aclose()

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_httpx_async_client
        mock_cm.__aexit__ = AsyncMock(side_effect=_aexit)
        mock_create_client.return_value = mock_cm

        async with dynamo_langchain(dynamo_cfg_with_prefix, mock_builder) as client:
            mock_create_client.assert_called_once_with(dynamo_cfg_with_prefix)

            # Verify ChatOpenAI was called with the custom httpx client
            mock_chat.assert_called_once()
            kwargs = mock_chat.call_args.kwargs

            assert kwargs["model"] == "test-model"
            assert kwargs["http_async_client"] is mock_httpx_async_client
            assert client is mock_chat.return_value

        # Verify the httpx client was properly closed
        mock_httpx_async_client.aclose.assert_awaited_once()

    @patch("nat.plugins.langchain.llm._create_httpx_client_with_dynamo_hooks")
    @patch("langchain_openai.ChatOpenAI")
    async def test_responses_api_branch(self, mock_chat, mock_create_client, dynamo_cfg_responses_api, mock_builder):
        """When APIType==RESPONSES, special flags should be added."""
        mock_httpx_client = MagicMock()
        mock_httpx_client.aclose = AsyncMock()

        async def _aexit(*a, **k):
            await mock_httpx_client.aclose()

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_httpx_client
        mock_cm.__aexit__ = AsyncMock(side_effect=_aexit)
        mock_create_client.return_value = mock_cm

        async with dynamo_langchain(dynamo_cfg_responses_api, mock_builder):
            pass

        kwargs = mock_chat.call_args.kwargs
        assert kwargs["use_responses_api"] is True
        assert kwargs["use_previous_response_id"] is True
        assert kwargs["stream_usage"] is True

        # Verify the httpx client was properly closed
        mock_httpx_client.aclose.assert_awaited_once()

    @patch("nat.plugins.langchain.llm._create_httpx_client_with_dynamo_hooks")
    @patch("langchain_openai.ChatOpenAI")
    async def test_excludes_dynamo_specific_fields(self,
                                                   mock_chat,
                                                   mock_create_client,
                                                   dynamo_cfg_with_prefix,
                                                   mock_builder):
        """Dynamo-specific fields should be excluded from ChatOpenAI kwargs.

        DynamoModelConfig has fields (enable_nvext_hints, nvext_prefix_id_template,
        nvext_prefix_total_requests, nvext_prefix_osl, nvext_prefix_iat, request_timeout)
        that are only used internally by NAT to configure the custom httpx client for
        Dynamo nvext request-body injection (injects nvext.agent_hints / nvext.cache_control
        into the JSON body). These fields must NOT be passed to ChatOpenAI because:

        1. ChatOpenAI doesn't understand them and would error or ignore them
        2. They configure NAT's nvext request-body injection behavior, not the LLM client itself

        This test ensures the `exclude` set in model_dump() properly filters these fields.
        If someone accidentally removes a field from the exclude set, this test will fail.
        """
        mock_httpx_client = MagicMock()
        mock_httpx_client.aclose = AsyncMock()

        async def _aexit(*a, **k):
            await mock_httpx_client.aclose()

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_httpx_client
        mock_cm.__aexit__ = AsyncMock(side_effect=_aexit)
        mock_create_client.return_value = mock_cm

        async with dynamo_langchain(dynamo_cfg_with_prefix, mock_builder):
            pass

        kwargs = mock_chat.call_args.kwargs

        # These Dynamo-specific fields should NOT be passed to ChatOpenAI
        assert "nvext_prefix_id_template" not in kwargs
        assert "nvext_prefix_total_requests" not in kwargs
        assert "nvext_prefix_osl" not in kwargs
        assert "nvext_prefix_iat" not in kwargs
        assert "enable_nvext_hints" not in kwargs
        assert "request_timeout" not in kwargs

        # Verify the httpx client was properly closed
        mock_httpx_client.aclose.assert_awaited_once()

    @pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
    @patch("langchain_openai.ChatOpenAI")
    async def test_verify_ssl_passed_to_client(self,
                                               mock_chat,
                                               dynamo_cfg_no_prefix,
                                               mock_builder,
                                               mock_httpx_async_client,
                                               verify_ssl):
        """Test that verify_ssl is passed to the underlying httpx.AsyncClient as verify."""
        dynamo_cfg_no_prefix.verify_ssl = verify_ssl
        async with dynamo_langchain(dynamo_cfg_no_prefix, mock_builder):
            pass
        mock_httpx_async_client.assert_called_once()
        assert mock_httpx_async_client.call_args.kwargs["verify"] is verify_ssl


# ---------------------------------------------------------------------------
# LiteLLM → LangChain wrapper tests
# ---------------------------------------------------------------------------


class TestLiteLlmLangChain:
    """Tests for the litellm_langchain wrapper."""

    @pytest.fixture
    def litellm_cfg(self):
        return LiteLlmModelConfig(model_name="gpt-4", base_url="http://localhost:4000", api_key="test-key")

    @pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
    @patch("nat.llm.utils.http_client._handle_litellm_verify_ssl")
    @patch("langchain_litellm.ChatLiteLLM")
    async def test_verify_ssl_calls_handle_litellm_verify_ssl(self,
                                                              mock_chat,
                                                              mock_handle_verify_ssl,
                                                              litellm_cfg,
                                                              mock_builder,
                                                              verify_ssl):
        """Test that litellm_langchain calls _handle_litellm_verify_ssl with the config's verify_ssl value."""
        litellm_cfg.verify_ssl = verify_ssl
        async with litellm_langchain(litellm_cfg, mock_builder):
            mock_handle_verify_ssl.assert_called_once_with(litellm_cfg)


# ---------------------------------------------------------------------------
# Registration decorator sanity check
# ---------------------------------------------------------------------------


@patch("nat.cli.type_registry.GlobalTypeRegistry")
def test_decorator_registration(mock_global_registry):
    """Ensure register_llm_client decorators registered the LangChain wrappers."""
    registry = MagicMock()
    mock_global_registry.get.return_value = registry

    registry._llm_client_map = {
        (NIMModelConfig, LLMFrameworkEnum.LANGCHAIN): nim_langchain,
        (OpenAIModelConfig, LLMFrameworkEnum.LANGCHAIN): openai_langchain,
        (AWSBedrockModelConfig, LLMFrameworkEnum.LANGCHAIN): aws_bedrock_langchain,
        (DynamoModelConfig, LLMFrameworkEnum.LANGCHAIN): dynamo_langchain,
    }

    assert registry._llm_client_map[(NIMModelConfig, LLMFrameworkEnum.LANGCHAIN)] is nim_langchain
    assert registry._llm_client_map[(OpenAIModelConfig, LLMFrameworkEnum.LANGCHAIN)] is openai_langchain
    assert registry._llm_client_map[(AWSBedrockModelConfig, LLMFrameworkEnum.LANGCHAIN)] is aws_bedrock_langchain
    assert registry._llm_client_map[(DynamoModelConfig, LLMFrameworkEnum.LANGCHAIN)] is dynamo_langchain
