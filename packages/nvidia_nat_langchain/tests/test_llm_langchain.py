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

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.llm import APITypeEnum
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.llm.dynamo_llm import DynamoModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.langchain.llm import aws_bedrock_langchain
from nat.plugins.langchain.llm import dynamo_langchain
from nat.plugins.langchain.llm import nim_langchain
from nat.plugins.langchain.llm import openai_langchain

# ---------------------------------------------------------------------------
# NIM → LangChain wrapper tests
# ---------------------------------------------------------------------------


class TestNimLangChain:
    """Tests for the nim_langchain wrapper."""

    @pytest.fixture
    def mock_builder(self):
        return MagicMock(spec=Builder)

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


# ---------------------------------------------------------------------------
# OpenAI → LangChain wrapper tests
# ---------------------------------------------------------------------------


class TestOpenAILangChain:
    """Tests for the openai_langchain wrapper."""

    @pytest.fixture
    def mock_builder(self):
        return MagicMock(spec=Builder)

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


# ---------------------------------------------------------------------------
# AWS Bedrock → LangChain wrapper tests
# ---------------------------------------------------------------------------


class TestBedrockLangChain:
    """Tests for the aws_bedrock_langchain wrapper."""

    @pytest.fixture
    def mock_builder(self):
        return MagicMock(spec=Builder)

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
    def mock_builder(self):
        return MagicMock(spec=Builder)

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
            request_timeout=300.0,
        )

    @pytest.fixture
    def dynamo_cfg_responses_api(self):
        """Dynamo config with RESPONSES API type."""
        return DynamoModelConfig(
            model_name="test-model",
            base_url="http://localhost:8000/v1",
            api_type=APITypeEnum.RESPONSES,
            prefix_template="session-{uuid}",
        )

    @patch("langchain_openai.ChatOpenAI")
    async def test_basic_creation_without_prefix(self, mock_chat, dynamo_cfg_no_prefix, mock_builder):
        """Wrapper should create ChatOpenAI without custom httpx client when no prefix template."""
        async with dynamo_langchain(dynamo_cfg_no_prefix, mock_builder) as client:
            mock_chat.assert_called_once()
            kwargs = mock_chat.call_args.kwargs

            assert kwargs["model"] == "test-model"
            assert kwargs["base_url"] == "http://localhost:8000/v1"
            assert kwargs["stream_usage"] is True
            # Should NOT have custom httpx client
            assert "http_async_client" not in kwargs
            assert client is mock_chat.return_value

    @patch("nat.plugins.langchain.llm.create_httpx_client_with_dynamo_hooks")
    @patch("langchain_openai.ChatOpenAI")
    async def test_creation_with_prefix_template(self,
                                                 mock_chat,
                                                 mock_create_client,
                                                 dynamo_cfg_with_prefix,
                                                 mock_builder):
        """Wrapper should create ChatOpenAI with custom httpx client when prefix template is set."""
        mock_httpx_client = MagicMock()
        mock_httpx_client.aclose = AsyncMock()  # Make aclose awaitable
        mock_create_client.return_value = mock_httpx_client

        async with dynamo_langchain(dynamo_cfg_with_prefix, mock_builder) as client:
            # Verify httpx client was created with correct parameters
            mock_create_client.assert_called_once_with(
                prefix_template="session-{uuid}",
                total_requests=15,
                osl="HIGH",
                iat="LOW",
                timeout=300.0,
            )

            # Verify ChatOpenAI was called with the custom httpx client
            mock_chat.assert_called_once()
            kwargs = mock_chat.call_args.kwargs

            assert kwargs["model"] == "test-model"
            assert kwargs["http_async_client"] is mock_httpx_client
            assert client is mock_chat.return_value

        # Verify the httpx client was properly closed
        mock_httpx_client.aclose.assert_awaited_once()

    @patch("nat.plugins.langchain.llm.create_httpx_client_with_dynamo_hooks")
    @patch("langchain_openai.ChatOpenAI")
    async def test_responses_api_branch(self, mock_chat, mock_create_client, dynamo_cfg_responses_api, mock_builder):
        """When APIType==RESPONSES, special flags should be added."""
        mock_httpx_client = MagicMock()
        mock_httpx_client.aclose = AsyncMock()  # Make aclose awaitable
        mock_create_client.return_value = mock_httpx_client

        async with dynamo_langchain(dynamo_cfg_responses_api, mock_builder):
            pass

        kwargs = mock_chat.call_args.kwargs
        assert kwargs["use_responses_api"] is True
        assert kwargs["use_previous_response_id"] is True
        assert kwargs["stream_usage"] is True

        # Verify the httpx client was properly closed
        mock_httpx_client.aclose.assert_awaited_once()

    @patch("nat.plugins.langchain.llm.create_httpx_client_with_dynamo_hooks")
    @patch("langchain_openai.ChatOpenAI")
    async def test_excludes_dynamo_specific_fields(self,
                                                   mock_chat,
                                                   mock_create_client,
                                                   dynamo_cfg_with_prefix,
                                                   mock_builder):
        """Dynamo-specific fields should be excluded from ChatOpenAI kwargs.

        DynamoModelConfig has fields (prefix_template, prefix_total_requests, prefix_osl,
        prefix_iat, request_timeout) that are only used internally by NAT to configure
        the custom httpx client for Dynamo header injection. These fields must NOT be
        passed to ChatOpenAI because:

        1. ChatOpenAI doesn't understand them and would error or ignore them
        2. They configure NAT's header injection behavior, not the LLM client itself

        This test ensures the `exclude` set in model_dump() properly filters these fields.
        If someone accidentally removes a field from the exclude set, this test will fail.
        """
        mock_httpx_client = MagicMock()
        mock_httpx_client.aclose = AsyncMock()  # Make aclose awaitable
        mock_create_client.return_value = mock_httpx_client

        async with dynamo_langchain(dynamo_cfg_with_prefix, mock_builder):
            pass

        kwargs = mock_chat.call_args.kwargs

        # These Dynamo-specific fields should NOT be passed to ChatOpenAI
        assert "prefix_template" not in kwargs
        assert "prefix_total_requests" not in kwargs
        assert "prefix_osl" not in kwargs
        assert "prefix_iat" not in kwargs
        assert "request_timeout" not in kwargs

        # Verify the httpx client was properly closed
        mock_httpx_client.aclose.assert_awaited_once()


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
