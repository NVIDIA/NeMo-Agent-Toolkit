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
# pylint: disable=unused-argument, not-async-context-manager

import logging
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.llm import APITypeEnum
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.langchain.llm import aws_bedrock_langchain
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
        with pytest.raises(ValueError, match="only supports chat completion"):
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
        with pytest.raises(ValueError, match="only supports chat completion"):
            async with aws_bedrock_langchain(bedrock_cfg_wrong_api, mock_builder):
                pass
        mock_chat.assert_not_called()


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
    }

    assert registry._llm_client_map[(NIMModelConfig, LLMFrameworkEnum.LANGCHAIN)] is nim_langchain
    assert registry._llm_client_map[(OpenAIModelConfig, LLMFrameworkEnum.LANGCHAIN)] is openai_langchain
    assert registry._llm_client_map[(AWSBedrockModelConfig, LLMFrameworkEnum.LANGCHAIN)] is aws_bedrock_langchain
