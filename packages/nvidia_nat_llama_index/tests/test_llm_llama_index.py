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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.llm import APITypeEnum
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.llama_index.llm import aws_bedrock_llama_index
from nat.plugins.llama_index.llm import nim_llama_index
from nat.plugins.llama_index.llm import openai_llama_index

# ---------------------------------------------------------------------------
# NIM → Llama-Index wrapper tests
# ---------------------------------------------------------------------------


class TestNimLlamaIndex:
    """Tests for nim_llama_index."""

    @pytest.fixture
    def mock_builder(self):
        return MagicMock(spec=Builder)

    @pytest.fixture
    def nim_cfg(self):
        return NIMModelConfig(model_name="nemotron-3b")

    @pytest.fixture
    def nim_cfg_bad_api(self):
        return NIMModelConfig(model_name="nemotron-3b", api_type=APITypeEnum.RESPONSES)

    @patch("llama_index.llms.nvidia.NVIDIA")
    async def test_basic_creation(self, mock_nv, nim_cfg, mock_builder):
        """Wrapper should instantiate llama_index.llms.nvidia.NVIDIA."""
        async with nim_llama_index(nim_cfg, mock_builder) as llm:
            mock_nv.assert_called_once()
            kwargs = mock_nv.call_args.kwargs
            assert kwargs["model"] == "nemotron-3b"
            assert llm is mock_nv.return_value

    @patch("llama_index.llms.nvidia.NVIDIA")
    async def test_api_type_validation(self, mock_nv, nim_cfg_bad_api, mock_builder):
        """Non-chat API types must raise."""
        with pytest.raises(ValueError):
            async with nim_llama_index(nim_cfg_bad_api, mock_builder):
                pass
        mock_nv.assert_not_called()


# ---------------------------------------------------------------------------
# OpenAI → Llama-Index wrapper tests
# ---------------------------------------------------------------------------


class TestOpenAILlamaIndex:
    """Tests for openai_llama_index."""

    @pytest.fixture
    def mock_builder(self):
        return MagicMock(spec=Builder)

    @pytest.fixture
    def oa_cfg_chat(self):
        return OpenAIModelConfig(model_name="gpt-4o", base_url=None)

    @pytest.fixture
    def oa_cfg_responses(self):
        return OpenAIModelConfig(model_name="gpt-4o", api_type=APITypeEnum.RESPONSES, temperature=0.1)

    @patch("llama_index.llms.openai.OpenAI")
    async def test_chat_completion_branch(self, mock_openai, oa_cfg_chat, mock_builder):
        """CHAT_COMPLETION should create an OpenAI client, omitting base_url when None."""
        async with openai_llama_index(oa_cfg_chat, mock_builder) as llm:
            mock_openai.assert_called_once()
            kwargs = mock_openai.call_args.kwargs
            assert kwargs["model"] == "gpt-4o"
            assert "base_url" not in kwargs
            assert llm is mock_openai.return_value

    @patch("llama_index.llms.openai.OpenAIResponses")
    async def test_responses_branch(self, mock_resp, oa_cfg_responses, mock_builder):
        """RESPONSES API type should instantiate OpenAIResponses."""
        async with openai_llama_index(oa_cfg_responses, mock_builder) as llm:
            mock_resp.assert_called_once()
            kwargs = mock_resp.call_args.kwargs
            assert kwargs["model"] == "gpt-4o"
            assert kwargs["temperature"] == 0.1
            assert llm is mock_resp.return_value


# ---------------------------------------------------------------------------
# AWS Bedrock → Llama-Index wrapper tests
# ---------------------------------------------------------------------------


class TestBedrockLlamaIndex:
    """Tests for aws_bedrock_llama_index."""

    @pytest.fixture
    def mock_builder(self):
        return MagicMock(spec=Builder)

    @pytest.fixture
    def br_cfg(self):
        return AWSBedrockModelConfig(model_name="ai21.j2-ultra")

    @pytest.fixture
    def br_cfg_bad_api(self):
        return AWSBedrockModelConfig(model_name="ai21.j2-ultra", api_type=APITypeEnum.RESPONSES)

    @patch("llama_index.llms.bedrock.Bedrock")
    async def test_basic_creation(self, mock_bedrock, br_cfg, mock_builder):
        async with aws_bedrock_llama_index(br_cfg, mock_builder) as llm:
            mock_bedrock.assert_called_once()
            assert mock_bedrock.call_args.kwargs["model"] == "ai21.j2-ultra"
            assert llm is mock_bedrock.return_value

    @patch("llama_index.llms.bedrock.Bedrock")
    async def test_api_type_validation(self, mock_bedrock, br_cfg_bad_api, mock_builder):
        with pytest.raises(ValueError):
            async with aws_bedrock_llama_index(br_cfg_bad_api, mock_builder):
                pass
        mock_bedrock.assert_not_called()


# ---------------------------------------------------------------------------
# Registration decorator sanity check
# ---------------------------------------------------------------------------


@patch("nat.cli.type_registry.GlobalTypeRegistry")
def test_decorator_registration(mock_global_registry):
    """Ensure register_llm_client decorators registered the Llama-Index wrappers."""
    registry = MagicMock()
    mock_global_registry.get.return_value = registry

    registry._llm_client_map = {
        (NIMModelConfig, LLMFrameworkEnum.LLAMA_INDEX): nim_llama_index,
        (OpenAIModelConfig, LLMFrameworkEnum.LLAMA_INDEX): openai_llama_index,
        (AWSBedrockModelConfig, LLMFrameworkEnum.LLAMA_INDEX): aws_bedrock_llama_index,
    }

    assert registry._llm_client_map[(NIMModelConfig, LLMFrameworkEnum.LLAMA_INDEX)] is nim_llama_index
    assert registry._llm_client_map[(OpenAIModelConfig, LLMFrameworkEnum.LLAMA_INDEX)] is openai_llama_index
    assert registry._llm_client_map[(AWSBedrockModelConfig, LLMFrameworkEnum.LLAMA_INDEX)] is aws_bedrock_llama_index
