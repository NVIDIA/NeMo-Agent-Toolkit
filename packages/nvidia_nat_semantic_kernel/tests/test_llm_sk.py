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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.llm import APITypeEnum
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.semantic_kernel.llm import azure_openai_semantic_kernel
from nat.plugins.semantic_kernel.llm import openai_semantic_kernel

# ---------------------------------------------------------------------------
# OpenAI → Semantic-Kernel wrapper tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("set_test_api_keys")
class TestOpenAISemanticKernel:
    """Tests for the openai_semantic_kernel wrapper."""

    @pytest.fixture
    def oa_cfg(self):
        return OpenAIModelConfig(model_name="gpt-4o")

    @pytest.fixture
    def oa_cfg_responses(self):
        # Using the RESPONSES API must be rejected by the wrapper.
        return OpenAIModelConfig(model_name="gpt-4o", api_type=APITypeEnum.RESPONSES)

    @patch("semantic_kernel.connectors.ai.open_ai.OpenAIChatCompletion")
    async def test_basic_creation(self, mock_sk, oa_cfg, mock_builder):
        """Ensure the wrapper instantiates OpenAIChatCompletion with the right model id."""
        async with openai_semantic_kernel(oa_cfg, mock_builder) as llm_obj:
            mock_sk.assert_called_once()
            assert mock_sk.call_args.kwargs["ai_model_id"] == "gpt-4o"
            assert llm_obj is mock_sk.return_value

    @patch("semantic_kernel.connectors.ai.open_ai.OpenAIChatCompletion")
    async def test_responses_api_blocked(self, mock_sk, oa_cfg_responses, mock_builder):
        """Selecting APIType.RESPONSES must raise a ValueError."""
        with pytest.raises(ValueError, match="Responses API is not supported"):
            async with openai_semantic_kernel(oa_cfg_responses, mock_builder):
                pass
        mock_sk.assert_not_called()

    @pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
    @patch("openai.AsyncOpenAI")
    @patch("semantic_kernel.connectors.ai.open_ai.OpenAIChatCompletion")
    async def test_verify_ssl_passed_to_client(self,
                                               mock_sk,
                                               mock_async_openai,
                                               oa_cfg,
                                               mock_builder,
                                               mock_httpx_async_client,
                                               verify_ssl):
        """Test that verify_ssl is passed to the underlying httpx.AsyncClient as verify."""
        mock_async_openai.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_async_openai.return_value.__aexit__ = AsyncMock(return_value=None)
        oa_cfg.verify_ssl = verify_ssl
        async with openai_semantic_kernel(oa_cfg, mock_builder):
            mock_httpx_async_client.assert_called_once()
            assert mock_httpx_async_client.call_args.kwargs["verify"] is verify_ssl


# ---------------------------------------------------------------------------
# Azure OpenAI → Semantic-Kernel wrapper tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("set_test_api_keys")
class TestAzureOpenAISemanticKernel:
    """Tests for the azure_openai_semantic_kernel wrapper."""

    @pytest.fixture
    def azure_cfg(self):
        return AzureOpenAIModelConfig(
            azure_deployment="gpt-4",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-01",
        )

    @pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
    @patch("openai.AsyncAzureOpenAI")
    @patch("semantic_kernel.connectors.ai.open_ai.AzureChatCompletion")
    async def test_verify_ssl_passed_to_client(self,
                                               mock_azure_chat,
                                               mock_async_azure_openai,
                                               azure_cfg,
                                               mock_builder,
                                               mock_httpx_async_client,
                                               verify_ssl):
        """Test that verify_ssl is passed to the underlying httpx.AsyncClient as verify."""
        mock_async_azure_openai.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_async_azure_openai.return_value.__aexit__ = AsyncMock(return_value=None)
        azure_cfg.verify_ssl = verify_ssl
        async with azure_openai_semantic_kernel(azure_cfg, mock_builder):
            mock_httpx_async_client.assert_called_once()
            assert mock_httpx_async_client.call_args.kwargs["verify"] is verify_ssl


# ---------------------------------------------------------------------------
# Registration decorator sanity check
# ---------------------------------------------------------------------------


@patch("nat.cli.type_registry.GlobalTypeRegistry")
def test_decorator_registration(mock_global_registry):
    """Verify that register_llm_client decorated the Semantic-Kernel wrapper."""
    registry = MagicMock()
    mock_global_registry.get.return_value = registry

    # Pretend decorator execution populated the map.
    registry._llm_client_map = {
        (OpenAIModelConfig, LLMFrameworkEnum.SEMANTIC_KERNEL): openai_semantic_kernel,
    }

    assert (registry._llm_client_map[(OpenAIModelConfig, LLMFrameworkEnum.SEMANTIC_KERNEL)] is openai_semantic_kernel)
