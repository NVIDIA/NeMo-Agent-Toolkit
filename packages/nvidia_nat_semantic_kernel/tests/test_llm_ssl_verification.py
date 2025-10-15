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
"""Tests for SSL verification feature in Semantic Kernel LLM plugin."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from openai import AsyncOpenAI

from nat.builder.builder import Builder
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.semantic_kernel.llm import openai_semantic_kernel


class TestOpenAISemanticKernelSSLVerification:
    """Tests for SSL verification in openai_semantic_kernel function."""

    @pytest.fixture
    def mock_builder(self):
        """Create a mock Builder object."""
        return MagicMock(spec=Builder)

    @pytest.fixture
    def openai_config(self):
        """Create an OpenAIModelConfig instance."""
        return OpenAIModelConfig(model_name="gpt-4", api_key="test-key")

    @patch("semantic_kernel.connectors.ai.open_ai.OpenAIChatCompletion")
    async def test_ssl_verification_enabled_by_default(self, mock_openai_completion, openai_config, mock_builder):
        """Test that SSL verification is enabled by default."""
        async with openai_semantic_kernel(openai_config, mock_builder):
            # Verify OpenAIChatCompletion was called
            mock_openai_completion.assert_called_once()
            call_kwargs = mock_openai_completion.call_args[1]

            # When verify_ssl is True (default), no custom async_client should be passed
            assert "async_client" not in call_kwargs

    @patch("semantic_kernel.connectors.ai.open_ai.OpenAIChatCompletion")
    async def test_ssl_verification_disabled(self, mock_openai_completion, openai_config, mock_builder):
        """Test that SSL verification can be disabled."""
        # Disable SSL verification
        openai_config.verify_ssl = False

        async with openai_semantic_kernel(openai_config, mock_builder):
            # Verify OpenAIChatCompletion was called
            mock_openai_completion.assert_called_once()
            call_kwargs = mock_openai_completion.call_args[1]

            # When verify_ssl is False, custom async_client should be provided
            assert "async_client" in call_kwargs

            # Verify the client is an AsyncOpenAI client instance
            async_client = call_kwargs["async_client"]
            assert isinstance(async_client, AsyncOpenAI)

    @patch("semantic_kernel.connectors.ai.open_ai.OpenAIChatCompletion")
    async def test_api_key_and_base_url_passed(self, mock_openai_completion, openai_config, mock_builder):
        """Test that api_key and base_url are passed to OpenAIChatCompletion."""
        openai_config.base_url = "https://custom.endpoint.com/v1"

        async with openai_semantic_kernel(openai_config, mock_builder):
            # Verify OpenAIChatCompletion was called
            mock_openai_completion.assert_called_once()
            call_kwargs = mock_openai_completion.call_args[1]

            # Check that api_key and base_url are included
            assert call_kwargs["ai_model_id"] == "gpt-4"
            assert call_kwargs["api_key"] == "test-key"
            assert call_kwargs["base_url"] == "https://custom.endpoint.com/v1"

    @patch("semantic_kernel.connectors.ai.open_ai.OpenAIChatCompletion")
    async def test_all_config_params_passed_correctly(self, mock_openai_completion, openai_config, mock_builder):
        """Test that all config parameters are passed correctly when SSL is disabled."""
        # Configure with multiple parameters
        openai_config.verify_ssl = False
        openai_config.base_url = "https://custom.endpoint.com/v1"

        async with openai_semantic_kernel(openai_config, mock_builder):
            call_kwargs = mock_openai_completion.call_args[1]

            # Check that standard params are passed
            assert call_kwargs["ai_model_id"] == "gpt-4"
            assert call_kwargs["api_key"] == "test-key"
            assert call_kwargs["base_url"] == "https://custom.endpoint.com/v1"

            # Check that async_client is included
            assert "async_client" in call_kwargs
