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
"""Tests for SSL verification feature in CrewAI LLM plugin."""

import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.builder.builder import Builder
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.crewai.llm import openai_crewai


class TestOpenAICrewAISSLVerification:
    """Tests for SSL verification in openai_crewai function."""

    @pytest.fixture
    def mock_builder(self):
        """Create a mock Builder object."""
        return MagicMock(spec=Builder)

    @pytest.fixture
    def openai_config(self):
        """Create an OpenAIModelConfig instance."""
        return OpenAIModelConfig(model_name="gpt-4", api_key="test-key")

    @patch("crewai.LLM")
    async def test_ssl_verification_enabled_by_default(self, mock_llm, openai_config, mock_builder):
        """Test that SSL verification is enabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            async with openai_crewai(openai_config, mock_builder):
                # Verify LLM was called
                mock_llm.assert_called_once()

                # When verify_ssl is True (default), LITELLM_SSL_VERIFY should not be set to "false"
                assert os.environ.get("LITELLM_SSL_VERIFY") != "false"

    @patch("crewai.LLM")
    async def test_ssl_verification_disabled(self, mock_llm, openai_config, mock_builder):
        """Test that SSL verification can be disabled via environment variable."""
        # Disable SSL verification
        openai_config.verify_ssl = False

        with patch.dict(os.environ, {}, clear=True):
            async with openai_crewai(openai_config, mock_builder):
                # Verify LLM was called
                mock_llm.assert_called_once()

                # When verify_ssl is False, LITELLM_SSL_VERIFY should be set to "false"
                assert os.environ.get("LITELLM_SSL_VERIFY") == "false"

    @patch("crewai.LLM")
    async def test_verify_ssl_field_excluded_from_config(self, mock_llm, openai_config, mock_builder):
        """Test that verify_ssl field is not passed to LLM."""
        openai_config.verify_ssl = False

        async with openai_crewai(openai_config, mock_builder):
            # Verify LLM was called
            mock_llm.assert_called_once()
            call_kwargs = mock_llm.call_args[1]

            # verify_ssl should not be in the kwargs passed to LLM
            assert "verify_ssl" not in call_kwargs

    @patch("crewai.LLM")
    async def test_all_config_params_passed_correctly(self, mock_llm, openai_config, mock_builder):
        """Test that all config parameters are passed correctly when SSL is disabled."""
        # Configure with multiple parameters
        openai_config.verify_ssl = False
        openai_config.base_url = "https://custom.endpoint.com/v1"
        openai_config.temperature = 0.7

        with patch.dict(os.environ, {}, clear=True):
            async with openai_crewai(openai_config, mock_builder):
                call_kwargs = mock_llm.call_args[1]

                # Check that standard params are passed
                assert call_kwargs["model"] == "gpt-4"
                assert call_kwargs["api_key"] == "test-key"
                assert call_kwargs["base_url"] == "https://custom.endpoint.com/v1"
                assert call_kwargs["temperature"] == 0.7

                # Check that SSL verification param is excluded and env var is set
                assert "verify_ssl" not in call_kwargs
                assert os.environ.get("LITELLM_SSL_VERIFY") == "false"
