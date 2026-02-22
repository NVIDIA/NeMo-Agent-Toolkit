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

import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.llm import APITypeEnum
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.crewai.llm import nim_crewai
from nat.plugins.crewai.llm import openai_crewai

# ---------------------------------------------------------------------------
# NIM → CrewAI wrapper tests
# ---------------------------------------------------------------------------


class TestNimCrewAI:
    """Tests for the nim_crewai wrapper."""

    @pytest.fixture
    def mock_builder(self) -> Builder:
        return MagicMock(spec=Builder)

    @pytest.fixture
    def nim_cfg(self):
        return NIMModelConfig(model_name="test-nim")

    @pytest.fixture
    def nim_cfg_responses(self):
        return NIMModelConfig(model_name="test-nim", api_type=APITypeEnum.RESPONSES)

    @patch("crewai.LLM")
    async def test_basic_creation(self, mock_llm, nim_cfg, mock_builder):
        """Wrapper should yield a crewai.LLM configured for the NIM model."""
        async with nim_crewai(nim_cfg, mock_builder) as llm_obj:
            mock_llm.assert_called_once()
            kwargs = mock_llm.call_args.kwargs
            assert kwargs["model"] == "nvidia_nim/test-nim"
            assert llm_obj is mock_llm.return_value

    @patch("crewai.LLM")
    async def test_responses_api_blocked(self, mock_llm, nim_cfg_responses, mock_builder):
        """Selecting the Responses API must raise a ValueError."""
        with pytest.raises(ValueError, match="Responses API is not supported"):
            async with nim_crewai(nim_cfg_responses, mock_builder):
                pass
        mock_llm.assert_not_called()

    @patch("crewai.LLM")
    @patch.dict(os.environ, {"NVIDIA_API_KEY": "legacy-key"}, clear=True)
    async def test_env_key_transfer(self, mock_llm, nim_cfg, mock_builder):
        """
        If NVIDIA_NIM_API_KEY is not set but NVIDIA_API_KEY is,
        the wrapper should copy it for LiteLLM compatibility.
        """
        assert "NVIDIA_NIM_API_KEY" not in os.environ
        async with nim_crewai(nim_cfg, mock_builder):
            pass
        assert os.environ["NVIDIA_NIM_API_KEY"] == "legacy-key"
        mock_llm.assert_called_once()


# ---------------------------------------------------------------------------
# OpenAI → CrewAI wrapper tests
# ---------------------------------------------------------------------------


class TestOpenAICrewAI:
    """Tests for the openai_crewai wrapper."""

    @pytest.fixture
    def mock_builder(self) -> Builder:
        return MagicMock(spec=Builder)

    @pytest.fixture
    def openai_cfg(self):
        return OpenAIModelConfig(model_name="gpt-4o")

    @pytest.fixture
    def openai_cfg_responses(self):
        return OpenAIModelConfig(model_name="gpt-4o", api_type=APITypeEnum.RESPONSES)

    @patch("crewai.LLM")
    async def test_basic_creation(self, mock_llm, openai_cfg, mock_builder):
        """Wrapper should yield a crewai.LLM for OpenAI models."""
        async with openai_crewai(openai_cfg, mock_builder) as llm_obj:
            mock_llm.assert_called_once()
            assert mock_llm.call_args.kwargs["model"] == "gpt-4o"
            assert llm_obj is mock_llm.return_value

    @patch("crewai.LLM")
    async def test_param_passthrough(self, mock_llm, openai_cfg, mock_builder):
        """Arbitrary config kwargs must reach crewai.LLM unchanged."""
        openai_cfg.temperature = 0.3
        openai_cfg.api_key = SecretStr("sk-abc123")
        async with openai_crewai(openai_cfg, mock_builder):
            pass
        kwargs = mock_llm.call_args.kwargs
        assert kwargs["temperature"] == 0.3
        assert kwargs["api_key"] == "sk-abc123"

    @patch("crewai.LLM")
    async def test_responses_api_blocked(self, mock_llm, openai_cfg_responses, mock_builder):
        with pytest.raises(ValueError, match="Responses API is not supported"):
            async with openai_crewai(openai_cfg_responses, mock_builder):
                pass
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Registration decorator sanity check
# ---------------------------------------------------------------------------


@patch("nat.cli.type_registry.GlobalTypeRegistry")
def test_decorator_registration(mock_global_registry):
    """Verify that register_llm_client decorators registered the CrewAI wrappers."""
    registry = MagicMock()
    mock_global_registry.get.return_value = registry

    # Pretend the decorators already executed.
    registry._llm_client_map = {
        (NIMModelConfig, LLMFrameworkEnum.CREWAI): nim_crewai,
        (OpenAIModelConfig, LLMFrameworkEnum.CREWAI): openai_crewai,
    }

    assert registry._llm_client_map[(NIMModelConfig, LLMFrameworkEnum.CREWAI)] is nim_crewai
    assert registry._llm_client_map[(OpenAIModelConfig, LLMFrameworkEnum.CREWAI)] is openai_crewai
