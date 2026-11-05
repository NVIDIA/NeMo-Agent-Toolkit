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

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.llm import APITypeEnum
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.litellm_llm import LiteLlmModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.crewai.llm import azure_openai_crewai
from nat.plugins.crewai.llm import litellm_crewai
from nat.plugins.crewai.llm import nim_crewai
from nat.plugins.crewai.llm import openai_crewai

# ---------------------------------------------------------------------------
# NIM → CrewAI wrapper tests
# ---------------------------------------------------------------------------


class TestNimCrewAI:
    """Tests for the nim_crewai wrapper."""

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

    @patch("nat.plugins.crewai.llm._handle_litellm_verify_ssl")
    @patch("crewai.LLM")
    @pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
    async def test_nim_verify_ssl_passed_to_handle(self, mock_llm, mock_handle_verify_ssl, verify_ssl, mock_builder):
        """verify_ssl from config is passed to _handle_litellm_verify_ssl."""
        nim_cfg = NIMModelConfig(model_name="test-nim", verify_ssl=verify_ssl)
        async with nim_crewai(nim_cfg, mock_builder):
            mock_handle_verify_ssl.assert_called_once_with(nim_cfg)


# ---------------------------------------------------------------------------
# OpenAI → CrewAI wrapper tests
# ---------------------------------------------------------------------------


class TestOpenAICrewAI:
    """Tests for the openai_crewai wrapper."""

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

    @patch("nat.plugins.crewai.llm._handle_litellm_verify_ssl")
    @patch("crewai.LLM")
    @pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
    async def test_openai_verify_ssl_passed_to_handle(self, mock_llm, mock_handle_verify_ssl, verify_ssl, mock_builder):
        """verify_ssl from config is passed to _handle_litellm_verify_ssl."""
        openai_cfg = OpenAIModelConfig(model_name="gpt-4o", verify_ssl=verify_ssl)
        async with openai_crewai(openai_cfg, mock_builder):
            mock_handle_verify_ssl.assert_called_once_with(openai_cfg)


# ---------------------------------------------------------------------------
# verify_ssl must not reach crewai.LLM
# ---------------------------------------------------------------------------

_AZURE_KWARGS = {"azure_deployment": "gpt-4o", "azure_endpoint": "https://test.openai.azure.com", "api_key": "test-key"}

_VERIFY_SSL_CASES = [
    pytest.param(nim_crewai, NIMModelConfig, {"model_name": "test-nim"}, id="nim"),
    pytest.param(openai_crewai, OpenAIModelConfig, {"model_name": "gpt-4o"}, id="openai"),
    pytest.param(litellm_crewai, LiteLlmModelConfig, {"model_name": "openai/gpt-4o"}, id="litellm"),
    pytest.param(azure_openai_crewai, AzureOpenAIModelConfig, _AZURE_KWARGS, id="azure_openai"),
]


@patch.dict(os.environ)
@patch("nat.plugins.crewai.llm._handle_litellm_verify_ssl", new=MagicMock())
@patch("crewai.LLM")
@pytest.mark.parametrize("wrapper, config_cls, config_kwargs", _VERIFY_SSL_CASES)
@pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
async def test_verify_ssl_not_forwarded_to_llm(mock_llm, wrapper, config_cls, config_kwargs, verify_ssl, mock_builder):
    """verify_ssl is applied through _handle_litellm_verify_ssl.

    crewai.LLM sends unknown kwargs with every request, so the native OpenAI client raises a TypeError from
    create() and the LiteLLM path puts verify_ssl into the request body.
    """
    cfg = config_cls(**config_kwargs, verify_ssl=verify_ssl)
    async with wrapper(cfg, mock_builder):
        pass
    assert "verify_ssl" not in mock_llm.call_args.kwargs


@patch.dict(os.environ)
@patch("nat.plugins.crewai.llm._handle_litellm_verify_ssl", new=MagicMock())
@patch("crewai.LLM")
@pytest.mark.parametrize("wrapper, config_cls, config_kwargs", _VERIFY_SSL_CASES)
async def test_verify_ssl_not_forwarded_after_round_trip(mock_llm, wrapper, config_cls, config_kwargs, mock_builder):
    """`nat serve` reloads a round-trip dump of the config, which marks verify_ssl as set."""
    cfg = config_cls(**config_kwargs)
    cfg = config_cls.model_validate(cfg.model_dump(mode="json", by_alias=True, round_trip=True))
    assert "verify_ssl" in cfg.model_fields_set

    async with wrapper(cfg, mock_builder):
        pass
    assert "verify_ssl" not in mock_llm.call_args.kwargs


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
