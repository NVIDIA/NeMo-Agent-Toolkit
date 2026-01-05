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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.llm import APITypeEnum
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.semantic_kernel.llm import openai_semantic_kernel

# ---------------------------------------------------------------------------
# OpenAI → Semantic-Kernel wrapper tests
# ---------------------------------------------------------------------------


class TestOpenAISemanticKernel:
    """Tests for the openai_semantic_kernel wrapper."""

    @pytest.fixture
    def mock_builder(self) -> Builder:
        return MagicMock(spec=Builder)

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
