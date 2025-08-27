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

import pytest
from pydantic import ValidationError

from nat.data_models.thinking_mixin import ThinkingMixin


class TestThinkingMixin:
    """Tests for ThinkingMixin behavior and thinking_system_prompt generation."""

    def test_supported_nvidia_thinking_prompts(self):

        class Model(ThinkingMixin):
            model_name: str

        m_true = Model(model_name="nvidia/nvidia-nemotron-8b", thinking=True)
        assert m_true.thinking_system_prompt == "/think"

        m_false = Model(model_name="nvidia/nvidia-nemotron-8b", thinking=False)
        assert m_false.thinking_system_prompt == "/no_think"

    def test_supported_llama_thinking_prompts_case_insensitive(self):

        class Model(ThinkingMixin):
            model_name: str

        m_true = Model(model_name="NVIDIA/LLaMa-3.1-Nemotron", thinking=True)
        assert m_true.thinking_system_prompt == "detailed thinking on"

        m_false = Model(model_name="nvidia/llama-nemotron", thinking=False)
        assert m_false.thinking_system_prompt == "detailed thinking off"

    def test_supported_default_remains_none(self):

        class Model(ThinkingMixin):
            model_name: str

        m = Model(model_name="nvidia/llama-nemotron")
        assert m.thinking is None
        assert m.thinking_system_prompt is None

    def test_unsupported_model_allows_none(self):

        class Model(ThinkingMixin):
            model_name: str

        m = Model(model_name="gpt-4o")
        assert m.thinking is None
        assert m.thinking_system_prompt is None

    def test_unsupported_model_rejects_non_none_value(self):

        class Model(ThinkingMixin):
            model_name: str

        with pytest.raises(ValidationError, match=r"thinking is not supported for model_name: gpt-4o"):
            _ = Model(model_name="gpt-4o", thinking=True)

    def test_support_detected_on_model_key_when_model_name_missing(self):

        class Model(ThinkingMixin):
            model: str

        m = Model(model="nvidia/nvidia-some-nemotron", thinking=False)
        assert m.thinking_system_prompt == "/no_think"

    def test_support_detected_on_azure_deployment_when_others_missing(self):

        class Model(ThinkingMixin):
            azure_deployment: str

        m = Model(azure_deployment="nvidia/llama3-nemotron", thinking=True)
        assert m.thinking_system_prompt == "detailed thinking on"

    def test_no_keys_present_defaults_supported_and_prompt_none(self):
        m = ThinkingMixin(thinking=True)
        assert m.thinking is True
        assert m.thinking_system_prompt is None
