# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nat.data_models.retry_mixin import RetryMixin
from nat.llm.openai_llm import OpenAIModelConfig


class TestRetryMixin:
    """Tests for `RetryMixin` field validation."""

    def test_defaults_are_valid(self):
        m = RetryMixin()
        assert m.do_auto_retry is True
        assert m.num_retries == 5

    def test_accepts_positive_retry_budget(self):
        m = RetryMixin(num_retries=1)
        assert m.num_retries == 1

    @pytest.mark.parametrize("budget", [0, -1])
    def test_rejects_non_positive_retry_budget(self, budget: int):
        with pytest.raises(ValidationError, match="num_retries"):
            RetryMixin(num_retries=budget)

    def test_disabling_retries_uses_do_auto_retry(self):
        m = RetryMixin(do_auto_retry=False)
        assert m.do_auto_retry is False
        assert m.num_retries == 5


class TestRetryMixinProviderConfig:
    """The constraint must hold for provider configs that inherit `RetryMixin` alongside other mixins."""

    @pytest.mark.parametrize("budget", [0, -1])
    def test_provider_config_rejects_non_positive_retry_budget(self, budget: int):
        with pytest.raises(ValidationError, match="num_retries"):
            OpenAIModelConfig(model_name="gpt-4o", num_retries=budget)

    def test_provider_config_accepts_positive_retry_budget(self):
        config = OpenAIModelConfig(model_name="gpt-4o", num_retries=1)
        assert config.num_retries == 1
