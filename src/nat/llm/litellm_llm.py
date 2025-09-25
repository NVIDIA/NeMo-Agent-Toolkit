# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections.abc import AsyncIterator

from pydantic import AliasChoices
from pydantic import ConfigDict
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_provider
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.temperature_mixin import TemperatureMixin
from nat.data_models.thinking_mixin import ThinkingMixin
from nat.data_models.top_p_mixin import TopPMixin


class LiteLlmModelConfig(
        LLMBaseConfig,
        RetryMixin,
        TemperatureMixin,
        TopPMixin,
        ThinkingMixin,
        name="litellm",
):
    """A LiteLlm provider to be used with an LLM client."""

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    api_key: str | None = Field(default=None, description="API key to interact with hosted model.")
    base_url: str | None = Field(default=None,
                                 description="Base url to the hosted model.",
                                 validation_alias=AliasChoices("base_url", "api_base"),
                                 serialization_alias="api_base")
    model_name: str = Field(validation_alias=AliasChoices("model_name", "model"),
                            serialization_alias="model",
                            description="The LiteLlm hosted model name.")
    seed: int | None = Field(default=None, description="Random seed to set for generation.")


@register_llm_provider(config_type=LiteLlmModelConfig)
async def litellm_model(
    config: LiteLlmModelConfig,
    _builder: Builder,
) -> AsyncIterator[LLMProviderInfo]:
    """Litellm model provider.

    Args:
        config (LiteLlmModelConfig): The LiteLlm model configuration.
        _builder (Builder): The NAT builder instance.

    Returns:
        AsyncIterator[LLMProviderInfo]: An async iterator that yields an LLMProviderInfo object.
    """
    yield LLMProviderInfo(config=config, description="A LiteLlm model for use with an LLM client.")
