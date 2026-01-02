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

from pydantic import AliasChoices
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PositiveInt

from nat.builder.builder import Builder
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_provider
from nat.data_models.common import OptionalSecretStr
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.optimizable import OptimizableField
from nat.data_models.optimizable import OptimizableMixin
from nat.data_models.optimizable import SearchSpace
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin


class NIMModelConfig(LLMBaseConfig, RetryMixin, OptimizableMixin, ThinkingMixin, name="nim"):
    """An NVIDIA Inference Microservice (NIM) llm provider to be used with an LLM client."""

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    api_key: OptionalSecretStr = Field(default=None, description="NVIDIA API key to interact with hosted NIM.")
    base_url: str | None = Field(default=None, description="Base url to the hosted NIM.")
    model_name: str = OptimizableField(validation_alias=AliasChoices("model_name", "model"),
                                       serialization_alias="model",
                                       description="The model name for the hosted NIM.")
    max_tokens: PositiveInt = OptimizableField(default=300,
                                               description="Maximum number of tokens to generate.",
                                               space=SearchSpace(high=2176, low=128, step=512))
    temperature: float | None = OptimizableField(
        default=None,
        ge=0.0,
        description="Sampling temperature to control randomness in the output.",
        space=SearchSpace(high=0.9, low=0.1, step=0.2))
    top_p: float | None = OptimizableField(default=None,
                                           ge=0.0,
                                           le=1.0,
                                           description="Top-p for distribution sampling.",
                                           space=SearchSpace(high=1.0, low=0.5, step=0.1))


@register_llm_provider(config_type=NIMModelConfig)
async def nim_model(llm_config: NIMModelConfig, _builder: Builder):

    yield LLMProviderInfo(config=llm_config, description="A NIM model for use with an LLM client.")
