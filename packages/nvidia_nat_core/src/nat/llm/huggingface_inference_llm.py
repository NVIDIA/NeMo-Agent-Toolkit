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

from pydantic import ConfigDict
from pydantic import Field

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


class HuggingFaceInferenceLLMConfig(LLMBaseConfig,
                                    RetryMixin,
                                    OptimizableMixin,
                                    ThinkingMixin,
                                    name="huggingface_inference"):
    """HuggingFace Inference API LLM provider for remote model inference.

    Supports:
    - Serverless Inference API (default)
    - Dedicated Inference Endpoints (via endpoint_url)
    - Self-hosted TGI servers (via endpoint_url)
    """

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    model_name: str = Field(description="HuggingFace model identifier (e.g., 'meta-llama/Llama-3.2-8B-Instruct')")
    api_key: OptionalSecretStr = Field(
        default=None,
        description=
        "HuggingFace API token for authentication. Required for Serverless API and private Inference Endpoints.")
    endpoint_url: str | None = Field(
        default=None,
        description=
        "Custom endpoint URL for Inference Endpoints or self-hosted TGI servers. If not provided, uses Serverless API.")
    max_new_tokens: int | None = OptimizableField(default=512,
                                                  ge=1,
                                                  description="Maximum number of new tokens to generate.",
                                                  space=SearchSpace(high=2048, low=128, step=128))
    temperature: float | None = OptimizableField(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature to control randomness in the output.",
        space=SearchSpace(high=1.0, low=0.1, step=0.1))
    top_p: float | None = OptimizableField(default=None,
                                           ge=0.0,
                                           le=1.0,
                                           description="Top-p (nucleus) sampling parameter.",
                                           space=SearchSpace(high=1.0, low=0.5, step=0.1))
    top_k: int | None = Field(default=None, ge=1, description="Top-k sampling parameter.")
    repetition_penalty: float | None = Field(default=None, ge=0.0, description="Penalty for repeating tokens.")
    seed: int | None = Field(default=None, description="Random seed for reproducible generation.")
    timeout: float = Field(default=120.0, ge=1.0, description="Request timeout in seconds.")


@register_llm_provider(config_type=HuggingFaceInferenceLLMConfig)
async def huggingface_inference_provider(config: HuggingFaceInferenceLLMConfig, _builder: Builder):
    """Register HuggingFace Inference API as an LLM provider."""

    endpoint_type = "Serverless API" if config.endpoint_url is None else "Custom Endpoint"
    description = f"HuggingFace {endpoint_type}: {config.model_name}"

    yield LLMProviderInfo(config=config, description=description)
