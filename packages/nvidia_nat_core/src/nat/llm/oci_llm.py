# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pydantic import model_validator

from nat.builder.builder import Builder
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_provider
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.optimizable import OptimizableField
from nat.data_models.optimizable import OptimizableMixin
from nat.data_models.optimizable import SearchSpace
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin


class OCIModelConfig(LLMBaseConfig, RetryMixin, OptimizableMixin, ThinkingMixin, name="oci"):
    """OCI Generative AI LLM provider."""

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    region: str = Field(
        default="us-chicago-1",
        description="OCI region for the Generative AI service. Used to build the endpoint when endpoint is not set.",
    )
    endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("endpoint", "service_endpoint", "base_url"),
        description="OCI Generative AI service endpoint URL. Auto-derived from region when omitted.",
    )
    compartment_id: str | None = Field(default=None, description="OCI compartment OCID for Generative AI requests.")

    @model_validator(mode="after")
    def _derive_endpoint_from_region(self) -> "OCIModelConfig":
        if self.endpoint is None:
            self.endpoint = f"https://inference.generativeai.{self.region}.oci.oraclecloud.com"
        return self

    auth_type: str = Field(default="API_KEY",
                           description="OCI SDK authentication type: API_KEY, SECURITY_TOKEN, INSTANCE_PRINCIPAL, "
                           "or RESOURCE_PRINCIPAL.")
    auth_profile: str = Field(default="DEFAULT",
                              description="OCI config profile to use for API_KEY or SECURITY_TOKEN auth.")
    auth_file_location: str = Field(default="~/.oci/config",
                                    description="Path to the OCI config file used for SDK authentication.")
    model_name: str = OptimizableField(validation_alias=AliasChoices("model_name", "model"),
                                       serialization_alias="model",
                                       description="The OCI Generative AI model ID.")
    provider: str | None = Field(default=None,
                                 description="Optional OCI provider override such as cohere, google, meta, or openai.")
    context_size: int | None = Field(
        default=1024,
        gt=0,
        description="The maximum number of tokens available for input.",
    )
    seed: int | None = Field(default=None, description="Random seed to set for generation.")
    max_retries: int = Field(default=10, description="The max number of retries for the request.")
    max_tokens: int | None = Field(default=None, gt=0, description="Maximum number of output tokens.")
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
    request_timeout: float | None = Field(default=None, gt=0.0, description="HTTP request timeout in seconds.")


@register_llm_provider(config_type=OCIModelConfig)
async def oci_llm(config: OCIModelConfig, _builder: Builder) -> AsyncIterator[LLMProviderInfo]:
    """Yield provider metadata for an OCI Generative AI model.

    Args:
        config: OCI model configuration.
        _builder: Builder instance.

    Yields:
        LLMProviderInfo describing the configured OCI model.
    """

    yield LLMProviderInfo(config=config, description="An OCI Generative AI model for use with an LLM client.")
