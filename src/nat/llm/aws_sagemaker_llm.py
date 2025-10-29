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

from nat.builder.builder import Builder
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_provider
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.optimizable import OptimizableField
from nat.data_models.optimizable import OptimizableMixin
from nat.data_models.optimizable import SearchSpace
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.temperature_mixin import TemperatureMixin
from nat.data_models.thinking_mixin import ThinkingMixin
from nat.data_models.top_p_mixin import TopPMixin


class AWSSageMakerModelConfig(LLMBaseConfig,
                              RetryMixin,
                              OptimizableMixin,
                              TemperatureMixin,
                              TopPMixin,
                              ThinkingMixin,
                              name="aws_sagemaker"):
    """An AWS SageMaker llm provider to be used with an LLM client."""

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    # Completion parameters
    endpoint_name: str = Field(
        validation_alias=AliasChoices("endpoint_name", "model_name", "model"),
        serialization_alias="endpoint_name",
        description="The SageMaker endpoint name for the hosted model."
    )
    max_tokens: int = OptimizableField(default=300,
                                       description="Maximum number of tokens to generate.",
                                       space=SearchSpace(high=2176, low=128, step=512))
    context_size: int | None = Field(
        default=1024,
        gt=0,
        description="The maximum number of tokens available for input. This is only required for LlamaIndex. "
        "This field is ignored for LangChain/LangGraph.",
    )

    # Client parameters
    region_name: str | None = Field(default=None, description="AWS region where the SageMaker endpoint is deployed.")
    credentials_profile_name: str | None = Field(
        default=None, description="The name of the profile in the ~/.aws/credentials or ~/.aws/config files.")

    # SageMaker-specific parameters
    content_handler: str | None = Field(
        default=None,
        description="Content handler class for formatting requests/responses. If not provided, a default handler will be used."
    )
    model_kwargs: dict | None = Field(
        default=None,
        description="Additional model-specific parameters to pass to the SageMaker endpoint."
    )


@register_llm_provider(config_type=AWSSageMakerModelConfig)
async def aws_sagemaker_model(llm_config: AWSSageMakerModelConfig, _builder: Builder):

    yield LLMProviderInfo(config=llm_config, description="An AWS SageMaker model for use with an LLM client.")
