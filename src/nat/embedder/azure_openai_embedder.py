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

from pydantic import AliasChoices
from pydantic import ConfigDict
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.embedder import EmbedderProviderInfo
from nat.cli.register_workflow import register_embedder_provider
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.retry_mixin import RetryMixin


class AzureOpenAIEmbedderModelConfig(EmbedderBaseConfig, RetryMixin, name="azure_openai"):
    """An Azure OpenAI embedder provider to be used with an embedder client."""

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    api_key: str | None = Field(default=None, description="Azure OpenAI API key to interact with hosted model.")
    api_version: str = Field(default="2025-04-01-preview", description="Azure OpenAI API version.")
    azure_endpoint: str | None = Field(validation_alias=AliasChoices("azure_endpoint", "base_url"),
                                       serialization_alias="azure_endpoint",
                                       default=None,
                                       description="Base URL for the hosted model.")
    azure_deployment: str = Field(validation_alias=AliasChoices("azure_deployment", "model_name", "model"),
                                  serialization_alias="azure_deployment",
                                  description="The Azure OpenAI hosted model/deployment name.")


@register_embedder_provider(config_type=AzureOpenAIEmbedderModelConfig)
async def azure_openai_embedder_model(config: AzureOpenAIEmbedderModelConfig, _builder: Builder):

    yield EmbedderProviderInfo(config=config, description="An Azure OpenAI model for use with an Embedder client.")
