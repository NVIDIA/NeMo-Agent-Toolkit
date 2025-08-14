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
from nat.builder.embedder import EmbedderProviderInfo
from nat.cli.register_workflow import register_embedder_provider
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.retry_mixin import RetryMixin


class AWSBedrockEmbedderModelConfig(EmbedderBaseConfig, RetryMixin, name="aws_bedrock"):
    """An AWS Bedrock embedder provider to be used with an embedder client."""

    model_config = ConfigDict(protected_namespaces=())

    # Completion parameters
    model_name: str = Field(validation_alias=AliasChoices("model_name", "model"),
                            serialization_alias="model",
                            description="The model name for the hosted AWS Bedrock.")

    # Client parameters
    region_name: str | None = Field(default="None", description="AWS region to use.")
    base_url: str | None = Field(
        default=None, description="Bedrock endpoint to use. Needed if you don't want to default to us-east-1 endpoint.")
    credentials_profile_name: str | None = Field(
        default=None, description="The name of the profile in the ~/.aws/credentials or ~/.aws/config files.")


@register_embedder_provider(config_type=AWSBedrockEmbedderModelConfig)
async def aws_bedrock_embedder(embedder_config: AWSBedrockEmbedderModelConfig, _builder: Builder):

    yield EmbedderProviderInfo(config=embedder_config,
                               description="An AWS Bedrock embedder for use with an embedder client.")
