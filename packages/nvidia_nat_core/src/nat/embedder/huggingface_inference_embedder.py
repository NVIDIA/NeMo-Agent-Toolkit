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
from nat.builder.embedder import EmbedderProviderInfo
from nat.cli.register_workflow import register_embedder_provider
from nat.data_models.common import OptionalSecretStr
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.retry_mixin import RetryMixin


class HuggingFaceInferenceEmbedderConfig(EmbedderBaseConfig, RetryMixin, name="huggingface_inference"):
    """HuggingFace remote embedder provider for TEI servers and Inference Endpoints.

    Connects to remote embedding services via the HuggingFace Inference API
    or custom Text Embeddings Inference (TEI) servers.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    endpoint_url: str = Field(description="Endpoint URL for TEI server or HuggingFace Inference Endpoint")
    api_key: OptionalSecretStr = Field(default=None, description="HuggingFace API token for authentication")
    timeout: float = Field(default=120.0, ge=1.0, description="Request timeout in seconds")


@register_embedder_provider(config_type=HuggingFaceInferenceEmbedderConfig)
async def huggingface_inference_embedder_provider(config: HuggingFaceInferenceEmbedderConfig, _builder: Builder):
    """Register HuggingFace remote embedder as a provider."""

    description = f"HuggingFace Remote Embedder: {config.endpoint_url}"

    yield EmbedderProviderInfo(config=config, description=description)
