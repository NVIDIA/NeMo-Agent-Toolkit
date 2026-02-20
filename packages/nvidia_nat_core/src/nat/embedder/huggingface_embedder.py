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
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.retry_mixin import RetryMixin


class HuggingFaceEmbedderConfig(EmbedderBaseConfig, RetryMixin, name="huggingface"):
    """HuggingFace local embedder provider using sentence-transformers.

    Loads models locally via the sentence-transformers library for embedding generation.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    model_name: str = Field(
        description=
        "HuggingFace model identifier (e.g., 'BAAI/bge-large-en-v1.5', 'sentence-transformers/all-MiniLM-L6-v2')")
    device: str = Field(default="auto", description="Device to use for local models ('cpu', 'cuda', 'mps', or 'auto')")
    normalize_embeddings: bool = Field(default=True, description="Whether to normalize embeddings to unit length")
    batch_size: int = Field(default=32, ge=1, description="Batch size for embedding generation")
    max_seq_length: int | None = Field(default=None, ge=1, description="Maximum sequence length for input text")
    trust_remote_code: bool = Field(default=False, description="Whether to trust remote code when loading models")


@register_embedder_provider(config_type=HuggingFaceEmbedderConfig)
async def huggingface_embedder_provider(config: HuggingFaceEmbedderConfig, _builder: Builder):
    """Register HuggingFace local embedder as a provider."""

    description = f"HuggingFace Local Embedder: {config.model_name}"

    yield EmbedderProviderInfo(config=config, description=description)
