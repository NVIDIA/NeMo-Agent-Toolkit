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

import typing

from pydantic import AliasChoices
from pydantic import ConfigDict
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.embedder import EmbedderProviderInfo
from nat.cli.register_workflow import register_embedder_provider
from nat.data_models.common import OptionalSecretStr
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.ssl_verification_mixin import SSLVerificationMixin

# Supported model identifiers for the Perplexity Embeddings API.
# Standard embeddings are for independent texts (queries/documents).
# Contextualized embeddings are document-aware; chunks from the same document share context.
PerplexityEmbeddingModel = typing.Literal[
    "pplx-embed-v1-0.6b",
    "pplx-embed-v1-4b",
    "pplx-embed-context-v1-0.6b",
    "pplx-embed-context-v1-4b",
]


class PerplexityEmbedderModelConfig(EmbedderBaseConfig, RetryMixin, SSLVerificationMixin, name="perplexity"):
    """A Perplexity Embeddings API provider to be used with an embedder client.

    Perplexity exposes a dedicated embeddings endpoint at
    ``https://api.perplexity.ai/v1/embeddings`` (standard) and
    ``https://api.perplexity.ai/v1/contextualizedembeddings`` (contextualized).
    Authentication uses ``PERPLEXITY_API_KEY``.

    Reference: https://docs.perplexity.ai/api-reference/embeddings-post
    """

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    api_key: OptionalSecretStr = Field(
        default=None,
        description="Perplexity API key to interact with the embeddings endpoint. "
        "Falls back to the ``PERPLEXITY_API_KEY`` environment variable when unset.",
    )
    base_url: str = Field(
        default="https://api.perplexity.ai/v1",
        description="Base URL for the Perplexity API. The embedder appends ``/embeddings`` "
        "for standard models and ``/contextualizedembeddings`` for context models.",
    )
    model_name: PerplexityEmbeddingModel = Field(
        default="pplx-embed-v1-0.6b",
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description="Perplexity embedding model. Standard: ``pplx-embed-v1-0.6b`` (1024-dim) "
        "or ``pplx-embed-v1-4b`` (2560-dim). Contextualized: ``pplx-embed-context-v1-0.6b`` "
        "or ``pplx-embed-context-v1-4b``.",
    )
    dimensions: int | None = Field(
        default=None,
        ge=128,
        le=2560,
        description="Matryoshka output dimensions. Range is 128–1024 for ``0.6b`` models and "
        "128–2560 for ``4b`` models. Defaults to full dimensions when unset.",
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        le=512,
        description="Maximum number of input texts to send per request. The Perplexity API "
        "accepts up to 512 inputs per call (subject to a 120,000 total-token cap).",
    )
    encoding_format: typing.Literal["base64_int8", "base64_binary"] = Field(
        default="base64_int8",
        description="On-wire encoding for the embedding payload. ``base64_int8`` (default) "
        "returns signed int8 values; ``base64_binary`` returns 1-bit-per-dimension packed bits.",
    )


@register_embedder_provider(config_type=PerplexityEmbedderModelConfig)
async def perplexity_embedder_model(config: PerplexityEmbedderModelConfig, _builder: Builder):
    yield EmbedderProviderInfo(
        config=config,
        description="A Perplexity Embeddings API model for use with an Embedder client.",
    )
