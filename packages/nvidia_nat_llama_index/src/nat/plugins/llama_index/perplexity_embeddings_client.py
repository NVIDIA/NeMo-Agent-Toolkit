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
"""LlamaIndex ``BaseEmbedding`` client for the Perplexity Embeddings API.

Wraps the framework-agnostic ``PerplexityEmbeddings`` LangChain client to satisfy
LlamaIndex's :class:`~llama_index.core.embeddings.BaseEmbedding` interface so the
same NAT config can be consumed by LlamaIndex-powered retrievers.
"""
from __future__ import annotations

from typing import Any

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding

from nat.plugins.langchain.perplexity_embeddings_client import PerplexityEmbeddings


class PerplexityLlamaIndexEmbedding(BaseEmbedding):
    """LlamaIndex embedding adapter for the Perplexity Embeddings API."""

    _client: PerplexityEmbeddings = PrivateAttr()

    def __init__(self, *, client: PerplexityEmbeddings, **data: Any) -> None:
        data.setdefault("model_name", getattr(client, "_model", "pplx-embed-v1-0.6b"))
        super().__init__(**data)
        self._client = client

    @classmethod
    def class_name(cls) -> str:
        return "PerplexityLlamaIndexEmbedding"

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------
    def _get_query_embedding(self, query: str) -> list[float]:
        return self._client.embed_query(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._client.embed_query(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------
    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await self._client.aembed_query(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await self._client.aembed_query(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await self._client.aembed_documents(texts)
