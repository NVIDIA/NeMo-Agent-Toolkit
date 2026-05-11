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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from nat.embedder.perplexity_embedder import PerplexityEmbedderModelConfig
from nat.plugins.llama_index.embedder import perplexity_llama_index


class TestPerplexityLlamaIndexRegistration:

    async def test_requires_api_key(self, monkeypatch, mock_builder):
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        cfg = PerplexityEmbedderModelConfig()
        with pytest.raises(ValueError, match="non-empty API key"):
            async with perplexity_llama_index(cfg, mock_builder):
                pass

    async def test_yields_llama_index_embedding(self, monkeypatch, mock_builder):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "env-key")
        cfg = PerplexityEmbedderModelConfig()
        async with perplexity_llama_index(cfg, mock_builder) as client:
            # LlamaIndex's ``BaseEmbedding`` exposes ``get_text_embedding``.
            assert hasattr(client, "get_text_embedding")
            assert hasattr(client, "get_query_embedding")


class TestPerplexityLlamaIndexAdapter:

    def test_get_text_embedding_delegates_to_client(self):
        from nat.plugins.langchain.perplexity_embeddings_client import PerplexityEmbeddings
        from nat.plugins.llama_index.perplexity_embeddings_client import PerplexityLlamaIndexEmbedding

        inner = MagicMock(spec=PerplexityEmbeddings)
        inner._model = "pplx-embed-v1-0.6b"
        inner.embed_query.return_value = [0.1, 0.2, 0.3]
        inner.embed_documents.return_value = [[0.1], [0.2]]
        inner.aembed_query = AsyncMock(return_value=[0.5])
        inner.aembed_documents = AsyncMock(return_value=[[0.6]])

        adapter = PerplexityLlamaIndexEmbedding(client=inner)

        assert adapter._get_query_embedding("q") == [0.1, 0.2, 0.3]
        assert adapter._get_text_embedding("t") == [0.1, 0.2, 0.3]
        assert adapter._get_text_embeddings(["a", "b"]) == [[0.1], [0.2]]

    async def test_async_methods_delegate(self):
        from nat.plugins.langchain.perplexity_embeddings_client import PerplexityEmbeddings
        from nat.plugins.llama_index.perplexity_embeddings_client import PerplexityLlamaIndexEmbedding

        inner = MagicMock(spec=PerplexityEmbeddings)
        inner._model = "pplx-embed-v1-0.6b"
        inner.aembed_query = AsyncMock(return_value=[0.5])
        inner.aembed_documents = AsyncMock(return_value=[[0.6]])

        adapter = PerplexityLlamaIndexEmbedding(client=inner)

        assert await adapter._aget_query_embedding("q") == [0.5]
        assert await adapter._aget_text_embedding("t") == [0.5]
        assert await adapter._aget_text_embeddings(["x"]) == [[0.6]]
