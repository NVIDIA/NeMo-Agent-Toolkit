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
"""Tests for NVIDIA RAG library integration."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from pydantic import HttpUrl

from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import RetrieverRef
from nat.embedder.nim_embedder import NIMEmbedderModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.retriever.milvus.register import MilvusRetrieverConfig

# NOTE: First nvidia_rag import takes ~20s due to module-level initialization.

# =============================================================================
# Fixtures
# =============================================================================

LLM_CONFIGS: dict[str, NIMModelConfig] = {
    "nim_llm_llama8b":
        NIMModelConfig(
            model_name="meta/llama-3.1-8b-instruct",
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=0.2,
            top_p=0.95,
            max_tokens=4096,
        ),
    "nim_llm_llama70b":
        NIMModelConfig(
            model_name="meta/llama-3.1-70b-instruct",
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=0.1,
            top_p=0.9,
            max_tokens=4096,
        ),
}

EMBEDDER_CONFIGS: dict[str, NIMEmbedderModelConfig] = {
    # nvidia/llama-3.2-nv-embedqa-1b-v2: supports dimensions parameter
    "nim_embedder":
        NIMEmbedderModelConfig(
            model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
            base_url="https://integrate.api.nvidia.com/v1",
        ),  # nvidia/nv-embedqa-e5-v5: REJECTS dimensions param
    "nim_embedder_e5":
        NIMEmbedderModelConfig(
            model_name="nvidia/nv-embedqa-e5-v5",
            base_url="https://integrate.api.nvidia.com/v1",
        ),
}

RETRIEVER_CONFIGS: dict[str, MilvusRetrieverConfig] = {
    "milvus_retriever":
        MilvusRetrieverConfig(
            uri=HttpUrl("http://localhost:19530"),
            collection_name="test_collection",
            embedding_model="nim_embedder",
        ),
}


@pytest.fixture(name="mock_builder")
def fixture_mock_builder() -> MagicMock:
    """Create mock NAT builder with component resolution."""
    builder: MagicMock = MagicMock()

    def get_llm_config(ref: LLMRef) -> NIMModelConfig:
        return LLM_CONFIGS[str(ref)]

    builder.get_llm_config = MagicMock(side_effect=get_llm_config)

    def get_embedder_config(ref: EmbedderRef) -> NIMEmbedderModelConfig:
        return EMBEDDER_CONFIGS[str(ref)]

    builder.get_embedder_config = MagicMock(side_effect=get_embedder_config)

    async def get_retriever_config(ref: RetrieverRef) -> MilvusRetrieverConfig:
        return RETRIEVER_CONFIGS[str(ref)]

    builder.get_retriever_config = AsyncMock(side_effect=get_retriever_config)

    return builder


# =============================================================================
# NvidiaRAG Functional Tests
# =============================================================================


class TestNvidiaRAGMethods:
    """Test NvidiaRAG class can be imported and has expected methods."""

    def test_import_and_instantiate_nvidia_rag(self) -> None:
        """Verify nvidia_rag can be imported and instantiated."""
        from nvidia_rag.rag_server.main import NvidiaRAG

        rag = NvidiaRAG()
        assert rag is not None
        assert isinstance(rag, NvidiaRAG)

    def test_generate_method_exists(self) -> None:
        """NvidiaRAG should have a generate method."""
        from nvidia_rag.rag_server.main import NvidiaRAG

        assert hasattr(NvidiaRAG, "generate")
        assert callable(NvidiaRAG.generate)

    def test_search_method_exists(self) -> None:
        """NvidiaRAG should have a search method."""
        from nvidia_rag.rag_server.main import NvidiaRAG

        assert hasattr(NvidiaRAG, "search")
        assert callable(NvidiaRAG.search)

    def test_health_method_exists(self) -> None:
        """NvidiaRAG should have a health method."""
        from nvidia_rag.rag_server.main import NvidiaRAG

        assert hasattr(NvidiaRAG, "health")
        assert callable(NvidiaRAG.health)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestNvidiaRAGIntegration:
    """Integration tests for NvidiaRAG with live services."""

    @pytest.fixture(name="create_collection")
    def fixture_create_collection(self, milvus_uri: str):
        """Factory to create Milvus collections with specific embedding models."""
        from langchain_core.documents import Document
        from langchain_milvus import Milvus
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        from pymilvus import MilvusClient

        created: list[str] = []

        def _create(embedder_ref: str) -> str:
            import re

            model_name = EMBEDDER_CONFIGS[embedder_ref].model_name
            sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", model_name)
            collection_name = f"test_{sanitized}"
            client = MilvusClient(uri=milvus_uri)
            if client.has_collection(collection_name):
                client.drop_collection(collection_name)

            embeddings = NVIDIAEmbeddings(model=model_name)
            Milvus.from_documents(
                documents=[Document(page_content="Test document", metadata={"source": "test"})],
                embedding=embeddings,
                collection_name=collection_name,
                connection_args={"uri": milvus_uri},
            )
            created.append(collection_name)
            return collection_name

        yield _create

        client = MilvusClient(uri=milvus_uri)
        for name in created:
            if client.has_collection(name):
                client.drop_collection(name)

    @pytest.mark.parametrize("llm_ref", list(LLM_CONFIGS.keys()))
    @pytest.mark.parametrize(
        "embedder_ref",
        [
            "nim_embedder",
            pytest.param(
                "nim_embedder_e5",
                marks=pytest.mark.xfail(reason="nvidia_rag passes dimensions param which nv-embedqa-e5-v5 rejects")),
        ])
    @pytest.mark.parametrize("retriever_ref", list(RETRIEVER_CONFIGS.keys()))
    async def test_search(
        self,
        mock_builder: MagicMock,
        create_collection,
        milvus_uri: str,
        llm_ref: str,
        embedder_ref: str,
        retriever_ref: str,
    ) -> None:
        """Test NvidiaRAG search() with different component configs."""
        from nvidia_rag.rag_server.main import NvidiaRAG
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        collection_name = create_collection(embedder_ref)

        llm_config = LLM_CONFIGS[llm_ref]
        embedder_config = EMBEDDER_CONFIGS[embedder_ref]

        rag_config = NvidiaRAGConfig()
        rag_config.llm.model_name = llm_config.model_name
        rag_config.llm.server_url = llm_config.base_url
        rag_config.embeddings.model_name = embedder_config.model_name
        rag_config.embeddings.server_url = embedder_config.base_url
        rag_config.vector_store.url = milvus_uri
        rag_config.vector_store.default_collection_name = collection_name

        rag = NvidiaRAG(config=rag_config)
        result = await rag.search(query="test query")

        assert result is not None

    @pytest.mark.parametrize("llm_ref", list(LLM_CONFIGS.keys()))
    @pytest.mark.parametrize(
        "embedder_ref",
        [
            "nim_embedder",
            pytest.param(
                "nim_embedder_e5",
                marks=pytest.mark.xfail(reason="nvidia_rag passes dimensions param which nv-embedqa-e5-v5 rejects")),
        ])
    @pytest.mark.parametrize("retriever_ref", list(RETRIEVER_CONFIGS.keys()))
    async def test_generate(
        self,
        mock_builder: MagicMock,
        milvus_uri: str,
        llm_ref: str,
        embedder_ref: str,
        retriever_ref: str,
    ) -> None:
        """Test NvidiaRAG generate() with different component configs."""
        from nvidia_rag.rag_server.main import NvidiaRAG
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        llm_config = LLM_CONFIGS[llm_ref]
        embedder_config = EMBEDDER_CONFIGS[embedder_ref]

        rag_config = NvidiaRAGConfig()
        rag_config.llm.model_name = llm_config.model_name
        rag_config.llm.server_url = llm_config.base_url
        rag_config.embeddings.model_name = embedder_config.model_name
        rag_config.embeddings.server_url = embedder_config.base_url
        rag_config.vector_store.url = milvus_uri

        rag = NvidiaRAG(config=rag_config)
        messages = [{"role": "user", "content": "What is RAG?"}]
        result = await rag.generate(messages=messages, use_knowledge_base=False)

        assert result is not None

    @pytest.mark.parametrize("llm_ref", list(LLM_CONFIGS.keys()))
    @pytest.mark.parametrize(
        "embedder_ref",
        [
            "nim_embedder",
            pytest.param(
                "nim_embedder_e5",
                marks=pytest.mark.xfail(reason="nvidia_rag passes dimensions param which nv-embedqa-e5-v5 rejects")),
        ])
    @pytest.mark.parametrize("retriever_ref", list(RETRIEVER_CONFIGS.keys()))
    async def test_health(
        self,
        mock_builder: MagicMock,
        milvus_uri: str,
        llm_ref: str,
        embedder_ref: str,
        retriever_ref: str,
    ) -> None:
        """Test NvidiaRAG health() with different component configs."""
        from nvidia_rag.rag_server.main import NvidiaRAG
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        llm_config = LLM_CONFIGS[llm_ref]
        embedder_config = EMBEDDER_CONFIGS[embedder_ref]

        rag_config = NvidiaRAGConfig()
        rag_config.llm.model_name = llm_config.model_name
        rag_config.llm.server_url = llm_config.base_url
        rag_config.embeddings.model_name = embedder_config.model_name
        rag_config.embeddings.server_url = embedder_config.base_url
        rag_config.vector_store.url = milvus_uri

        rag = NvidiaRAG(config=rag_config)
        result = await rag.health()

        assert result is not None
