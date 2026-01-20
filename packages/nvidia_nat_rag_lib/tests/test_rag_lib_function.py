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
"""Tests for NVIDIA RAG library integration.

KNOWN ISSUE - Embedding Model Compatibility:
    nvidia_rag.EmbeddingConfig always passes a `dimensions` parameter to the embedding API.
    Some models (e.g., nvidia/nv-embedqa-e5-v5) reject this parameter entirely, causing errors:
        "This model does not support 'dimensions', but a value of '2048' was provided."

    Compatible models: nvidia/llama-3.2-nv-embedqa-1b-v2 (supports dimensions param)
    Incompatible models: nvidia/nv-embedqa-e5-v5 (fixed 1024-dim, rejects dimensions param)

    Upstream fix needed: nvidia_rag should allow dimensions=None to not pass the parameter.

TODO: Add integration tests to catch config compatibility issues:
    - Test search/generate with different embedding models
    - Test with different LLM providers
    - Test with different retriever configs (Milvus, NeMo)
    - Parametrize tests across model combinations to catch API rejections early
"""

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
            temperature=0.2,
            top_p=0.95,
            max_tokens=4096,
        ),
    "nim_llm_llama70b":
        NIMModelConfig(
            model_name="meta/llama-3.1-70b-instruct",
            temperature=0.1,
            top_p=0.9,
            max_tokens=4096,
        ),
}

EMBEDDER_CONFIGS: dict[str, NIMEmbedderModelConfig] = {
    # nvidia/llama-3.2-nv-embedqa-1b-v2: supports dimensions parameter
    "nim_embedder": NIMEmbedderModelConfig(model_name="nvidia/llama-3.2-nv-embedqa-1b-v2"),
    # nvidia/nv-embedqa-e5-v5: REJECTS dimensions param
    "nim_embedder_e5": NIMEmbedderModelConfig(model_name="nvidia/nv-embedqa-e5-v5"),
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
# Config Resolution Tests
# =============================================================================


class TestConfigResolution:
    """Test NvidiaRAGLibConfig translation to NvidiaRAGConfig."""

    async def test_resolve_llm_from_ref(self, mock_builder: MagicMock) -> None:
        """Test LLMRef from NvidiaRAGLibConfig resolves to NvidiaRAGConfig.llm."""

        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        from nat.plugins.rag_lib.client import _resolve_llm_config

        rag_config: NvidiaRAGConfig = NvidiaRAGConfig()
        await _resolve_llm_config(LLMRef("nim_llm_llama8b"), mock_builder, rag_config)

        assert rag_config.llm.model_name == "meta/llama-3.1-8b-instruct"
        assert rag_config.llm.parameters.temperature == 0.2
        assert rag_config.llm.parameters.top_p == 0.95
        assert rag_config.llm.parameters.max_tokens == 4096

    async def test_resolve_llm_from_nim_config(self, mock_builder: MagicMock) -> None:
        """Test NIMModelConfig from NvidiaRAGLibConfig resolves to NvidiaRAGConfig.llm."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        from nat.plugins.rag_lib.client import _resolve_llm_config

        nim_config: NIMModelConfig = NIMModelConfig(
            model_name="meta/llama-3.1-8b-instruct",
            base_url="http://nim:8000/v1",
            api_key="direct-api-key",
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
        )

        rag_config: NvidiaRAGConfig = NvidiaRAGConfig()
        await _resolve_llm_config(nim_config, mock_builder, rag_config)

        assert rag_config.llm.model_name == "meta/llama-3.1-8b-instruct"
        assert rag_config.llm.server_url == "http://nim:8000/v1"
        assert rag_config.llm.api_key.get_secret_value() == "direct-api-key"
        assert rag_config.llm.parameters.temperature == 0.7
        assert rag_config.llm.parameters.top_p == 0.9
        assert rag_config.llm.parameters.max_tokens == 2048

    async def test_resolve_llm_none_uses_defaults(self, mock_builder: MagicMock) -> None:
        """Test None llm in NvidiaRAGLibConfig preserves NvidiaRAGConfig defaults."""
        from nvidia_rag.utils.configuration import LLMConfig
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        from nat.plugins.rag_lib.client import _resolve_llm_config

        rag_config: NvidiaRAGConfig = NvidiaRAGConfig()
        original_llm: LLMConfig = rag_config.llm

        await _resolve_llm_config(None, mock_builder, rag_config)

        assert rag_config.llm is original_llm

    async def test_resolve_llm_native_config_passthrough(self, mock_builder: MagicMock) -> None:
        """Test NvidiaRAGLLMConfig passes through to NvidiaRAGConfig unchanged."""
        from nvidia_rag.utils.configuration import LLMConfig as NvidiaRAGLLMConfig
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        from nat.plugins.rag_lib.client import _resolve_llm_config

        native_config: NvidiaRAGLLMConfig = NvidiaRAGLLMConfig(
            model_name="custom/model",
            server_url="http://custom:8000",
            model_engine="custom-engine",
        )

        rag_config: NvidiaRAGConfig = NvidiaRAGConfig()
        await _resolve_llm_config(native_config, mock_builder, rag_config)

        assert rag_config.llm is native_config

    async def test_resolve_llm_unsupported_type_raises(self, mock_builder: MagicMock) -> None:
        """Test unsupported llm type in NvidiaRAGLibConfig raises ValueError."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        from nat.plugins.rag_lib.client import _resolve_llm_config

        rag_config: NvidiaRAGConfig = NvidiaRAGConfig()

        with pytest.raises(ValueError, match="Unsupported LLM config type"):
            await _resolve_llm_config({"invalid": "config"}, mock_builder, rag_config)

    async def test_resolve_embedder_from_ref(self, mock_builder: MagicMock) -> None:
        """Test EmbedderRef from NvidiaRAGLibConfig resolves to NvidiaRAGConfig.embeddings."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        from nat.plugins.rag_lib.client import _resolve_embedder_config

        rag_config: NvidiaRAGConfig = NvidiaRAGConfig()
        await _resolve_embedder_config(EmbedderRef("nim_embedder"), mock_builder, rag_config)

        assert rag_config.embeddings.model_name == "nvidia/llama-3.2-nv-embedqa-1b-v2"

    async def test_resolve_embedder_from_nim_config(self, mock_builder: MagicMock) -> None:
        """Test NIMEmbedderModelConfig from NvidiaRAGLibConfig resolves to NvidiaRAGConfig.embeddings."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        from nat.plugins.rag_lib.client import _resolve_embedder_config

        nim_config: NIMEmbedderModelConfig = NIMEmbedderModelConfig(
            model_name="nvidia/nv-embedqa-e5-v5",
            base_url="http://embedder:8000/v1",
            api_key="direct-embedder-key",
        )

        rag_config: NvidiaRAGConfig = NvidiaRAGConfig()
        await _resolve_embedder_config(nim_config, mock_builder, rag_config)

        assert rag_config.embeddings.model_name == "nvidia/nv-embedqa-e5-v5"
        assert rag_config.embeddings.server_url == "http://embedder:8000/v1"
        assert rag_config.embeddings.api_key.get_secret_value() == "direct-embedder-key"

    async def test_resolve_embedder_none_uses_defaults(self, mock_builder: MagicMock) -> None:
        """Test None embedder in NvidiaRAGLibConfig preserves NvidiaRAGConfig defaults."""
        from nvidia_rag.utils.configuration import EmbeddingConfig
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        from nat.plugins.rag_lib.client import _resolve_embedder_config

        rag_config: NvidiaRAGConfig = NvidiaRAGConfig()
        original_embeddings: EmbeddingConfig = rag_config.embeddings

        await _resolve_embedder_config(None, mock_builder, rag_config)

        assert rag_config.embeddings is original_embeddings

    async def test_resolve_embedder_native_config_passthrough(self, mock_builder: MagicMock) -> None:
        """Test NvidiaRAGEmbeddingConfig passes through to NvidiaRAGConfig unchanged."""
        from nvidia_rag.utils.configuration import EmbeddingConfig as NvidiaRAGEmbeddingConfig
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        from nat.plugins.rag_lib.client import _resolve_embedder_config

        native_config: NvidiaRAGEmbeddingConfig = NvidiaRAGEmbeddingConfig(
            model_name="custom/embedder",
            server_url="http://custom:8000",
        )

        rag_config: NvidiaRAGConfig = NvidiaRAGConfig()
        await _resolve_embedder_config(native_config, mock_builder, rag_config)

        assert rag_config.embeddings is native_config

    async def test_resolve_embedder_unsupported_type_raises(self, mock_builder: MagicMock) -> None:
        """Test unsupported embedder type in NvidiaRAGLibConfig raises ValueError."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        from nat.plugins.rag_lib.client import _resolve_embedder_config

        rag_config: NvidiaRAGConfig = NvidiaRAGConfig()

        with pytest.raises(ValueError, match="Unsupported embedder config type"):
            await _resolve_embedder_config({"invalid": "config"}, mock_builder, rag_config)

    async def test_resolve_retriever_from_ref(self, mock_builder: MagicMock) -> None:
        """Test RetrieverRef from NvidiaRAGLibConfig resolves to NvidiaRAGConfig.vector_store."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        from nat.plugins.rag_lib.client import _resolve_retriever_config

        rag_config: NvidiaRAGConfig = NvidiaRAGConfig()
        await _resolve_retriever_config(RetrieverRef("milvus_retriever"), mock_builder, rag_config)

        assert rag_config.vector_store.name == "milvus"
        assert rag_config.vector_store.url == "http://localhost:19530/"
        assert rag_config.vector_store.default_collection_name == "test_collection"

    async def test_resolve_retriever_from_milvus_config(self, mock_builder: MagicMock) -> None:
        """Test MilvusRetrieverConfig from NvidiaRAGLibConfig resolves to NvidiaRAGConfig.vector_store."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        from nat.plugins.rag_lib.client import _resolve_retriever_config

        milvus_config: MilvusRetrieverConfig = MilvusRetrieverConfig(
            uri=HttpUrl("http://milvus:19530"),
            collection_name="my_collection",
            embedding_model="nvidia/nv-embedqa-e5-v5",
            connection_args={
                "user": "admin", "password": "secret123"
            },
        )

        rag_config: NvidiaRAGConfig = NvidiaRAGConfig()
        await _resolve_retriever_config(milvus_config, mock_builder, rag_config)

        assert rag_config.vector_store.url == "http://milvus:19530/"
        assert rag_config.vector_store.default_collection_name == "my_collection"
        assert rag_config.vector_store.username == "admin"
        assert rag_config.vector_store.password.get_secret_value() == "secret123"

    async def test_resolve_retriever_from_nemo_config(self, mock_builder: MagicMock) -> None:
        """Test NemoRetrieverConfig from NvidiaRAGLibConfig resolves to NvidiaRAGConfig.vector_store."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        from nat.plugins.rag_lib.client import _resolve_retriever_config
        from nat.retriever.nemo_retriever.register import NemoRetrieverConfig

        nemo_config: NemoRetrieverConfig = NemoRetrieverConfig(
            uri=HttpUrl("http://nemo-retriever:8000"),
            collection_name="nemo_collection",
            nvidia_api_key="nemo-api-key",
        )

        rag_config: NvidiaRAGConfig = NvidiaRAGConfig()
        await _resolve_retriever_config(nemo_config, mock_builder, rag_config)

        assert rag_config.vector_store.url == "http://nemo-retriever:8000/"
        assert rag_config.vector_store.default_collection_name == "nemo_collection"
        assert rag_config.vector_store.api_key.get_secret_value() == "nemo-api-key"

    async def test_resolve_retriever_none_uses_defaults(self, mock_builder: MagicMock) -> None:
        """Test None retriever in NvidiaRAGLibConfig preserves NvidiaRAGConfig defaults."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig
        from nvidia_rag.utils.configuration import VectorStoreConfig

        from nat.plugins.rag_lib.client import _resolve_retriever_config

        rag_config: NvidiaRAGConfig = NvidiaRAGConfig()
        original_vector_store: VectorStoreConfig = rag_config.vector_store

        await _resolve_retriever_config(None, mock_builder, rag_config)

        assert rag_config.vector_store is original_vector_store

    async def test_resolve_retriever_native_config_passthrough(self, mock_builder: MagicMock) -> None:
        """Test NvidiaRAGVectorStoreConfig passes through to NvidiaRAGConfig unchanged."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig
        from nvidia_rag.utils.configuration import VectorStoreConfig as NvidiaRAGVectorStoreConfig

        from nat.plugins.rag_lib.client import _resolve_retriever_config

        native_config: NvidiaRAGVectorStoreConfig = NvidiaRAGVectorStoreConfig(
            name="custom",
            url="http://custom:19530",
        )

        rag_config: NvidiaRAGConfig = NvidiaRAGConfig()
        await _resolve_retriever_config(native_config, mock_builder, rag_config)

        assert rag_config.vector_store is native_config

    async def test_resolve_retriever_unsupported_type_raises(self, mock_builder: MagicMock) -> None:
        """Test unsupported retriever type in NvidiaRAGLibConfig raises ValueError."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        from nat.plugins.rag_lib.client import _resolve_retriever_config

        rag_config: NvidiaRAGConfig = NvidiaRAGConfig()

        with pytest.raises(ValueError, match="Unsupported retriever config type"):
            await _resolve_retriever_config({"invalid": "config"}, mock_builder, rag_config)  # type: ignore[arg-type]


# =============================================================================
# RAGPipelineConfig Mapping Tests
# =============================================================================


class TestRAGPipelineConfigMapping:
    """Test RAGPipelineConfig fields are correctly mapped to NvidiaRAGConfig."""

    async def test_pipeline_fields_with_defaults(self, mock_builder: MagicMock) -> None:
        """Test that default RAGPipelineConfig creates valid NvidiaRAGConfig with all required fields."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig

        from nat.plugins.rag_lib.client import NvidiaRAGLibConfig
        from nat.plugins.rag_lib.client import _build_nvidia_rag_config

        config = NvidiaRAGLibConfig()
        rag_config: NvidiaRAGConfig = await _build_nvidia_rag_config(config, mock_builder)

        # Fields that always have values (from default_factory)
        assert rag_config.ranking is not None
        assert rag_config.retriever is not None

        # Fields that get defaults when None
        assert rag_config.vlm is not None
        assert rag_config.query_rewriter is not None
        assert rag_config.filter_expression_generator is not None
        assert rag_config.query_decomposition is not None
        assert rag_config.reflection is not None

        # Boolean flags have correct defaults
        assert rag_config.enable_citations is True
        assert rag_config.enable_guardrails is False
        assert rag_config.enable_vlm_inference is False
        assert rag_config.vlm_to_llm_fallback is True
        assert rag_config.default_confidence_threshold == 0.0

    async def test_pipeline_fields_passthrough(self, mock_builder: MagicMock) -> None:
        """Test that explicit RAGPipelineConfig values are passed through to NvidiaRAGConfig."""
        from nvidia_rag.utils.configuration import NvidiaRAGConfig
        from nvidia_rag.utils.configuration import QueryRewriterConfig
        from nvidia_rag.utils.configuration import RankingConfig
        from nvidia_rag.utils.configuration import RetrieverConfig
        from nvidia_rag.utils.configuration import VLMConfig

        from nat.plugins.rag_lib.client import NvidiaRAGLibConfig
        from nat.plugins.rag_lib.client import _build_nvidia_rag_config
        from nat.plugins.rag_lib.config import RAGPipelineConfig

        custom_ranking = RankingConfig(enable_reranker=False)
        custom_retriever = RetrieverConfig(top_k=20, vdb_top_k=200)
        custom_vlm = VLMConfig(model_name="custom/vlm-model", temperature=0.5)
        custom_query_rewriter = QueryRewriterConfig(enable_query_rewriter=True)

        pipeline = RAGPipelineConfig(
            ranking=custom_ranking,
            search_settings=custom_retriever,
            vlm=custom_vlm,
            query_rewriter=custom_query_rewriter,
            enable_citations=False,
            enable_guardrails=True,
            enable_vlm_inference=True,
            vlm_to_llm_fallback=False,
            default_confidence_threshold=0.5,
        )

        config = NvidiaRAGLibConfig(rag_pipeline=pipeline)
        rag_config: NvidiaRAGConfig = await _build_nvidia_rag_config(config, mock_builder)

        # Explicit values passed through
        assert rag_config.ranking.enable_reranker is False
        assert rag_config.retriever.top_k == 20
        assert rag_config.retriever.vdb_top_k == 200
        assert rag_config.vlm.model_name == "custom/vlm-model"
        assert rag_config.vlm.temperature == 0.5
        assert rag_config.query_rewriter.enable_query_rewriter is True

        # Boolean flags passed through
        assert rag_config.enable_citations is False
        assert rag_config.enable_guardrails is True
        assert rag_config.enable_vlm_inference is True
        assert rag_config.vlm_to_llm_fallback is False
        assert rag_config.default_confidence_threshold == 0.5

    async def test_none_optional_fields_get_defaults(self, mock_builder: MagicMock) -> None:
        """Test that None optional fields receive proper default config objects."""
        from nvidia_rag.utils.configuration import FilterExpressionGeneratorConfig
        from nvidia_rag.utils.configuration import NvidiaRAGConfig
        from nvidia_rag.utils.configuration import QueryDecompositionConfig
        from nvidia_rag.utils.configuration import QueryRewriterConfig
        from nvidia_rag.utils.configuration import ReflectionConfig
        from nvidia_rag.utils.configuration import VLMConfig

        from nat.plugins.rag_lib.client import NvidiaRAGLibConfig
        from nat.plugins.rag_lib.client import _build_nvidia_rag_config
        from nat.plugins.rag_lib.config import RAGPipelineConfig

        # Explicitly set optional fields to None
        pipeline = RAGPipelineConfig(
            vlm=None,
            query_rewriter=None,
            filter_generator=None,
            query_decomposition=None,
            reflection=None,
        )

        config = NvidiaRAGLibConfig(rag_pipeline=pipeline)
        rag_config: NvidiaRAGConfig = await _build_nvidia_rag_config(config, mock_builder)

        # All should be valid config objects, not None
        assert isinstance(rag_config.vlm, VLMConfig)
        assert isinstance(rag_config.query_rewriter, QueryRewriterConfig)
        assert isinstance(rag_config.filter_expression_generator, FilterExpressionGeneratorConfig)
        assert isinstance(rag_config.query_decomposition, QueryDecompositionConfig)
        assert isinstance(rag_config.reflection, ReflectionConfig)


# =============================================================================
# NvidiaRAG Functional Tests
# =============================================================================


class TestNvidiaRAGMethods:
    """Test NvidiaRAG class can be imported and has expected methods."""

    def test_import_and_instantiate_nvidia_rag(self) -> None:
        """Verify nvidia_rag can be imported and instantiated."""
        from nvidia_rag import NvidiaRAG

        rag = NvidiaRAG()
        assert rag is not None
        assert isinstance(rag, NvidiaRAG)

    def test_generate_method_exists(self) -> None:
        """NvidiaRAG should have a generate method."""
        from nvidia_rag import NvidiaRAG

        assert hasattr(NvidiaRAG, "generate")
        assert callable(getattr(NvidiaRAG, "generate"))

    def test_search_method_exists(self) -> None:
        """NvidiaRAG should have a search method."""
        from nvidia_rag import NvidiaRAG

        assert hasattr(NvidiaRAG, "search")
        assert callable(getattr(NvidiaRAG, "search"))

    def test_health_method_exists(self) -> None:
        """NvidiaRAG should have a health method."""
        from nvidia_rag import NvidiaRAG

        assert hasattr(NvidiaRAG, "health")
        assert callable(getattr(NvidiaRAG, "health"))


@pytest.mark.integration
class TestNvidiaRAGIntegration:
    """Parameterized tests for NvidiaRAG generate(), search(), and health() methods."""

    @pytest.fixture(name="create_collection")
    def fixture_create_collection(self):
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
            client = MilvusClient(uri="http://localhost:19530")
            if client.has_collection(collection_name):
                client.drop_collection(collection_name)

            embeddings = NVIDIAEmbeddings(model=model_name)
            Milvus.from_documents(
                documents=[Document(page_content="Test document", metadata={"source": "test"})],
                embedding=embeddings,
                collection_name=collection_name,
                connection_args={"uri": "http://localhost:19530"},
            )
            created.append(collection_name)
            return collection_name

        yield _create

        client = MilvusClient(uri="http://localhost:19530")
        for name in created:
            if client.has_collection(name):
                client.drop_collection(name)

    @pytest.mark.parametrize("llm_ref", list(LLM_CONFIGS.keys()))
    @pytest.mark.parametrize(
        "embedder_ref",
        [
            "nim_embedder",
            # TODO: nvidia_rag always passes dimensions param which nv-embedqa-e5-v5 rejects
            # "nim_embedder_e5",
        ])
    @pytest.mark.parametrize("retriever_ref", list(RETRIEVER_CONFIGS.keys()))
    async def test_search(
        self,
        mock_builder: MagicMock,
        create_collection,
        llm_ref: str,
        embedder_ref: str,
        retriever_ref: str,
    ) -> None:
        """Test NvidiaRAG search() with different component configs."""
        from nvidia_rag import NvidiaRAG

        from nat.plugins.rag_lib.client import NvidiaRAGLibConfig
        from nat.plugins.rag_lib.client import _build_nvidia_rag_config

        collection_name = create_collection(embedder_ref)

        config = NvidiaRAGLibConfig(
            llm=LLMRef(llm_ref),
            embedder=EmbedderRef(embedder_ref),
            retriever=RetrieverRef(retriever_ref),
        )
        rag_config = await _build_nvidia_rag_config(config, mock_builder)
        rag_config.vector_store.default_collection_name = collection_name
        rag = NvidiaRAG(config=rag_config)

        result = await rag.search(query="test query")

        assert result is not None

    @pytest.mark.parametrize("llm_ref", list(LLM_CONFIGS.keys()))
    @pytest.mark.parametrize(
        "embedder_ref",
        [
            "nim_embedder",
            # TODO: nvidia_rag always passes dimensions param which nv-embedqa-e5-v5 rejects
            # "nim_embedder_e5",
        ])
    @pytest.mark.parametrize("retriever_ref", list(RETRIEVER_CONFIGS.keys()))
    async def test_generate(
        self,
        mock_builder: MagicMock,
        llm_ref: str,
        embedder_ref: str,
        retriever_ref: str,
    ) -> None:
        """Test NvidiaRAG generate() with different component configs."""
        from nvidia_rag import NvidiaRAG

        from nat.plugins.rag_lib.client import NvidiaRAGLibConfig
        from nat.plugins.rag_lib.client import _build_nvidia_rag_config

        config = NvidiaRAGLibConfig(
            llm=LLMRef(llm_ref),
            embedder=EmbedderRef(embedder_ref),
            retriever=RetrieverRef(retriever_ref),
        )
        rag_config = await _build_nvidia_rag_config(config, mock_builder)
        rag = NvidiaRAG(config=rag_config)

        messages = [{"role": "user", "content": "What is RAG?"}]
        result = await rag.generate(messages=messages, use_knowledge_base=False)

        assert result is not None

    @pytest.mark.parametrize("llm_ref", list(LLM_CONFIGS.keys()))
    @pytest.mark.parametrize(
        "embedder_ref",
        [
            "nim_embedder",
            # TODO: nvidia_rag always passes dimensions param which nv-embedqa-e5-v5 rejects
            # "nim_embedder_e5",
        ])
    @pytest.mark.parametrize("retriever_ref", list(RETRIEVER_CONFIGS.keys()))
    async def test_health(
        self,
        mock_builder: MagicMock,
        llm_ref: str,
        embedder_ref: str,
        retriever_ref: str,
    ) -> None:
        """Test NvidiaRAG health() with different component configs."""
        from nvidia_rag import NvidiaRAG

        from nat.plugins.rag_lib.client import NvidiaRAGLibConfig
        from nat.plugins.rag_lib.client import _build_nvidia_rag_config

        config = NvidiaRAGLibConfig(
            llm=LLMRef(llm_ref),
            embedder=EmbedderRef(embedder_ref),
            retriever=RetrieverRef(retriever_ref),
        )
        rag_config = await _build_nvidia_rag_config(config, mock_builder)
        rag = NvidiaRAG(config=rag_config)

        result = await rag.health()

        assert result is not None
