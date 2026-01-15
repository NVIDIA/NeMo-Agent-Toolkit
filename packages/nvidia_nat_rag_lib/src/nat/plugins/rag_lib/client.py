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

import logging
from logging import Logger

from nvidia_rag.utils.configuration import EmbeddingConfig as NvidiaRAGEmbeddingConfig
from nvidia_rag.utils.configuration import FilterExpressionGeneratorConfig as NvidiaRAGFilterGeneratorConfig
from nvidia_rag.utils.configuration import LLMConfig as NvidiaRAGLLMConfig
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.configuration import QueryDecompositionConfig as NvidiaRAGQueryDecompositionConfig
from nvidia_rag.utils.configuration import QueryRewriterConfig as NvidiaRAGQueryRewriterConfig
from nvidia_rag.utils.configuration import ReflectionConfig as NvidiaRAGReflectionConfig
from nvidia_rag.utils.configuration import VectorStoreConfig as NvidiaRAGVectorStoreConfig
from nvidia_rag.utils.configuration import VLMConfig as NvidiaRAGVLMConfig
from pydantic import Field
from pydantic import SecretStr

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import RetrieverRef
from nat.data_models.function import FunctionBaseConfig
from nat.embedder.nim_embedder import NIMEmbedderModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.plugins.rag_lib.config import EmbedderConfigType
from nat.plugins.rag_lib.config import LLMConfigType
from nat.plugins.rag_lib.config import RAGPipelineConfig
from nat.plugins.rag_lib.config import RetrieverConfigType
from nat.retriever.milvus.register import MilvusRetrieverConfig
from nat.retriever.nemo_retriever.register import NemoRetrieverConfig

logger: Logger = logging.getLogger(__name__)


class NvidiaRAGLibConfig(FunctionBaseConfig, name="nvidia_rag_lib"):
    """Configuration for NVIDIA RAG Library.

    All component configs are optional - NvidiaRAGConfig provides defaults.
    """

    llm: LLMConfigType = Field(default=None, description="LLM configuration")
    embedder: EmbedderConfigType = Field(default=None, description="Embedder configuration")
    retriever: RetrieverConfigType = Field(default=None, description="Vector store configuration")

    rag_pipeline: RAGPipelineConfig = Field(default_factory=RAGPipelineConfig)


@register_function(config_type=NvidiaRAGLibConfig)  # type: ignore[arg-type]
async def nvidia_rag_lib(config: NvidiaRAGLibConfig, builder: Builder):
    """Initialize NVIDIA RAG with flexible config resolution."""
    try:
        from nvidia_rag import NvidiaRAG
    except ImportError as e:
        raise ImportError("nvidia-rag package is not installed.") from e

    rag_config: NvidiaRAGConfig = await build_nvidia_rag_config(config, builder)
    logger.info("NVIDIA RAG initialized")
    yield NvidiaRAG(config=rag_config)


async def build_nvidia_rag_config(config: NvidiaRAGLibConfig, builder: Builder) -> NvidiaRAGConfig:
    """Build NvidiaRAGConfig by resolving NAT refs/components to nvidia_rag configs."""

    pipeline: RAGPipelineConfig = config.rag_pipeline

    # Create base config with pipeline settings and defaults
    rag_config: NvidiaRAGConfig = NvidiaRAGConfig(
        ranking=pipeline.ranking,
        retriever=pipeline.search_settings,
        vlm=pipeline.vlm or NvidiaRAGVLMConfig(),
        query_rewriter=pipeline.query_rewriter or NvidiaRAGQueryRewriterConfig(),
        filter_expression_generator=pipeline.filter_generator or NvidiaRAGFilterGeneratorConfig(),
        query_decomposition=pipeline.query_decomposition or NvidiaRAGQueryDecompositionConfig(),
        reflection=pipeline.reflection or NvidiaRAGReflectionConfig(),
        enable_citations=pipeline.enable_citations,
        enable_guardrails=pipeline.enable_guardrails,
        enable_vlm_inference=pipeline.enable_vlm_inference,
        vlm_to_llm_fallback=pipeline.vlm_to_llm_fallback,
        default_confidence_threshold=pipeline.default_confidence_threshold,
    )

    # Resolve and map each component's fields (mutates rag_config)
    await _resolve_llm_config(config.llm, builder, rag_config)
    await _resolve_embedder_config(config.embedder, builder, rag_config)
    await _resolve_retriever_config(config.retriever, builder, rag_config)

    return rag_config


async def _resolve_llm_config(llm: LLMConfigType, builder: Builder, rag_config: NvidiaRAGConfig) -> None:
    """Resolve LLM config and map all fields to NvidiaRAGConfig.llm."""

    if llm is None:
        return

    if isinstance(llm, NvidiaRAGLLMConfig):
        rag_config.llm = llm
        return

    if isinstance(llm, LLMRef):
        llm = builder.get_llm_config(llm)

    if isinstance(llm, NIMModelConfig):
        rag_config.llm.model_name = llm.model_name
        if llm.base_url:
            rag_config.llm.server_url = llm.base_url
        if llm.api_key:
            rag_config.llm.api_key = llm.api_key
        if llm.temperature is not None:
            rag_config.llm.parameters.temperature = llm.temperature
        if llm.top_p is not None:
            rag_config.llm.parameters.top_p = llm.top_p
        if llm.max_tokens is not None:
            rag_config.llm.parameters.max_tokens = llm.max_tokens
        return

    raise ValueError(f"Unsupported LLM config type: {type(llm)}")


async def _resolve_embedder_config(embedder: EmbedderConfigType, builder: Builder, rag_config: NvidiaRAGConfig) -> None:
    """Resolve embedder config and map all fields to NvidiaRAGConfig.embeddings."""

    if embedder is None:
        return

    if isinstance(embedder, NvidiaRAGEmbeddingConfig):
        rag_config.embeddings = embedder
        return

    if isinstance(embedder, EmbedderRef):
        embedder = builder.get_embedder_config(embedder)

    if isinstance(embedder, NIMEmbedderModelConfig):
        rag_config.embeddings.model_name = embedder.model_name
        if embedder.base_url:
            rag_config.embeddings.server_url = embedder.base_url
        if embedder.api_key:
            rag_config.embeddings.api_key = embedder.api_key
        return

    raise ValueError(f"Unsupported embedder config type: {type(embedder)}")


async def _resolve_retriever_config(retriever: RetrieverConfigType, builder: Builder,
                                    rag_config: NvidiaRAGConfig) -> None:
    """Resolve retriever config and map all fields to NvidiaRAGConfig.vector_store."""

    if retriever is None:
        return

    if isinstance(retriever, NvidiaRAGVectorStoreConfig):
        rag_config.vector_store = retriever
        return

    if isinstance(retriever, RetrieverRef):
        retriever = await builder.get_retriever_config(retriever)

    if isinstance(retriever, MilvusRetrieverConfig):
        rag_config.vector_store.url = str(retriever.uri)
        if retriever.collection_name:
            rag_config.vector_store.default_collection_name = retriever.collection_name
        if retriever.connection_args:
            if "user" in retriever.connection_args:
                rag_config.vector_store.username = retriever.connection_args["user"]
            if "password" in retriever.connection_args:
                rag_config.vector_store.password = SecretStr(retriever.connection_args["password"])
        return

    if isinstance(retriever, NemoRetrieverConfig):
        rag_config.vector_store.url = str(retriever.uri)
        if retriever.collection_name:
            rag_config.vector_store.default_collection_name = retriever.collection_name
        if retriever.nvidia_api_key:
            rag_config.vector_store.api_key = retriever.nvidia_api_key
        return

    raise ValueError(f"Unsupported retriever config type: {type(retriever)}")
