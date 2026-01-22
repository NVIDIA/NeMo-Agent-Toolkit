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
from collections.abc import AsyncGenerator
from logging import Logger
from typing import TYPE_CHECKING

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
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import RetrieverRef
from nat.data_models.finetuning import OpenAIMessage
from nat.data_models.function import FunctionGroupBaseConfig
from nat.embedder.nim_embedder import NIMEmbedderModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.plugins.rag_lib.config import EmbedderConfigType
from nat.plugins.rag_lib.config import LLMConfigType
from nat.plugins.rag_lib.config import RAGPipelineConfig
from nat.plugins.rag_lib.config import RetrieverConfigType
from nat.plugins.rag_lib.models import RAGGenerateResult
from nat.plugins.rag_lib.models import RAGSearchResult
from nat.retriever.milvus.register import MilvusRetrieverConfig
from nat.retriever.nemo_retriever.register import NemoRetrieverConfig

if TYPE_CHECKING:
    from nvidia_rag.rag_server.response_generator import RAGResponse

logger: Logger = logging.getLogger(__name__)


class NvidiaRAGLibConfig(FunctionGroupBaseConfig, name="nvidia_rag_lib"):
    """Configuration for NVIDIA RAG Library.

    Exposes search and generate tools that share a single RAG client.
    """

    llm: LLMConfigType = Field(default=None, description="LLM configuration")
    embedder: EmbedderConfigType = Field(default=None, description="Embedder configuration")
    retriever: RetrieverConfigType = Field(default=None, description="Vector store configuration")
    rag_pipeline: RAGPipelineConfig = Field(default_factory=RAGPipelineConfig)

    topic: str | None = Field(default=None, description="Topic for tool descriptions.")
    collection_names: list[str] | None = Field(default=None, description="Collections to query.")
    reranker_top_k: int = Field(default=10, ge=1, description="Number of results after reranking.")


@register_function_group(config_type=NvidiaRAGLibConfig)
async def nvidia_rag_lib(config: NvidiaRAGLibConfig, builder: Builder):
    """NVIDIA RAG Library - exposes search and generate tools."""
    try:
        from nvidia_rag import NvidiaRAG
    except ImportError as e:
        raise ImportError("nvidia-rag package is not installed.") from e

    rag_config: NvidiaRAGConfig = await _build_nvidia_rag_config(config, builder)
    rag_client: NvidiaRAG = NvidiaRAG(config=rag_config)
    logger.info("NVIDIA RAG client initialized")

    topic_str: str = f" about {config.topic}" if config.topic else ""

    async def search(query: str) -> RAGSearchResult:
        """Search for relevant documents."""
        from nvidia_rag.rag_server.response_generator import Citations

        try:
            citations: Citations = await rag_client.search(
                query=query,
                collection_names=config.collection_names,
                reranker_top_k=config.reranker_top_k,
            )
            return RAGSearchResult(citations=citations)
        except Exception:
            logger.exception("RAG search failed")
            return RAGSearchResult(citations=Citations(total_results=0, results=[]))

    async def generate(query: str) -> RAGGenerateResult:
        """Generate an answer using the knowledge base."""
        from nvidia_rag.rag_server.response_generator import ChainResponse
        from nvidia_rag.rag_server.response_generator import Citations

        chunks: list[str] = []
        final_citations: Citations | None = None

        try:
            response: RAGResponse = await rag_client.generate(
                messages=[OpenAIMessage(role="user", content=query).model_dump()],
                collection_names=config.collection_names,
                reranker_top_k=config.reranker_top_k,
            )

            stream: AsyncGenerator[str, None] = (response.generator if hasattr(response, "generator") else response)

            async for raw_chunk in stream:
                if raw_chunk.startswith("data: "):
                    raw_chunk = raw_chunk[len("data: "):].strip()
                if not raw_chunk or raw_chunk == "[DONE]":
                    continue
                try:
                    parsed: ChainResponse = ChainResponse.model_validate_json(raw_chunk)
                    if parsed.choices:
                        choice = parsed.choices[0]
                        if choice.delta and choice.delta.content:
                            content = choice.delta.content
                            if isinstance(content, str):
                                chunks.append(content)
                    if parsed.citations and parsed.citations.results:
                        final_citations = parsed.citations
                except Exception:
                    continue

            answer: str = "".join(chunks) if chunks else "No response generated."
            return RAGGenerateResult(answer=answer, citations=final_citations)

        except Exception:
            logger.exception("RAG generate failed")
            return RAGGenerateResult(answer="Error generating answer. Please try again.")

    group = FunctionGroup(config=config)

    group.add_function(
        "search",
        search,
        description=(
            f"Retrieve grounded excerpts{topic_str}. "
            "Returns document chunks from indexed sources - use this to ground your response in cited source material "
            "rather than general knowledge."),
    )
    group.add_function(
        "generate",
        generate,
        description=(f"Generate a grounded, cited answer{topic_str}. "
                     "Synthesizes an answer from retrieved documents, ensuring the response is grounded in cited "
                     "source material rather than general knowledge."),
    )
    yield group


async def _build_nvidia_rag_config(config: NvidiaRAGLibConfig, builder: Builder) -> NvidiaRAGConfig:
    """Build NvidiaRAGConfig by resolving NAT refs/components to nvidia_rag configs."""

    pipeline: RAGPipelineConfig = config.rag_pipeline

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

        if "model_name" not in rag_config.query_rewriter.model_fields_set:
            rag_config.query_rewriter.model_name = llm.model_name
        if "server_url" not in rag_config.query_rewriter.model_fields_set and llm.base_url:
            rag_config.query_rewriter.server_url = llm.base_url
        if "api_key" not in rag_config.query_rewriter.model_fields_set and llm.api_key:
            rag_config.query_rewriter.api_key = llm.api_key

        if "model_name" not in rag_config.reflection.model_fields_set:
            rag_config.reflection.model_name = llm.model_name
        if "server_url" not in rag_config.reflection.model_fields_set and llm.base_url:
            rag_config.reflection.server_url = llm.base_url
        if "api_key" not in rag_config.reflection.model_fields_set and llm.api_key:
            rag_config.reflection.api_key = llm.api_key

        if "model_name" not in rag_config.filter_expression_generator.model_fields_set:
            rag_config.filter_expression_generator.model_name = llm.model_name
        if "server_url" not in rag_config.filter_expression_generator.model_fields_set and llm.base_url:
            rag_config.filter_expression_generator.server_url = llm.base_url
        if "api_key" not in rag_config.filter_expression_generator.model_fields_set and llm.api_key:
            rag_config.filter_expression_generator.api_key = llm.api_key
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
        if embedder.dimensions is not None:
            rag_config.embeddings.dimensions = embedder.dimensions
        return

    raise ValueError(f"Unsupported embedder config type: {type(embedder)}")


async def _resolve_retriever_config(retriever: RetrieverConfigType, builder: Builder,
                                    rag_config: NvidiaRAGConfig) -> None:
    """Resolve retriever config and map fields to NvidiaRAGConfig.vector_store and retriever."""

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
        if retriever.top_k:
            rag_config.retriever.top_k = retriever.top_k
        return

    if isinstance(retriever, NemoRetrieverConfig):
        rag_config.vector_store.url = str(retriever.uri)
        if retriever.collection_name:
            rag_config.vector_store.default_collection_name = retriever.collection_name
        if retriever.nvidia_api_key:
            rag_config.vector_store.api_key = retriever.nvidia_api_key
        if retriever.top_k:
            rag_config.retriever.top_k = retriever.top_k
        return

    raise ValueError(f"Unsupported retriever config type: {type(retriever)}")
