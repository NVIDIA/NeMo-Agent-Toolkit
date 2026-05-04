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
"""RAG (Retrieval-Augmented Generation) client plugin for the NeMo Agent Toolkit.

This module integrates NVIDIA's RAG pipeline into the toolkit function group system, exposing
search and generate tools that leverage LLMs, embedders, and retrievers for augmented document retrieval
and synthesis. It provides a configuration schema and workflow registration for seamless RAG support.
"""

import logging
from collections.abc import AsyncGenerator

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import RetrieverRef
from nat.data_models.function import FunctionGroupBaseConfig
from nat.plugins.rag.config import RAGPipelineConfig
from nat.plugins.rag.models import RAGSearchResult

logger: logging.Logger = logging.getLogger(__name__)


class NATRAGConfig(FunctionGroupBaseConfig, name="nat_rag"):
    """Configuration for NVIDIA RAG Library.

    Exposes search and generate tools that share a single RAG client.
    """
    llm: LLMRef = Field(description="LLM for response generation and query rewriting.")
    embedder: EmbedderRef = Field(description="Embedder for query and document vectorization.")
    retriever: RetrieverRef = Field(description="Vector store retriever for document search.")
    rag_pipeline: RAGPipelineConfig = Field(default_factory=RAGPipelineConfig,
                                            description="Advanced RAG pipeline settings.")
    topic: str | None = Field(default=None, description="Topic for tool descriptions.")
    collection_names: list[str] = Field(min_length=1, description="Collections to query.")
    reranker_top_k: int = Field(default=10, ge=1, description="Number of results after reranking.")


@register_function_group(config_type=NATRAGConfig)
async def nat_rag(config: NATRAGConfig, builder: Builder) -> AsyncGenerator[FunctionGroup, None]:
    """NVIDIA RAG Library - exposes search and generate tools."""
    from pydantic import SecretStr

    from nat.data_models.finetuning import OpenAIMessage
    from nat.embedder.nim_embedder import NIMEmbedderModelConfig
    from nat.llm.nim_llm import NIMModelConfig
    from nat.plugins.rag.models import RAGGenerateResult
    from nat.retriever.milvus.register import MilvusRetrieverConfig
    from nat.retriever.nemo_retriever.register import NemoRetrieverConfig
    try:
        from nvidia_rag.rag_server.main import NvidiaRAG
        from nvidia_rag.rag_server.response_generator import ChainResponse
        from nvidia_rag.rag_server.response_generator import Citations
        from nvidia_rag.utils.configuration import FilterExpressionGeneratorConfig
        from nvidia_rag.utils.configuration import NvidiaRAGConfig
        from nvidia_rag.utils.configuration import QueryDecompositionConfig
        from nvidia_rag.utils.configuration import QueryRewriterConfig
        from nvidia_rag.utils.configuration import ReflectionConfig
        from nvidia_rag.utils.configuration import VLMConfig
    except ImportError as e:
        raise ImportError("nvidia-rag package is not installed.") from e

    pipeline: RAGPipelineConfig = config.rag_pipeline

    rag_config: NvidiaRAGConfig = NvidiaRAGConfig(
        ranking=pipeline.ranking,
        retriever=pipeline.search_settings,
        vlm=pipeline.vlm or VLMConfig(),
        query_rewriter=pipeline.query_rewriter or QueryRewriterConfig(),
        filter_expression_generator=pipeline.filter_generator or FilterExpressionGeneratorConfig(),
        query_decomposition=pipeline.query_decomposition or QueryDecompositionConfig(),
        reflection=pipeline.reflection or ReflectionConfig(),
        enable_citations=pipeline.enable_citations,
        enable_guardrails=pipeline.enable_guardrails,
        enable_vlm_inference=pipeline.enable_vlm_inference,
        vlm_to_llm_fallback=pipeline.vlm_to_llm_fallback,
        default_confidence_threshold=pipeline.default_confidence_threshold,
    )

    # resolve LLM config
    nim_llm_config = builder.get_llm_config(config.llm)
    if not isinstance(nim_llm_config, NIMModelConfig):
        raise ValueError(f"Unsupported LLM config type: {type(config.llm)}. Expected NIMModelConfig.")

    base_dict = nim_llm_config.model_dump(include={"base_url", "model_name", "api_key"}, exclude_none=True)
    if "base_url" not in base_dict:
        raise ValueError("base_url is required for LLM config specified in NVIDIA RAG Config.")
    base_dict["server_url"] = base_dict.pop("base_url")

    rag_config.llm.parameters = rag_config.llm.parameters.model_copy(
        update=nim_llm_config.model_dump(include={"temperature", "top_p", "max_tokens"}, exclude_none=True))

    rag_config.llm = rag_config.llm.model_copy(update=base_dict)
    rag_config.reflection = rag_config.reflection.model_copy(update=base_dict)
    rag_config.filter_expression_generator = rag_config.filter_expression_generator.model_copy(update=base_dict)

    # resolve embedder config
    nim_embedder_config = builder.get_embedder_config(config.embedder)
    if not isinstance(nim_embedder_config, NIMEmbedderModelConfig):
        raise ValueError(f"Unsupported embedder config type: {type(config.embedder)}. Expected NIMEmbedderModelConfig.")
    base_dict = nim_embedder_config.model_dump(include={"base_url", "model_name", "api_key", "dimensions"},
                                               exclude_none=True)
    if "base_url" not in base_dict:
        raise ValueError("base_url is required for embedder config specified in NVIDIA RAG Config.")
    base_dict["server_url"] = base_dict.pop("base_url")
    rag_config.embeddings = rag_config.embeddings.model_copy(update=base_dict)

    # resolve retriever config
    retriever_config = await builder.get_retriever_config(config.retriever)
    match retriever_config:
        case MilvusRetrieverConfig():
            rag_config.vector_store.url = str(retriever_config.uri)
            if retriever_config.collection_name:
                rag_config.vector_store.default_collection_name = retriever_config.collection_name
            if retriever_config.connection_args:
                if "user" in retriever_config.connection_args:
                    rag_config.vector_store.username = retriever_config.connection_args["user"]
                if "password" in retriever_config.connection_args:
                    rag_config.vector_store.password = SecretStr(retriever_config.connection_args["password"])
            if retriever_config.top_k:
                rag_config.retriever.top_k = retriever_config.top_k
        case NemoRetrieverConfig():
            rag_config.vector_store.url = str(retriever_config.uri)
            if retriever_config.collection_name:
                rag_config.vector_store.default_collection_name = retriever_config.collection_name
            if retriever_config.nvidia_api_key:
                rag_config.vector_store.api_key = retriever_config.nvidia_api_key
            if retriever_config.top_k:
                rag_config.retriever.top_k = retriever_config.top_k
        case _:
            raise ValueError(f"Unsupported retriever config type: {type(retriever_config)}")

    rag_client: NvidiaRAG = NvidiaRAG(config=rag_config)
    logger.info("NVIDIA RAG client initialized")

    topic_str: str = f" about {config.topic}" if config.topic else ""

    async def search(query: str) -> RAGSearchResult:
        """Search for relevant documents."""
        try:
            citations: Citations = await rag_client.search(
                query=query,
                collection_names=config.collection_names,
                reranker_top_k=config.reranker_top_k,
            )
            return RAGSearchResult(citations=citations)
        except Exception:
            logger.exception("RAG search failed")
            raise

    # Server-Sent Events (SSE) format prefix for parsing streaming response chunks
    DATA_PREFIX = "data: "
    DATA_PREFIX_WIDTH = len(DATA_PREFIX)

    async def generate(query: str) -> RAGGenerateResult:
        """Generate an answer using the knowledge base."""
        chunks: list[str] = []
        final_citations: Citations | None = None
        try:
            stream = await rag_client.generate(
                messages=[OpenAIMessage(role="user", content=query).model_dump()],
                collection_names=config.collection_names,
                reranker_top_k=config.reranker_top_k,
            )
            async for raw_chunk in stream:
                if raw_chunk.startswith(DATA_PREFIX):
                    raw_chunk = raw_chunk[DATA_PREFIX_WIDTH:].strip()
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
                except (ValueError, TypeError, KeyError) as e:
                    logger.debug("Failed to parse RAG response chunk: %s - %s", type(e).__name__, e)
                    continue

            answer: str = "".join(chunks) if chunks else "No response generated."
            return RAGGenerateResult(answer=answer, citations=final_citations)

        except Exception:
            logger.exception("RAG generate failed")
            raise

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
