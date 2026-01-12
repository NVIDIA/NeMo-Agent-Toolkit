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

"""NVIDIA RAG query and search functions for NeMo Agent Toolkit."""

import json
import logging
from typing import Optional

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class NvidiaRAGQueryConfig(FunctionBaseConfig, name="nvidia_rag_query"):
    """
    Tool that queries documents using NVIDIA RAG.
    Sends a chat-style query to the RAG system using configured models and endpoints.
    """

    config_file: str = Field(
        default="config.yaml",
        description="Path to the NVIDIA RAG configuration YAML file.",
    )
    collection_names: list[str] = Field(
        default_factory=list,
        description="List of collection names to query from.",
    )
    vdb_endpoint: str = Field(
        default="http://localhost:19530",
        description="Vector database endpoint URL.",
    )
    embedding_endpoint: Optional[str] = Field(
        default=None,
        description="Embedding endpoint URL. If None, uses cloud embeddings.",
    )
    use_knowledge_base: bool = Field(
        default=True,
        description="Whether to use the knowledge base for RAG.",
    )


@register_function(config_type=NvidiaRAGQueryConfig)
async def nvidia_rag_query(config: NvidiaRAGQueryConfig, builder: Builder):
    """Register the NVIDIA RAG query tool."""

    from nvidia_rag import NvidiaRAG
    from nvidia_rag.utils.configuration import NvidiaRAGConfig

    # Initialize the RAG client
    rag_config = NvidiaRAGConfig.from_yaml(config.config_file)
    rag = NvidiaRAG(config=rag_config)

    async def _nvidia_rag_query(query: str) -> str:
        """Query documents using NVIDIA RAG and return a generated response.

        This tool sends a query to the RAG system which retrieves relevant documents
        from the vector database and uses an LLM to generate a response.

        Args:
            query: The question or query to ask the RAG system.

        Returns:
            The generated response from the RAG system based on retrieved documents.
        """
        try:
            response = await rag.generate(
                messages=[{"role": "user", "content": query}],
                use_knowledge_base=config.use_knowledge_base,
                collection_names=config.collection_names,
                embedding_endpoint=config.embedding_endpoint,
            )

            if response.status_code != 200:
                return f"Error: RAG query failed with status code {response.status_code}"

            # Extract the response content from the streaming generator
            full_response = []
            async for chunk in response.generator:
                if chunk.startswith("data: "):
                    chunk = chunk[len("data: "):].strip()
                if not chunk:
                    continue
                try:
                    data = json.loads(chunk)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        text = delta.get("content")
                        if not text:
                            message = choices[0].get("message", {})
                            text = message.get("content", "")
                        if text:
                            full_response.append(text)
                except json.JSONDecodeError:
                    continue

            return "".join(full_response) if full_response else "No response generated."

        except Exception as e:
            logger.error("RAG query failed: %s", str(e))
            return f"Error querying RAG: {str(e)}"

    yield FunctionInfo.from_fn(
        _nvidia_rag_query,
        description=_nvidia_rag_query.__doc__,
    )


class NvidiaRAGSearchConfig(FunctionBaseConfig, name="nvidia_rag_search"):
    """
    Tool that searches for relevant documents in the vector database using NVIDIA RAG.
    """

    config_file: str = Field(
        default="config.yaml",
        description="Path to the NVIDIA RAG configuration YAML file.",
    )
    collection_names: list[str] = Field(
        default_factory=list,
        description="List of collection names to search in.",
    )
    vdb_endpoint: str = Field(
        default="http://localhost:19530",
        description="Vector database endpoint URL.",
    )
    embedding_endpoint: Optional[str] = Field(
        default=None,
        description="Embedding endpoint URL. If None, uses cloud embeddings.",
    )
    reranker_top_k: int = Field(
        default=10,
        description="Number of top results to return after reranking.",
    )
    vdb_top_k: int = Field(
        default=100,
        description="Number of top results to retrieve from vector database before reranking.",
    )


@register_function(config_type=NvidiaRAGSearchConfig)
async def nvidia_rag_search(config: NvidiaRAGSearchConfig, builder: Builder):
    """Register the NVIDIA RAG search tool."""

    from nvidia_rag import NvidiaRAG
    from nvidia_rag.utils.configuration import NvidiaRAGConfig

    # Initialize the RAG client
    rag_config = NvidiaRAGConfig.from_yaml(config.config_file)
    rag = NvidiaRAG(config=rag_config)

    async def _nvidia_rag_search(query: str) -> str:
        """Search for relevant documents in the vector database.

        This tool performs a semantic search in the vector database collections
        and returns relevant document chunks.

        Args:
            query: The search query to find relevant documents.

        Returns:
            A formatted string containing the search results with document names and content.
        """
        try:
            citations = await rag.search(
                query=query,
                collection_names=config.collection_names,
                reranker_top_k=config.reranker_top_k,
                vdb_top_k=config.vdb_top_k,
                embedding_endpoint=config.embedding_endpoint,
            )

            if not citations or not hasattr(citations, "results") or not citations.results:
                return "No documents found for the given query."

            # Format the results
            results = []
            for idx, citation in enumerate(citations.results):
                doc_name = getattr(citation, "document_name", f"Document {idx + 1}")
                content = getattr(citation, "content", "")
                doc_type = getattr(citation, "document_type", "text")
                description = getattr(citation, "metadata", {}).get("description", "")

                results.append(f"**{doc_name}** (type: {doc_type}):\n{description}")

            return "\n\n---\n\n".join(results)

        except Exception as e:
            logger.error("RAG search failed: %s", str(e))
            return f"Error searching documents: {str(e)}"

    yield FunctionInfo.from_fn(
        _nvidia_rag_search,
        description=_nvidia_rag_search.__doc__,
    )
