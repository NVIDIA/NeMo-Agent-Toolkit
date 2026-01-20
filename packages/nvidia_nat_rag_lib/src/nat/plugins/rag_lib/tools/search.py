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

import logging
from typing import TYPE_CHECKING

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.function import FunctionBaseConfig
from nat.plugins.rag_lib.models import RAGSearchResult

if TYPE_CHECKING:
    from nat.plugins.rag_lib.client import NvidiaRAGFunctionGroup

logger: logging.Logger = logging.getLogger(__name__)


class RAGLibSearchConfig(FunctionBaseConfig, name="rag_lib_search"):
    """Search tool configuration."""

    rag_client: FunctionGroupRef = Field(description="Reference to nvidia_rag_lib function group.")
    topic: str | None = Field(default=None, description="Topic injected into description via {topic} placeholder.")
    description: str = Field(
        default="Retrieve document chunks{topic} which can be used to answer the provided question.",
        description="Tool description. Use {topic} placeholder to include topic.")
    collection_names: list[str] | None = Field(default=None, description="Collections to search.")
    reranker_top_k: int | None = Field(default=None, ge=1, description="Number of results after reranking.")
    vdb_top_k: int | None = Field(
        default=None,
        ge=1,
        description="Number of candidates from vector DB before reranking. None uses client default.")
    confidence_threshold: float | None = Field(default=None,
                                               ge=0.0,
                                               le=1.0,
                                               description="Minimum relevance score. None uses client default.")
    enable_query_rewriting: bool | None = Field(default=None,
                                                description="Whether to rewrite queries. None uses client default.")
    enable_reranker: bool | None = Field(default=None,
                                         description="Whether to use reranking. None uses client default.")
    enable_filter_generator: bool | None = Field(
        default=None, description="Whether to auto-generate filters. None uses client default.")
    filter_expr: str | None = Field(
        default=None, description="Static metadata filter expression, e.g., 'year >= 2023'. None applies no filter.")


@register_function(config_type=RAGLibSearchConfig)
async def rag_lib_search(config: RAGLibSearchConfig, builder: Builder):
    """RAG Library Search Tool."""

    rag_group: NvidiaRAGFunctionGroup = await builder.get_function_group(config.rag_client)  # type: ignore[assignment]
    rag_client = rag_group.rag_client
    topic_str = f" related to {config.topic}" if config.topic else ""
    description = config.description.format(topic=topic_str)

    async def _search(query: str) -> RAGSearchResult:
        """Search for relevant documents."""
        from nvidia_rag.rag_server.response_generator import Citations

        try:
            citations: Citations = await rag_client.search(
                query=query,
                filter_expr=config.filter_expr or "",
                collection_names=config.collection_names,
                reranker_top_k=config.reranker_top_k,
                vdb_top_k=config.vdb_top_k,
                confidence_threshold=config.confidence_threshold,
                enable_query_rewriting=config.enable_query_rewriting,
                enable_reranker=config.enable_reranker,
                enable_filter_generator=config.enable_filter_generator,
            )
            logger.info("Search returned %d results for query: %s", citations.total_results, query[:50])
            return RAGSearchResult(citations=citations)
        except Exception:
            logger.exception("RAG search failed")
            return RAGSearchResult(citations=Citations(total_results=0, results=[]))

    yield FunctionInfo.from_fn(fn=_search, description=description)
