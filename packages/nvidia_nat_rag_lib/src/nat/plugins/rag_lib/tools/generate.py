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
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.finetuning import OpenAIMessage
from nat.data_models.function import FunctionBaseConfig
from nat.plugins.rag_lib.models import RAGGenerateResult

if TYPE_CHECKING:
    from nvidia_rag.rag_server.main import NvidiaRAG
    from nvidia_rag.rag_server.response_generator import RAGResponse

    from nat.plugins.rag_lib.client import NvidiaRAGFunctionGroup

logger: logging.Logger = logging.getLogger(__name__)


class RAGLibGenerateConfig(FunctionBaseConfig, name="rag_lib_generate"):
    """Generate tool configuration."""

    rag_client: FunctionGroupRef = Field(description="Reference to nvidia_rag_lib function group.")
    topic: str | None = Field(default=None, description="Topic injected into description via {topic} placeholder.")
    description: str = Field(default="Generate an answer{topic} with citations from the knowledge base.",
                             description="Tool description. Use {topic} placeholder to include topic.")
    collection_names: list[str] | None = Field(default=None,
                                               description="Collections for context. None uses client defaults.")
    use_knowledge_base: bool = Field(default=True, description="Whether to use RAG (True) or pure LLM (False).")
    confidence_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum relevance score for context. None uses client default.")
    enable_citations: bool | None = Field(default=None,
                                          description="Whether to include citations. None uses client default.")
    enable_guardrails: bool | None = Field(default=None,
                                           description="Whether to enable guardrails. None uses client default.")
    temperature: float | None = Field(default=None,
                                      ge=0.0,
                                      le=2.0,
                                      description="Sampling temperature. None uses client default.")
    max_tokens: int | None = Field(default=None,
                                   ge=1,
                                   description="Maximum tokens to generate. None uses client default.")
    top_p: float | None = Field(default=None,
                                ge=0.0,
                                le=1.0,
                                description="Nucleus sampling probability. None uses client default.")
    filter_expr: str | None = Field(
        default=None, description="Static metadata filter expression, e.g., 'year >= 2023'. None applies no filter.")


@register_function(config_type=RAGLibGenerateConfig)
async def rag_lib_generate(config: RAGLibGenerateConfig, builder: Builder):
    """RAG Library Generate Tool."""

    rag_group: NvidiaRAGFunctionGroup = await builder.get_function_group(config.rag_client)  # type: ignore[assignment]
    rag_client: NvidiaRAG = rag_group.rag_client
    topic_str: str = f" about {config.topic}" if config.topic else ""
    description: str = config.description.format(topic=topic_str)

    async def _generate(query: str) -> RAGGenerateResult:
        """Generate an answer using the knowledge base."""
        from nvidia_rag.rag_server.response_generator import ChainResponse
        from nvidia_rag.rag_server.response_generator import Citations

        chunks: list[str] = []
        final_citations: Citations | None = None

        try:
            response: RAGResponse = await rag_client.generate(
                messages=[OpenAIMessage(role="user", content=query).model_dump()],
                use_knowledge_base=config.use_knowledge_base,
                filter_expr=config.filter_expr or "",
                collection_names=config.collection_names,
                confidence_threshold=config.confidence_threshold,
                enable_citations=config.enable_citations,
                enable_guardrails=config.enable_guardrails,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
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

    yield FunctionInfo.from_fn(fn=_generate, description=description)
