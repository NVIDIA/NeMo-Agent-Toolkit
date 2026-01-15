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
"""Configuration models and type aliases for NVIDIA RAG integration."""

from nvidia_rag.utils.configuration import EmbeddingConfig as NvidiaRAGEmbeddingConfig
from nvidia_rag.utils.configuration import FilterExpressionGeneratorConfig as NvidiaRAGFilterGeneratorConfig
from nvidia_rag.utils.configuration import LLMConfig as NvidiaRAGLLMConfig
from nvidia_rag.utils.configuration import QueryDecompositionConfig as NvidiaRAGQueryDecompositionConfig
from nvidia_rag.utils.configuration import QueryRewriterConfig as NvidiaRAGQueryRewriterConfig
from nvidia_rag.utils.configuration import RankingConfig as NvidiaRAGRankingConfig
from nvidia_rag.utils.configuration import ReflectionConfig as NvidiaRAGReflectionConfig
from nvidia_rag.utils.configuration import RetrieverConfig as NvidiaRAGRetrieverConfig
from nvidia_rag.utils.configuration import VectorStoreConfig as NvidiaRAGVectorStoreConfig
from nvidia_rag.utils.configuration import VLMConfig as NvidiaRAGVLMConfig
from pydantic import BaseModel
from pydantic import Field

from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import RetrieverRef
from nat.embedder.nim_embedder import NIMEmbedderModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.retriever.milvus.register import MilvusRetrieverConfig
from nat.retriever.nemo_retriever.register import NemoRetrieverConfig

# Type aliases for component configuration
LLMConfigType = NIMModelConfig | NvidiaRAGLLMConfig | LLMRef | None

EmbedderConfigType = NIMEmbedderModelConfig | NvidiaRAGEmbeddingConfig | EmbedderRef | None

RetrieverConfigType = MilvusRetrieverConfig | NemoRetrieverConfig | NvidiaRAGVectorStoreConfig | RetrieverRef | None


class RAGPipelineConfig(BaseModel):
    """Native nvidia_rag pipeline settings.

    Groups all RAG-specific settings that control search behavior,
    query preprocessing, and response quality.
    """

    # Search behavior
    search_settings: NvidiaRAGRetrieverConfig = Field(
        default_factory=NvidiaRAGRetrieverConfig)  # type: ignore[arg-type]
    ranking: NvidiaRAGRankingConfig = Field(default_factory=NvidiaRAGRankingConfig)  # type: ignore[arg-type]

    # Query preprocessing (optional)
    query_rewriter: NvidiaRAGQueryRewriterConfig | None = None
    filter_generator: NvidiaRAGFilterGeneratorConfig | None = None
    query_decomposition: NvidiaRAGQueryDecompositionConfig | None = None

    # Response quality (optional)
    reflection: NvidiaRAGReflectionConfig | None = None

    # Multimodal (optional)
    vlm: NvidiaRAGVLMConfig | None = None

    # Pipeline flags
    enable_citations: bool = True
    enable_guardrails: bool = False
    enable_vlm_inference: bool = False
    vlm_to_llm_fallback: bool = True
    default_confidence_threshold: float = 0.0
