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
"""Configuration models and type aliases for NVIDIA RAG integration."""

from pydantic import BaseModel
from pydantic import Field

from nvidia_rag.utils.configuration import RetrieverConfig as NvidiaRAGRetrieverConfig
from nvidia_rag.utils.configuration import FilterExpressionGeneratorConfig as NvidiaRAGFilterGeneratorConfig
from nvidia_rag.utils.configuration import QueryDecompositionConfig as NvidiaRAGQueryDecompositionConfig
from nvidia_rag.utils.configuration import QueryRewriterConfig as NvidiaRAGQueryRewriterConfig
from nvidia_rag.utils.configuration import RankingConfig as NvidiaRAGRankingConfig
from nvidia_rag.utils.configuration import ReflectionConfig as NvidiaRAGReflectionConfig
from nvidia_rag.utils.configuration import VLMConfig as NvidiaRAGVLMConfig


class RAGPipelineConfig(BaseModel):
    """Native nvidia_rag pipeline settings.

    Groups all RAG-specific settings that control search behavior,
    query preprocessing, and response quality.
    """

    # Search behavior
    search_settings: NvidiaRAGRetrieverConfig = Field(default_factory=lambda: NvidiaRAGRetrieverConfig())
    ranking: NvidiaRAGRankingConfig = Field(default_factory=lambda: NvidiaRAGRankingConfig())

    # Query preprocessing (optional)
    query_rewriter: NvidiaRAGQueryRewriterConfig | None = Field(default=None)
    filter_generator: NvidiaRAGFilterGeneratorConfig | None = Field(default=None)
    query_decomposition: NvidiaRAGQueryDecompositionConfig | None = Field(default=None)

    # Response quality (optional)
    reflection: NvidiaRAGReflectionConfig | None = Field(default=None)

    # Multimodal (optional)
    vlm: NvidiaRAGVLMConfig | None = Field(default=None)

    # Pipeline flags
    enable_citations: bool = Field(default=True)
    enable_guardrails: bool = Field(default=False)
    enable_vlm_inference: bool = Field(default=False)
    vlm_to_llm_fallback: bool = Field(default=True)
    default_confidence_threshold: float = Field(default=0.0)
