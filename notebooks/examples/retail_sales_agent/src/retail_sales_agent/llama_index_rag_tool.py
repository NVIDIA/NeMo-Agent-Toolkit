# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import EmbedderRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LlamaIndexRAGConfig(FunctionBaseConfig, name="local_llama_index_rag"):

    llm_name: LLMRef = Field(description="The name of the LLM to use for the RAG engine.")
    embedder_name: EmbedderRef = Field(description="The name of the embedder to use for the RAG engine.")
    data_dir: str = Field(description="The directory containing the data to use for the RAG engine.")
    description: str = Field(description="A description of the knowledge included in the RAG system.")


@register_function(config_type=LlamaIndexRAGConfig, framework_wrappers=[LLMFrameworkEnum.LLAMA_INDEX])
async def llama_index_rag_tool(config: LlamaIndexRAGConfig, builder: Builder):
    from llama_index.core import Settings
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core import VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
    embedder = await builder.get_embedder(config.embedder_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)

    Settings.embed_model = embedder
    Settings.llm = llm

    docs = SimpleDirectoryReader(input_files=[config.data_dir]).load_data()
    logger.info(f"Loaded {len(docs)} documents from {config.data_dir}")
    if docs:
        logger.info(f"First document content preview: {docs[0].text[:200]}...")

    # Use SentenceSplitter with small chunks to stay under 512 token embedding limit
    parser = SentenceSplitter(
        chunk_size=400,  # Smaller chunks for agent environment
        chunk_overlap=20,  # Smaller overlap to reduce redundancy
        separator=" ",
    )
    nodes = parser.get_nodes_from_documents(docs)

    index = VectorStoreIndex(nodes)
    query_engine = index.as_query_engine(
        similarity_top_k=3,  # Use only 1 chunk to minimize context usage
        response_mode="compact"  # Use compact response mode
    )

    async def _arun(inputs: str) -> str:
        """
        Search product catalog for information about tablets, laptops, and smartphones
        Args:
            inputs: user query about product specifications
        """
        try:
            response = query_engine.query(inputs)
            return str(response.response)  # Convert to string to avoid serialization issues
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return f"Sorry, I couldn't retrieve information about that product. Error: {str(e)}"

    yield FunctionInfo.from_fn(
        _arun,
        description="Search product catalog for TabZen tablet, AeroBook laptop, NovaPhone specifications"
    )