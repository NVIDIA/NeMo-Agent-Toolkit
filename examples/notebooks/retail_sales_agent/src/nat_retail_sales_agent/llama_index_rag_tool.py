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

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LlamaIndexRAGConfig(FunctionBaseConfig, name="local_llama_index_rag"):

    llm_name: LLMRef = Field(description="The name of the LLM to use for the RAG engine.")
    embedder_name: EmbedderRef = Field(description="The name of the embedder to use for the RAG engine.")
    data_dir: str = Field(description="The directory containing the data to use for the RAG engine.")
    description: str = Field(description="A description of the knowledge included in the RAG system.")
    uri: str = Field(default="http://localhost:19530", description="The URI of the Milvus vector store.")
    use_milvus: bool = Field(default=False, description="Whether to use Milvus for the RAG engine.")
    collection_name: str = Field(default="context", description="The name of the collection to use for the RAG engine.")


@register_function(config_type=LlamaIndexRAGConfig, framework_wrappers=[LLMFrameworkEnum.LLAMA_INDEX])
async def llama_index_rag_tool(config: LlamaIndexRAGConfig, builder: Builder):
    from llama_index.core import Settings
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core import StorageContext
    from llama_index.core import VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.milvus import MilvusVectorStore
    from pymilvus.exceptions import MilvusException

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
    embedder = await builder.get_embedder(config.embedder_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)

    Settings.embed_model = embedder
    Settings.llm = llm

    docs = SimpleDirectoryReader(input_files=[config.data_dir]).load_data()
    logger.info("Loaded %s documents from %s", len(docs), config.data_dir)

    parser = SentenceSplitter(
        chunk_size=400,
        chunk_overlap=20,
        separator=" ",
    )
    nodes = parser.get_nodes_from_documents(docs)

    if config.use_milvus:
        try:
            vector_store = MilvusVectorStore(
                uri=config.uri,
                collection_name=config.collection_name,
                overwrite=True,
                dim=1024,
                enable_sparse=False,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex(nodes, storage_context=storage_context)

        except MilvusException as e:
            logger.error("Error initializing Milvus vector store: %s. Falling back to default vector store.", e)
            index = VectorStoreIndex(nodes)
    else:
        index = VectorStoreIndex(nodes)

    query_engine = index.as_query_engine(similarity_top_k=3, )

    async def _arun(inputs: str) -> str:
        """
        Search product catalog for information about tablets, laptops, and smartphones
        Args:
            inputs: user query about product specifications
        """
        try:
            response = query_engine.query(inputs)
            return str(response.response)

        except Exception as e:
            logger.error("RAG query failed: %s", e)
            return f"Sorry, I couldn't retrieve information about that product. Error: {str(e)}"

    yield FunctionInfo.from_fn(_arun, description=config.description)
