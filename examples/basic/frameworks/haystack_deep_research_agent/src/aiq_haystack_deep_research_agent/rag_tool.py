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
import os
from pathlib import Path

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class HaystackRAGConfig(FunctionBaseConfig, name="haystack_rag_tool"):
    document_store_host: str = "http://localhost:9200"
    index_name: str = "deep_research_docs"
    top_k: int = 15
    data_directory: str = "./examples/basic/frameworks/haystack_deep_research_agent/data"


@register_function(config_type=HaystackRAGConfig)
async def haystack_rag_tool(tool_config: HaystackRAGConfig, builder: Builder):
    """
    RAG tool using Haystack's OpenSearchDocumentStore with indexing and query pipelines
    Returns a SuperComponent that can be used with ComponentTool
    """
    from haystack import Pipeline
    from haystack.components.builders import ChatPromptBuilder
    from haystack.components.converters import PyPDFToDocument
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
    from haystack.components.writers import DocumentWriter
    from haystack.core.super_component import SuperComponent
    from haystack.dataclasses import ChatMessage
    from haystack.document_stores.types import DuplicatePolicy
    from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

    logger.info(f"Creating RAG tool with config: {tool_config}")

    # Initialize document store
    try:
        document_store = OpenSearchDocumentStore(
            hosts=[tool_config.document_store_host],
            index=tool_config.index_name
        )
        logger.info(f"Connected to OpenSearch at {tool_config.document_store_host}")
    except Exception as e:
        logger.warning(f"Failed to connect to OpenSearch: {e}. RAG functionality will be limited.")
        document_store = None

    # Create indexing pipeline
    indexing_pipeline = None
    if document_store:
        converter = PyPDFToDocument()
        cleaner = DocumentCleaner()
        splitter = DocumentSplitter(split_by="sentence", split_length=10, split_overlap=2)
        writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)

        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("converter", converter)
        indexing_pipeline.add_component("cleaner", cleaner)
        indexing_pipeline.add_component("splitter", splitter)
        indexing_pipeline.add_component("writer", writer)

        indexing_pipeline.connect("converter", "cleaner")
        indexing_pipeline.connect("cleaner", "splitter")
        indexing_pipeline.connect("splitter", "writer")

        # Index documents from data directory
        data_path = Path(tool_config.data_directory)
        if data_path.exists():
            pdf_files = list(data_path.glob("*.pdf"))
            txt_files = list(data_path.glob("*.txt"))
            all_files = pdf_files + txt_files

            if all_files:
                logger.info(f"Indexing {len(all_files)} files from {data_path}")
                try:
                    if pdf_files:
                        indexing_pipeline.run({"converter": {"sources": pdf_files}})
                    # For txt files, we need a different approach - let's add them manually
                    if txt_files:
                        from haystack import Document
                        txt_docs = []
                        for txt_file in txt_files:
                            with open(txt_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                                doc = Document(content=content, meta={"file_path": str(txt_file)})
                                txt_docs.append(doc)
                        # Process txt documents through the pipeline
                        splitter_output = splitter.run(documents=txt_docs)
                        writer.run(documents=splitter_output["documents"])

                    logger.info("Document indexing completed successfully")
                except Exception as e:
                    logger.error(f"Failed to index documents: {e}")
            else:
                logger.info(f"No PDF or TXT files found in {data_path}")

    # Create query pipeline exactly as in the notebook
    rag_component = None
    if document_store:
        retriever = OpenSearchBM25Retriever(document_store=document_store, top_k=tool_config.top_k)
        generator = OpenAIChatGenerator(model="gpt-4o-mini")

        # Use ChatPromptBuilder exactly as in the notebook
        template = """
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Please answer the question based on the given information.

{{query}}
"""
        prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(template)], required_variables="*")

        rag_pipeline = Pipeline()
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("llm", generator)

        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")

        # Create SuperComponent exactly as in the notebook
        rag_component = SuperComponent(
            pipeline=rag_pipeline,
            input_mapping={"query": ["retriever.query"]},
            output_mapping={"llm.replies": "rag_result"},
        )

    async def _arun(query: str) -> str:
        """
        Search internal document database using RAG

        Args:
            query: The search query

        Returns:
            Generated answer based on retrieved documents
        """
        if not rag_component:
            return "RAG functionality is not available. Please ensure OpenSearch is running and accessible."

        try:
            logger.info(f"Performing RAG query: {query}")
            result = rag_component.run({"query": query})

            if "rag_result" in result and result["rag_result"]:
                answer = result["rag_result"][0].content
                logger.info(f"RAG query completed successfully for: {query}")
                return answer
            else:
                logger.warning(f"No response generated for RAG query: {query}")
                return "No relevant information found in the document database."

        except Exception as e:
            logger.error(f"RAG query failed for '{query}': {str(e)}")
            return f"RAG search failed: {str(e)}"

    # Store the component for access by the workflow
    _arun.rag_component = rag_component

    yield FunctionInfo.from_fn(
        _arun,
        description="Use this tool to search in your internal database of documents with Retrieval Augmented Generation (RAG)."
    )