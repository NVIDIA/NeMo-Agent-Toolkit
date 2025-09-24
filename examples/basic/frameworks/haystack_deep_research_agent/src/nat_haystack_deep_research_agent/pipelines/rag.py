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

from haystack.components.builders import ChatPromptBuilder
from haystack.core.pipeline import Pipeline
from haystack.core.super_component import SuperComponent
from haystack.dataclasses import ChatMessage
from haystack.tools import ComponentTool
from haystack_integrations.components.generators.nvidia import NvidiaChatGenerator
from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever


def create_rag_tool(
    document_store,
    *,
    top_k: int = 15,
    generator: NvidiaChatGenerator | None = None,
) -> tuple[ComponentTool, Pipeline]:
    """
    Build a RAG tool composed of OpenSearch retriever and NvidiaChatGenerator.

    Args:
        document_store: OpenSearch document store instance.
        top_k: Number of documents to retrieve for RAG.
        generator: Pre-configured NvidiaChatGenerator created from builder LLM config.

    Returns:
        (ComponentTool, Pipeline): The tool and underlying pipeline.

    Raises:
        ValueError: If a generator is not provided.
    """
    retriever = OpenSearchBM25Retriever(document_store=document_store, top_k=top_k)
    if generator is None:
        raise ValueError("NvidiaChatGenerator instance must be provided via builder-configured LLM.")

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

    rag_component = SuperComponent(
        pipeline=rag_pipeline,
        input_mapping={"query": ["retriever.query", "prompt_builder.query"]},
        output_mapping={"llm.replies": "rag_result"},
    )

    rag_tool = ComponentTool(
        name="rag",
        description="Use this tool to search in our internal database of documents.",
        component=rag_component,
        outputs_to_string={"source": "rag_result"},
    )

    return rag_tool, rag_pipeline
