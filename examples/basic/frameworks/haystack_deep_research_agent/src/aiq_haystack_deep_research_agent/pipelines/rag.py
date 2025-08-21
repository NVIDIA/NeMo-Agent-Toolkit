# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from haystack.components.builders import ChatPromptBuilder
from haystack.core.pipeline import Pipeline
from haystack.core.super_component import SuperComponent
from haystack.dataclasses import ChatMessage
from haystack.tools import ComponentTool
from haystack_integrations.components.generators.nvidia import NvidiaChatGenerator
from haystack_integrations.components.retrievers.opensearch import (
    OpenSearchBM25Retriever,
)


def create_rag_tool(
    document_store,
    rag_model: str,
    nvidia_api_url: str,
    secret_provider,
    *,
    top_k: int = 15,
) -> Tuple[ComponentTool, Pipeline]:
    retriever = OpenSearchBM25Retriever(document_store=document_store, top_k=top_k)
    generator = NvidiaChatGenerator(
        model=rag_model,
        api_base_url=nvidia_api_url,
        api_key=secret_provider,
    )

    template = """
	{% for document in documents %}
		{{ document.content }}
	{% endfor %}

	Please answer the question based on the given information.

	{{query}}
	"""
    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_user(template)], required_variables="*"
    )

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
