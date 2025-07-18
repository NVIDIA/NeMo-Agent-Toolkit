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

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class HaystackDeepResearchWorkflowConfig(FunctionBaseConfig, name="haystack_deep_research_agent"):
    llm: LLMRef
    system_prompt: str = """
    You are a deep research assistant.
    You create comprehensive research reports to answer the user's questions.
    You use the 'search' tool to answer any questions by using web search.
    You use the 'rag' tool to answer any questions by using retrieval augmented generation on your internal document database.
    You perform multiple searches until you have the information you need to answer the question.
    Make sure you research different aspects of the question.
    Use markdown to format your response.
    When you use information from the websearch results, cite your sources using markdown links.
    When you use information from the document database, cite the text used from the source document.
    It is important that you cite accurately.
    """
    max_agent_steps: int = 20


@register_function(config_type=HaystackDeepResearchWorkflowConfig)
async def haystack_deep_research_agent_workflow(config: HaystackDeepResearchWorkflowConfig, builder: Builder):
    """
    Main workflow that creates and returns the deep research agent
    Implements the exact structure from the Blueprint_Deep_Research_Agent.ipynb notebook
    """
    from haystack.components.agents import Agent
    from haystack.components.builders import ChatPromptBuilder
    from haystack.components.converters.html import HTMLToDocument
    from haystack.components.converters.output_adapter import OutputAdapter
    from haystack.components.fetchers.link_content import LinkContentFetcher
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.websearch.serper_dev import SerperDevWebSearch
    from haystack.core.pipeline import Pipeline
    from haystack.core.super_component import SuperComponent
    from haystack.dataclasses import ChatMessage
    from haystack.tools import ComponentTool
    from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

    logger.info(f"Starting Haystack Deep Research Agent workflow with config: {config}")

    # Create search pipeline exactly as in the notebook
    search_pipeline = Pipeline()
    search_pipeline.add_component("search", SerperDevWebSearch(top_k=10))
    search_pipeline.add_component("fetcher", LinkContentFetcher(timeout=3, raise_on_failure=False, retry_attempts=2))
    search_pipeline.add_component("converter", HTMLToDocument())
    search_pipeline.add_component(
        "output_adapter",
        OutputAdapter(
            template="""
{%- for doc in docs -%}
  {%- if doc.content -%}
  <search-result url="{{ doc.meta.url }}">
  {{ doc.content|truncate(25000) }}
  </search-result>
  {%- endif -%}
{%- endfor -%}
""",
            output_type=str,
        ),
    )
    search_pipeline.connect("search.links", "fetcher.urls")
    search_pipeline.connect("fetcher.streams", "converter.sources")
    search_pipeline.connect("converter.documents", "output_adapter.docs")

    # Create SuperComponent for search exactly as in the notebook
    search_component = SuperComponent(
        pipeline=search_pipeline,
        input_mapping={"query": ["search.query"]},
        output_mapping={"output_adapter.output": "search_result"},
    )

    # Create search tool exactly as in the notebook
    search_tool = ComponentTool(
        name="search",
        description="Use this tool to search for information on the internet.",
        component=search_component,
        outputs_to_string={"source": "search_result"},
    )

    # Create RAG pipeline exactly as in the notebook
    try:
        document_store = OpenSearchDocumentStore(
            hosts=["http://localhost:9200"],
            index="deep_research_docs"
        )
        logger.info("Connected to OpenSearch successfully")

        retriever = OpenSearchBM25Retriever(document_store=document_store, top_k=15)
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

        # Create SuperComponent for RAG exactly as in the notebook
        rag_component = SuperComponent(
            pipeline=rag_pipeline,
            input_mapping={"query": ["retriever.query", "prompt_builder.query"]},
            output_mapping={"llm.replies": "rag_result"},
        )

        # Create RAG tool exactly as in the notebook
        rag_tool = ComponentTool(
            name="rag",
            description="Use this tool to search in your internal database of documents with Retrieval Augmented Generation (RAG).",
            component=rag_component,
            outputs_to_string={"source": "rag_result"},
        )

        tools = [search_tool, rag_tool]

    except Exception as e:
        logger.warning(f"Failed to connect to OpenSearch: {e}. Using search tool only.")
        tools = [search_tool]

    # Create the agent exactly as in the notebook
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
        tools=tools,
        system_prompt=config.system_prompt,
        exit_conditions=["text"],
        max_agent_steps=config.max_agent_steps,
    )

    # Warm up the agent
    try:
        agent.warm_up()
        logger.info("Agent warmed up successfully")
    except Exception as e:
        logger.warning(f"Agent warm up failed: {e}")

    async def _response_fn(input_message: str) -> str:
        """
        Process the input message and generate a research response
        Implements the exact logic from the notebook

        Args:
            input_message: The user's research question

        Returns:
            Comprehensive research report
        """
        try:
            logger.info(f"Processing research query: {input_message}")

            # Create messages exactly as in the notebook
            messages = [ChatMessage.from_user(input_message)]
            agent_output = agent.run(messages=messages)

            # Extract response exactly as in the notebook
            if "messages" in agent_output and agent_output["messages"]:
                response = agent_output["messages"][-1].text
                logger.info("Research query completed successfully")
                return response
            else:
                logger.warning(f"No response generated for query: {input_message}")
                return "I apologize, but I was unable to generate a response for your query."

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return f"I apologize, but an error occurred during research: {str(e)}"

    try:
        yield _response_fn
    except GeneratorExit:
        logger.exception("Workflow exited early!", exc_info=True)
    finally:
        logger.info("Cleaning up Haystack Deep Research Agent workflow.")