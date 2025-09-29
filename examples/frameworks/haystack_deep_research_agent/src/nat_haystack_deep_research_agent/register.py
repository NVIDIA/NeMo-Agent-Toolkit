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

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.llm.nim_llm import NIMModelConfig

logger = logging.getLogger(__name__)


class HaystackDeepResearchWorkflowConfig(FunctionBaseConfig, name="haystack_deep_research_agent"):  # type: ignore
    system_prompt: str = """
    You are a deep research assistant.
    You create comprehensive research reports to answer the user's questions.
    You use the 'search' tool to answer any questions by using web search.
    You use the 'rag' tool to answer any questions by using retrieval augmented generation on your internal document DB.
    You perform multiple searches until you have the information you need to answer the question.
    Make sure you research different aspects of the question.
    Use markdown to format your response.
    When you use information from the websearch results, cite your sources using markdown links.
    When you use information from the document database, cite the text used from the source document.
    It is important that you cite accurately.
    """
    max_agent_steps: int = 20
    search_top_k: int = 10
    rag_top_k: int = 15
    opensearch_url: str = "http://localhost:9200"
    # Indexing configuration
    index_on_startup: bool = True
    # Default to "/data" so users can mount a volume or place files at repo_root/data.
    # If it doesn't exist, we fall back to this example's bundled data folder.
    data_dir: str = "/data"


@register_function(config_type=HaystackDeepResearchWorkflowConfig)
async def haystack_deep_research_agent_workflow(config: HaystackDeepResearchWorkflowConfig, builder: Builder):
    """
    Main workflow that creates and returns the deep research agent.

    Uses top-level `llms` configuration via builder to instantiate Haystack NvidiaChatGenerator
    for both the agent and RAG tool, per review suggestions.
    """
    from haystack.components.agents import Agent
    from haystack.dataclasses import ChatMessage
    from haystack.tools import Toolset
    from haystack_integrations.components.generators.nvidia import NvidiaChatGenerator
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

    from nat_haystack_deep_research_agent import create_rag_tool
    from nat_haystack_deep_research_agent import create_search_tool
    from nat_haystack_deep_research_agent import run_startup_indexing

    logger.info(f"Starting Haystack Deep Research Agent workflow with config: {config}")

    # Create search tool
    search_tool = create_search_tool(top_k=config.search_top_k)

    # Create document store
    document_store = OpenSearchDocumentStore(hosts=[config.opensearch_url], index="deep_research_docs")
    logger.info("Connected to OpenSearch successfully")

    # Optionally index local data at startup
    if config.index_on_startup:
        run_startup_indexing(document_store=document_store, data_dir=config.data_dir, logger=logger)

    def _nim_to_haystack_generator(cfg: NIMModelConfig) -> NvidiaChatGenerator:
        return NvidiaChatGenerator(model=cfg.model_name)

    # Instantiate LLMs via builder configs (expecting NIM)
    rag_llm_cfg = builder.get_llm_config("rag_llm")
    agent_llm_cfg = builder.get_llm_config("agent_llm")

    if not isinstance(rag_llm_cfg, NIMModelConfig):
        raise TypeError("llms.rag_llm must be of type 'nim'.")
    if not isinstance(agent_llm_cfg, NIMModelConfig):
        raise TypeError("llms.agent_llm must be of type 'nim'.")

    rag_generator = _nim_to_haystack_generator(rag_llm_cfg)
    rag_tool, _ = create_rag_tool(
        document_store=document_store,
        top_k=config.rag_top_k,
        generator=rag_generator,
    )

    # Create the agent
    agent_generator = _nim_to_haystack_generator(agent_llm_cfg)

    agent = Agent(
        chat_generator=agent_generator,
        tools=Toolset(tools=[search_tool, rag_tool]),
        system_prompt=config.system_prompt,
        exit_conditions=["text"],
        max_agent_steps=config.max_agent_steps,
    )

    # Warm up the agent
    agent.warm_up()
    logger.info("Agent warmed up successfully")

    async def _response_fn(input_message: str) -> str:
        """
        Process the input message and generate a research response.

        Args:
            input_message: The user's research question

        Returns:
            Comprehensive research report.
        """
        try:
            logger.info(f"Processing research query: {input_message}")

            # Create messages
            messages = [ChatMessage.from_user(input_message)]
            agent_output = agent.run(messages=messages)

            # Extract response
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
