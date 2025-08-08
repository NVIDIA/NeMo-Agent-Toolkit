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

from pydantic import ConfigDict

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LlamaIndexRAGConfig(FunctionBaseConfig, name="llama_index_rag"):

    model_config = ConfigDict(protected_namespaces=())

    llm_name: LLMRef
    embedding_name: EmbedderRef
    data_dir: str
    api_key: str | None = None
    model_name: str


@register_function(config_type=LlamaIndexRAGConfig, framework_wrappers=[LLMFrameworkEnum.LLAMA_INDEX])
async def llama_index_rag_tool(tool_config: LlamaIndexRAGConfig, builder: Builder):

    from colorama import Fore
    from llama_index.core import Settings
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core import VectorStoreIndex
    from llama_index.core.agent import FunctionCallingAgentWorker
    from llama_index.core.node_parser import SimpleFileNodeParser
    from llama_index.core.tools import QueryEngineTool

    if (not tool_config.api_key):
        tool_config.api_key = os.getenv("NVIDIA_API_KEY")

    if not tool_config.api_key:
        raise ValueError(
            "API token must be provided in the configuration or in the environment variable `NVIDIA_API_KEY`")

    logger.info("##### processing data from ingesting files in this folder : %s", tool_config.data_dir)

    llm = await builder.get_llm(tool_config.llm_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
    embedder = await builder.get_embedder(tool_config.embedding_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)

    Settings.embed_model = embedder
    md_docs = SimpleDirectoryReader(input_files=[tool_config.data_dir]).load_data()
    parser = SimpleFileNodeParser()
    nodes = parser.get_nodes_from_documents(md_docs)
    index = VectorStoreIndex(nodes)
    Settings.llm = llm
    query_engine = index.as_query_engine(similarity_top_k=2)

    model_name = tool_config.model_name
    if not model_name.startswith('nvdev'):
        tool = QueryEngineTool.from_defaults(
            query_engine, name="rag", description="ingest data from README about this workflow with llama_index_rag")

        agent_worker = FunctionCallingAgentWorker.from_tools(
            [tool],
            llm=llm,
            verbose=True,
        )
        agent = agent_worker.as_agent()

    async def _arun(inputs: str) -> str:
        """
        rag using llama-index ingesting README markdown file
        Args:
            query : user query
        """
        if not model_name.startswith('nvdev'):
            agent_response = (await agent.achat(inputs))
            logger.info("response from llama-index Agent : \n %s %s", Fore.MAGENTA, agent_response.response)
            output = agent_response.response
        else:
            logger.info("%s %s %s %s", Fore.MAGENTA, type(query_engine), query_engine, inputs)
            output = query_engine.query(inputs).response

        return output

    yield FunctionInfo.from_fn(_arun, description="extract relevant data via llama-index's RAG per user input query")
