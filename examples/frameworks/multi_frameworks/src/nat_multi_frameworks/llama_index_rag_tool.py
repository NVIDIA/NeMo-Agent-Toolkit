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

import logging

from pydantic import ConfigDict

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.common import OptionalSecretStr
from nat.data_models.common import set_secret_from_env
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LlamaIndexRAGConfig(FunctionBaseConfig, name="llama_index_rag"):

    model_config = ConfigDict(protected_namespaces=())

    llm_name: LLMRef
    embedding_name: EmbedderRef
    data_dir: str
    api_key: OptionalSecretStr = None
    model_name: str


@register_function(config_type=LlamaIndexRAGConfig, framework_wrappers=[LLMFrameworkEnum.LLAMA_INDEX])
async def llama_index_rag_tool(tool_config: LlamaIndexRAGConfig, builder: Builder):

    from colorama import Fore
    from llama_index.core import Settings
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core import VectorStoreIndex
    from llama_index.core.node_parser import SimpleFileNodeParser

    if (not tool_config.api_key):
        set_secret_from_env(tool_config, "api_key", "NVIDIA_API_KEY")

    if not tool_config.api_key:
        raise ValueError(
            "API token must be provided in the configuration or in the environment variable `NVIDIA_API_KEY`")

    logger.info("##### processing data from ingesting files in this folder : %s", tool_config.data_dir)

    llm = await builder.get_llm(tool_config.llm_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
    embedder = await builder.get_embedder(tool_config.embedding_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)

    Settings.embed_model = embedder
    Settings.llm = llm

    md_docs = SimpleDirectoryReader(input_files=[tool_config.data_dir]).load_data()
    parser = SimpleFileNodeParser()
    nodes = parser.get_nodes_from_documents(md_docs)
    index = VectorStoreIndex(nodes)
    query_engine = index.as_query_engine(similarity_top_k=2)

    async def _arun(inputs: str) -> str:
        """
        rag using llama-index ingesting README markdown file
        Args:
            inputs : user query
        """
        try:
            logger.info("Querying llama-index RAG with input: %s", inputs)
            response = await query_engine.aquery(inputs)

            if response is None:
                logger.warning("Query engine returned None for input: %s", inputs)
                return ""

            # Extract the response text
            response_text = str(response.response) if hasattr(response, 'response') else str(response)
            logger.info("Response from llama-index RAG: %s%s", Fore.MAGENTA, response_text)
            return response_text

        except Exception as e:
            logger.error("Error running llama-index RAG: %s", str(e), exc_info=True)
            return f"Error processing query: {str(e)}"

    yield FunctionInfo.from_fn(_arun, description="extract relevant data via llama-index's RAG per user input query")
