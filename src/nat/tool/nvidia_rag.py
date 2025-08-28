# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class NVIDIARAGToolConfig(FunctionBaseConfig, name="nvidia_rag"):
    """
    Tool used to search the NVIDIA Developer database for documents across a variety of NVIDIA asset types.
    """
    base_url: str = Field(description="The base url to the RAG service.")
    timeout: int = Field(default=60, description="The timeout configuration to use when sending requests.")
    document_separator: str = Field(default="\n\n", description="The delimiter to use between retrieved documents.")
    document_prompt: str = Field(default=("-------\n\n" + "Title: {document_title}\n"
                                          "Text: {page_content}\nSource URL: {document_url}"),
                                 description="The prompt to use to retrieve documents from the RAG service")
    top_k: int = Field(default=4, description="The number of results to return from the RAG service.")
    collection_name: str = Field(default="nvidia_api_catalog",
                                 description=("The name of the collection to use when retrieving documents."))


@register_function(config_type=NVIDIARAGToolConfig)
async def nvidia_rag_tool(config: NVIDIARAGToolConfig, builder: Builder):
    import httpx
    from langchain.prompts import PromptTemplate
    from langchain_core.documents import Document
    from langchain_core.prompts import aformat_document

    document_prompt = PromptTemplate.from_template(config.document_prompt)

    async with httpx.AsyncClient(headers={
            "accept": "application/json", "Content-Type": "application/json"
    },
                                 timeout=config.timeout) as client:

        async def runnable(query: str) -> str:

            try:
                url = f"{config.base_url}/search"

                payload = {"query": query, "top_k": config.top_k, "collection_name": config.collection_name}

                logger.debug("Sending request to the RAG endpoint %s.", url)
                response = await client.post(url, content=json.dumps(payload))

                response.raise_for_status()

                output = response.json()

                docs = [
                    Document(
                        page_content=ret["content"],
                        metadata={
                            "document_title": ret["filename"],
                            "document_url": "nemo_framework",
                            "document_full_text": ret["content"],
                            "score_rerank": ret["score"]
                        },
                        type="Document",
                    ) for ret in output["chunks"]
                ]

                parsed_output = config.document_separator.join(
                    [await aformat_document(doc, document_prompt) for doc in docs])
                return parsed_output
            except Exception as e:
                logger.exception("Error while running the tool")
                return f"Error while running the tool: {e}"

        yield FunctionInfo.from_fn(
            runnable,
            description=("Search the NVIDIA Developer database for documents across a variety of "
                         "NVIDIA asset types"))
