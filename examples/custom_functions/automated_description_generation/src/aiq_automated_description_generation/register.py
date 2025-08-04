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

import logging

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function import Function
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.component_ref import RetrieverRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.retriever.models import RetrieverOutput

logger = logging.getLogger(__name__)


class AutomatedDescriptionMilvusWorkflow(FunctionBaseConfig, name="automated_description_milvus"):
    """
    Workflow which generates a description for a Milvus Collection by analyzing a subset of its contents.
    """
    llm_name: LLMRef = Field(description="LLM to use for summarizing documents and generating a description.")
    retriever_name: RetrieverRef = Field(description="Name of the retriever to use for fetching documents.")
    retrieval_tool_name: FunctionRef = Field(description="Name of the retrieval tool to use for fetching documents.")
    collection_name: str = Field(description="Name of the vector DB collection to generate a description for.")

    num_samples: int = Field(default=15, description="Number of documents to analyze for generating a description.")
    max_token: int = Field(default=100000, description="The maximum number of cumulative tokens for a single document.")
    batch_size: int = Field(default=5, description="Number of documents to process in a single LLM call")
    vector_field: str = Field(default="vector", description="Field holding the embeddings in the collection.")


# We want this to load a retriever, then generate a description for a Milvus collection.
# Then on invoke, return the result of the retriever invocation with the description set.


@register_function(config_type=AutomatedDescriptionMilvusWorkflow, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def automated_description_milvus_workflow(workflow_config: AutomatedDescriptionMilvusWorkflow, builder: Builder):
    from aiq_automated_description_generation.utils.description_generation import generate_description
    from aiq_automated_description_generation.utils.prompts import direct_summary_prompt
    from aiq_automated_description_generation.utils.prompts import map_prompt
    from aiq_automated_description_generation.utils.prompts import reduce_prompt
    from aiq_automated_description_generation.utils.workflow_utils import SummarizationWorkflow

    logger.info("Building necessary components for the Automated Description Generation Workflow")
    llm_n = await builder.get_llm(llm_name=workflow_config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Description generation needs a Langchain retriever
    vs_retriever = await builder.get_retriever(retriever_name=workflow_config.retriever_name,
                                               wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    # Get the retriever tool
    retriever_tool: Function = builder.get_function(workflow_config.retrieval_tool_name)

    vectorstore = vs_retriever.vectorstore

    logger.info("Components built, starting the Automated Description Generation Workflow")
    summarization_workflow = SummarizationWorkflow(llm=llm_n,
                                                   direct_summary_prompt=direct_summary_prompt,
                                                   map_prompt=map_prompt,
                                                   reduce_prompt=reduce_prompt,
                                                   max_token=workflow_config.max_token,
                                                   batch_size=workflow_config.batch_size)

    dynamic_description = await generate_description(workflow_config.collection_name,
                                                     workflow_config.num_samples,
                                                     workflow_config.vector_field,
                                                     vectorstore,
                                                     summarization_workflow)

    function_desc = f"Ask questions about the following collection of text: {dynamic_description}"
    logger.info("Generated the dynamic description: %s", function_desc)

    async def _entrypoint(query: str) -> RetrieverOutput:
        return await retriever_tool.acall_invoke(query)

    yield FunctionInfo.from_fn(_entrypoint, description=function_desc)
