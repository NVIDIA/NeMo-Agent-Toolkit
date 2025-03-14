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

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_retriever_client
from aiq.retriever.milvus.register import MilvusRetrieverConfig
from aiq.retriever.nemo_retriever.register import NemoRetrieverConfig


@register_retriever_client(config_type=NemoRetrieverConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def nemo_langchain(retriever_config: NemoRetrieverConfig, builder: Builder):
    from aiq.retriever.nemo_retriever.retriever import NemoLangchainRetriever
    from aiq.retriever.nemo_retriever.retriever import NemoRetriever

    retriever = NemoRetriever(**retriever_config.model_dump(exclude={"type", "top_k", "collection_name"}))
    optional_fields = ["collection_name", "top_k", "output_fields"]
    model_dict = retriever_config.model_dump()
    optional_args = {field: model_dict[field] for field in optional_fields if model_dict[field] is not None}

    retriever.bind(**optional_args)

    yield NemoLangchainRetriever(client=retriever)


@register_retriever_client(config_type=MilvusRetrieverConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def milvus_langchain(retriever_config: MilvusRetrieverConfig, builder: Builder):
    from langchain_milvus import Milvus

    retriever_config.connection_args.update({"uri": str(retriever_config.uri)})
    embedder = await builder.get_embedder(embedder_name=retriever_config.embedding_model,
                                          wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    yield Milvus(embedding_function=embedder,
                 **retriever_config.model_dump(include={
                     "connection_args",
                     "collection_name",
                     "content_field",
                     "vector_field",
                     "search_params",
                     "description"
                 },
                                               by_alias=True)).as_retriever()
