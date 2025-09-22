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

from pydantic import Field
from pydantic import HttpUrl

from nat.builder.builder import Builder
from nat.builder.builder import LLMFrameworkEnum
from nat.builder.retriever import RetrieverProviderInfo
from nat.cli.register_workflow import register_retriever_client
from nat.cli.register_workflow import register_retriever_provider
from nat.data_models.retriever import RetrieverBaseConfig


class MilvusRetrieverConfig(RetrieverBaseConfig, name="milvus_retriever"):
    """
    Configuration for a Retriever which pulls data from a Milvus service.
    """
    uri: HttpUrl = Field(description="The uri of Milvus service")
    connection_args: dict = Field(
        description="Dictionary of arguments used to connect to and authenticate with the Milvus service",
        default={},
    )
    embedding_model: str = Field(description="The name of the embedding model to use for vectorizing the query")
    collection_name: str | None = Field(description="The name of the milvus collection to search", default=None)
    content_field: str = Field(description="Name of the primary field to store/retrieve",
                               default="text",
                               alias="primary_field")
    top_k: int | None = Field(gt=0, description="The number of results to return", default=None)
    output_fields: list[str] | None = Field(
        default=None,
        description="A list of fields to return from the datastore. If 'None', all fields but the vector are returned.")
    search_params: dict = Field(default={"metric_type": "L2"},
                                description="Search parameters to use when performing vector search")
    vector_field: str = Field(default="vector", description="Name of the field to compare with the vectorized query")
    description: str | None = Field(default=None,
                                    description="If present it will be used as the tool description",
                                    alias="collection_description")


@register_retriever_provider(config_type=MilvusRetrieverConfig)
async def milvus_retriever(retriever_config: MilvusRetrieverConfig, builder: Builder):
    yield RetrieverProviderInfo(config=retriever_config,
                                description="An adapter for a Miluvs data store to use with a Retriever Client")


@register_retriever_client(config_type=MilvusRetrieverConfig, wrapper_type=None)
async def milvus_retriever_client(config: MilvusRetrieverConfig, builder: Builder):
    from pymilvus import MilvusClient

    from nat.retriever.milvus.retriever import MilvusRetriever

    embedder = await builder.get_embedder(embedder_name=config.embedding_model, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    milvus_client = MilvusClient(uri=str(config.uri), **config.connection_args)
    retriever = MilvusRetriever(
        client=milvus_client,
        embedder=embedder,
        content_field=config.content_field,
    )

    # Using parameters in the config to set default values which can be overridden during the function call.
    optional_fields = ["collection_name", "top_k", "output_fields", "search_params", "vector_field"]
    model_dict = config.model_dump()
    optional_args = {field: model_dict[field] for field in optional_fields if model_dict[field] is not None}

    retriever.bind(**optional_args)

    yield retriever
