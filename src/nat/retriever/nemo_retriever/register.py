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

from aiq.builder.builder import Builder
from aiq.builder.retriever import RetrieverProviderInfo
from aiq.cli.register_workflow import register_retriever_client
from aiq.cli.register_workflow import register_retriever_provider
from aiq.data_models.retriever import RetrieverBaseConfig


class NemoRetrieverConfig(RetrieverBaseConfig, name="nemo_retriever"):
    """
    Configuration for a Retriever which pulls data from a Nemo Retriever service.
    """
    uri: HttpUrl = Field(description="The uri of the Nemo Retriever service.")
    collection_name: str | None = Field(description="The name of the collection to search", default=None)
    top_k: int | None = Field(description="The number of results to return", gt=0, le=50, default=None)
    output_fields: list[str] | None = Field(
        default=None,
        description="A list of fields to return from the datastore. If 'None', all fields but the vector are returned.")
    timeout: int = Field(default=60, description="Maximum time to wait for results to be returned from the service.")
    nvidia_api_key: str | None = Field(
        description="API key used to authenticate with the service. If 'None', will use ENV Variable 'NVIDIA_API_KEY'",
        default=None,
    )


@register_retriever_provider(config_type=NemoRetrieverConfig)
async def nemo_retriever(retriever_config: NemoRetrieverConfig, builder: Builder):
    yield RetrieverProviderInfo(config=retriever_config,
                                description="An adapter for a  Nemo data store for use with a Retriever Client")


@register_retriever_client(config_type=NemoRetrieverConfig, wrapper_type=None)
async def nemo_retriever_client(config: NemoRetrieverConfig, builder: Builder):
    from aiq.retriever.nemo_retriever.retriever import NemoRetriever

    retriever = NemoRetriever(**config.model_dump(exclude={"type", "top_k", "collection_name"}))
    optional_fields = ["collection_name", "top_k", "output_fields"]
    model_dict = config.model_dump()
    optional_args = {field: model_dict[field] for field in optional_fields if model_dict[field] is not None}

    retriever.bind(**optional_args)

    yield retriever
