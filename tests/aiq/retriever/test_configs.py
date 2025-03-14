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

import pytest

from aiq.retriever.milvus.register import MilvusRetrieverConfig
from aiq.retriever.nemo_retriever.register import NemoRetrieverConfig


def test_milvus_config():
    # Create config with minimal parameters
    cfg = MilvusRetrieverConfig(uri="http://localhost:19530", embedding_model="nim_embedder")
    assert isinstance(cfg, MilvusRetrieverConfig)
    assert cfg == MilvusRetrieverConfig(
        uri="http://localhost:19530",
        embedding_model="nim_embedder",
        connection_args={},
        collection_name=None,
        content_field="text",
        output_fields=None,
        search_params={"metric_type": "L2"},
        vector_field="vector",
        description=None,
    )


def test_nemo_config():
    # Create config with minimal parameters
    cfg = NemoRetrieverConfig(uri="http://localhost:5000")
    assert isinstance(cfg, NemoRetrieverConfig)
    # Confirm that it's equivalent to the same config with defaults passed in
    assert cfg == NemoRetrieverConfig(uri="http://localhost:5000",
                                      collection_name=None,
                                      top_k=None,
                                      output_fields=None,
                                      timeout=60,
                                      nvidia_api_key=None)


@pytest.fixture(name="default_milvus_config", scope="module")
def get_default_milvus_config():
    return MilvusRetrieverConfig(uri="http://localhost:80", embedding_model="nim_embedder")


@pytest.fixture(name="default_nemo_retriever_config", scope="module")
def get_default_nemo_retriever_config():
    return NemoRetrieverConfig(uri="http://localhost:5000")


async def test_build_retrievers(default_milvus_config, default_nemo_retriever_config, httpserver):
    from aiq.retriever.milvus.retriever import MilvusRetriever
    from aiq.retriever.nemo_retriever.retriever import NemoRetriever

    class MockEmbedder:
        pass

    class MockMilvusClient:

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    nemo_retriever = NemoRetriever(**default_nemo_retriever_config.model_dump(
        exclude={"type", "top_k", "collection_name"}))
    optional_fields = ["collection_name", "top_k", "output_fields"]
    model_dict = default_nemo_retriever_config.model_dump()
    optional_args = {field: model_dict[field] for field in optional_fields if model_dict[field] is not None}

    nemo_retriever.bind(**optional_args)

    assert nemo_retriever.get_unbound_params() == ["query", "collection_name", "top_k"]

    nemo_retriever.bind(collection_name="my_collection", top_k=5)
    assert nemo_retriever.get_unbound_params() == ["query"]

    embedder = MockEmbedder()

    client = MockMilvusClient(uri=str(default_milvus_config.uri), **default_milvus_config.connection_args)
    milvus_retriever = MilvusRetriever(client=client,
                                       embedder=embedder,
                                       content_field=default_milvus_config.content_field)

    optional_fields = ["collection_name", "top_k", "output_fields", "search_params", "vector_field"]
    model_dict = default_milvus_config.model_dump()
    optional_args = {field: model_dict[field] for field in optional_fields if model_dict[field] is not None}

    milvus_retriever.bind(**optional_args)

    assert milvus_retriever.get_unbound_params() == ["query", "collection_name", "top_k", "filters"]

    milvus_retriever.bind(collection_name="my_collection", top_k=5)

    assert milvus_retriever.get_unbound_params() == ["query", "filters"]
