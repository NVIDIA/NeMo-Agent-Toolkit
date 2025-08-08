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
from langchain_core.embeddings import Embeddings
from pytest_httpserver import HTTPServer

from nat.retriever.milvus.retriever import CollectionNotFoundError
from nat.retriever.milvus.retriever import MilvusRetriever
from nat.retriever.models import Document
from nat.retriever.models import RetrieverOutput
from nat.retriever.nemo_retriever.retriever import CollectionUnavailableError
from nat.retriever.nemo_retriever.retriever import NemoRetriever


class CustomMilvusClient:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def list_collections(self):
        return ["collection1", "collection2"]

    def describe_collection(self, collection_name: str):
        collection_descriptions = {
            "collection1": {
                "collection_name": "collection1",
                "fields": [
                    {
                        "name": "text"
                    },
                    {
                        "name": "author"
                    },
                    {
                        "name": "title"
                    },
                    {
                        "name": "vector"
                    },
                ]
            },
            "collection2": {
                "collection_name": "collection1",
                "fields": [
                    {
                        "name": "text"
                    },
                    {
                        "name": "author"
                    },
                    {
                        "name": "title"
                    },
                    {
                        "name": "vector"
                    },
                ]
            },
        }

        return collection_descriptions[collection_name]

    def _get_entity_from_fields(self, output_fields: list, num: int):
        sample_dict = {
            "text": f"Text chunk #{num}",
            "title": f"Doc Title: {num}",
            "author": f"Author: {num}",
        }

        return {k: v for k, v in sample_dict.items() if k in output_fields}

    def search(
        self,
        *,
        collection_name: str,
        data: list,
        limit: int,
        search_params: dict,
        filter: str | None,  # pylint: disable=W0622
        output_fields: list[str] | None,
        timeout: float | None,
        anns_field: str,
    ):
        assert isinstance(collection_name, str)
        assert isinstance(data, list)
        assert isinstance(limit, int)
        assert limit > 0
        if filter:
            assert isinstance(filter, str)
        if output_fields:
            assert isinstance(output_fields, list)
            assert len(output_fields) > 0
        if timeout:
            assert isinstance(timeout, (float, int))
        assert isinstance(search_params, dict)
        assert isinstance(anns_field, str)
        to_return = min(limit, 4)

        return [[
            {
                'id': '1234', 'distance': 0.45, 'entity': self._get_entity_from_fields(output_fields, num=1)
            },
            {
                'id': '5678', 'distance': 0.55, 'entity': self._get_entity_from_fields(output_fields, num=2)
            },
            {
                'id': '2468', 'distance': 0.70, 'entity': self._get_entity_from_fields(output_fields, num=3)
            },
            {
                'id': '1357', 'distance': 0.85, 'entity': self._get_entity_from_fields(output_fields, num=4)
            },
        ][:to_return]]

    def search_iterator(
        self,
        *,
        collection_name: str,
        data: list,
        limit: int,
        batch_size: int,
        filter: str | None,  # pylint: disable=W0622
        output_fields: list[str] | None,
        search_params: dict,
        timeout: float | None,
        anns_field: str,
        round_decimal: int,
        partition_names: str | None,
    ):
        assert isinstance(collection_name, str)
        assert isinstance(data, list)
        assert isinstance(limit, int)
        assert isinstance(search_params, dict)
        assert isinstance(anns_field, str)
        assert isinstance(batch_size, int)
        if filter:
            assert isinstance(filter, str)
        if output_fields:
            assert isinstance(output_fields, list)
        if timeout:
            assert isinstance(timeout, float)
        assert limit > 0


class TestEmbeddings(Embeddings):

    def embed_query(self, text):
        if not text or len(text) == 0:
            raise ValueError("No query passed to embedding model")
        return [0, 1, 2, 3, 4, 5]

    def embed_documents(self, texts):
        return [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]


@pytest.fixture(name="milvus_retriever", scope="module")
def _get_milvus_retriever():
    test_client = CustomMilvusClient()

    return MilvusRetriever(
        client=test_client,
        embedder=TestEmbeddings(),
    )


def _validate_document_milvus(doc: Document, output_fields=None):
    assert isinstance(doc, Document)
    assert doc.page_content.startswith("Text")
    if not output_fields:
        assert "title" in doc.metadata
        assert "author" in doc.metadata
    else:
        for field in output_fields:
            assert field in doc.metadata
    assert "distance" in doc.metadata
    assert doc.document_id is not None


async def test_milvus_search(milvus_retriever):

    assert isinstance(milvus_retriever, MilvusRetriever)

    # Test top_k results are returned
    res = await milvus_retriever.search(
        query="Test query?",
        collection_name="collection1",
        top_k=3,
    )
    assert isinstance(res, RetrieverOutput)
    assert len(res) == 3
    doc = res.results[0]
    _validate_document_milvus(doc)

    # Test all results are returned if higher top_k value used
    res = await milvus_retriever.search(
        query="Test query?",
        collection_name="collection1",
        top_k=6,
    )
    assert isinstance(res, RetrieverOutput)
    assert len(res) == 4
    doc = res.results[0]
    _validate_document_milvus(doc)

    # Test output fields
    res = await milvus_retriever.search(query="Test query?",
                                        collection_name="collection2",
                                        top_k=2,
                                        output_fields=["title"])
    assert isinstance(res, RetrieverOutput)
    assert len(res) == 2
    doc = res.results[0]
    _validate_document_milvus(doc, ["title"])


async def test_milvus_retriever_binding(milvus_retriever):

    # Test invalid collection name
    with pytest.raises(CollectionNotFoundError):
        _ = await milvus_retriever.search(query="Test query", collection_name="collection_not_exist", top_k=4)

    milvus_retriever.bind(top_k=2)
    _ = await milvus_retriever.search(query="Test query", collection_name="collection2")

    # Test not supplying enough parameters
    with pytest.raises(TypeError):
        _ = await milvus_retriever.search(query="Test query no collection name")

    # Test that binding those parameters makes the same call work
    milvus_retriever.bind(top_k=2, collection_name="collection1")
    _ = await milvus_retriever.search(query="Test query")


async def test_milvus_validation(milvus_retriever):

    # Test validation for the vector field not being in the schema
    with pytest.raises(ValueError):
        _ = await milvus_retriever.search(query="Test query",
                                          collection_name="collection1",
                                          vector_field_name="v",
                                          top_k=2)

    # Test validation for the content field not being in the schema
    milvus_retriever.content_field = "c"
    with pytest.raises(ValueError):
        _ = await milvus_retriever.search(query="Test query", collection_name="collection1", top_k=2)


@pytest.fixture(name="nemo_retriever")
def get_nemo_retriever(httpserver: HTTPServer):
    httpserver.expect_request(
        "/v1/collections",
        method="GET",
    ).respond_with_json({
        "collections": [
            {
                'created_at': '2024-07-06T21:45:46.452826',
                'id': '92e2c5e6',
                'meta': 'null',
                'name': 'test_collection_1',
                'pipeline': 'hybrid'
            },
            {
                'created_at': '2024-07-06T21:45:46.452826',
                'id': '92e2c5e7',
                'meta': 'null',
                'name': 'test_collection_2',
                'pipeline': 'hybrid'
            },
        ]
    })

    httpserver.expect_request(
        "/v1/collections/92e2c5e6/search",
        method="POST",
    ).respond_with_json({
        "chunks": [
            {
                "content": "Text Chunk - 1",
                "format": "txt",
                "id": "bde719d3ae5c47e",
                "metadata": {
                    "title": "Title 1",
                    "author": "Author 1",
                },
                "score": 2.45425234
            },
            {
                "content": "Text Chunk - 2",
                "format": "txt",
                "id": "d3ae5c47ebde719",
                "metadata": {
                    "title": "Title 2",
                    "author": "Author 2",
                },
                "score": 1.42523445
            },
        ]
    })

    httpserver.expect_request(
        "/v1/collections/92e2c5e7/search",
        method="POST",
    ).respond_with_json({
        "chunks": [
            {
                "content": "Text Chunk - 3",
                "format": "txt",
                "id": "bde719d3ae5c47e",
                "metadata": {
                    "title": "Title 3",
                    "author": "Author 3",
                },
                "score": 1.45425234
            },
            {
                "content": "Text Chunk - 4",
                "format": "txt",
                "id": "d3ae5c47ebde719",
                "metadata": {
                    "title": "Title 4",
                    "author": "Author 4",
                },
                "score": 2.42523445
            },
        ]
    })

    return NemoRetriever(uri=httpserver.url_for("/"))


async def test_nemo_retriever_search(nemo_retriever):

    res = await nemo_retriever.search("Test query", collection_name="test_collection_1", top_k=2)
    assert isinstance(res, RetrieverOutput)
    assert len(res) == 2

    with pytest.raises(CollectionUnavailableError):
        _ = await nemo_retriever.search("Test query", collection_name="collection_not_exist", top_k=2)

    # Test output fields
    res = await nemo_retriever.search("Test query",
                                      collection_name="test_collection_1",
                                      top_k=2,
                                      output_fields=["title"])
    assert isinstance(res, RetrieverOutput)
    assert "title" in res.results[0].metadata
    assert "author" not in res.results[0].metadata

    res = await nemo_retriever.search("Test query",
                                      collection_name="test_collection_1",
                                      top_k=2,
                                      output_fields=["author"])
    assert isinstance(res, RetrieverOutput)
    assert "title" not in res.results[0].metadata
    assert "author" in res.results[0].metadata


async def test_nemo_binding(nemo_retriever):

    nemo_retriever.bind(top_k=2)
    _ = await nemo_retriever.search("Test query", collection_name="test_collection_2")

    with pytest.raises(TypeError):
        _ = await nemo_retriever.search("Test query")

    nemo_retriever.bind(top_k=2, collection_name="test_collection_1")
    _ = await nemo_retriever.search("Test query")
