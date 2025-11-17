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

import inspect
import logging
from functools import partial
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings
from pymilvus.client.abstract import Hit

if TYPE_CHECKING:
    from pymilvus import AsyncMilvusClient
    from pymilvus import MilvusClient

from nat.retriever.interface import Retriever
from nat.retriever.models import Document
from nat.retriever.models import RetrieverError
from nat.retriever.models import RetrieverOutput

logger = logging.getLogger(__name__)


class CollectionNotFoundError(RetrieverError):
    pass


class MilvusRetriever(Retriever):
    """
    Client for retrieving document chunks from a Milvus vectorstore
    """

    def __init__(
        self,
        client: "MilvusClient | AsyncMilvusClient",
        embedder: Embeddings,
        content_field: str = "text",
        use_iterator: bool = False,
    ) -> None:
        """
        Initialize the Milvus Retriever using a preconfigured MilvusClient or AsyncMilvusClient

        Args:
        """
        self._client: MilvusClient | AsyncMilvusClient = client
        self._embedder = embedder

        # Detect if client is async by inspecting method capabilities
        search_method = getattr(client, "search", None)
        list_collections_method = getattr(client, "list_collections", None)
        self._is_async = any(
            inspect.iscoroutinefunction(method) for method in (search_method, list_collections_method)
            if method is not None)
        logger.info("Initialized Milvus Retriever with %s client", "async" if self._is_async else "sync")

        if use_iterator and "search_iterator" not in dir(self._client):
            raise ValueError("This version of the pymilvus.MilvusClient does not support the search iterator.")

        self._search_func = self._search if not use_iterator else self._search_with_iterator
        self._default_params = None
        self._bound_params = []
        self.content_field = content_field
        logger.info("Milvus Retriever using %s for search.", self._search_func.__name__)

    def bind(self, **kwargs) -> None:
        """
        Bind default values to the search method. Cannot bind the 'query' parameter.

        Args:
          kwargs (dict): Key value pairs corresponding to the default values of search parameters.
        """
        if "query" in kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k != "query"}
        self._search_func = partial(self._search_func, **kwargs)
        self._bound_params = list(kwargs.keys())
        logger.debug("Binding paramaters for search function: %s", kwargs)

    def get_unbound_params(self) -> list[str]:
        """
        Returns a list of unbound parameters which will need to be passed to the search function.
        """
        return [param for param in ["query", "collection_name", "top_k", "filters"] if param not in self._bound_params]

    async def _validate_collection(self, collection_name: str) -> bool:
        """Validate that a collection exists."""
        if self._is_async:
            collections = await self._client.list_collections()
        else:
            collections = self._client.list_collections()
        return collection_name in collections

    async def search(self, query: str, **kwargs):
        return await self._search_func(query=query, **kwargs)

    async def _search_with_iterator(self,
                                    query: str,
                                    *,
                                    collection_name: str,
                                    top_k: int,
                                    filters: str | None = None,
                                    output_fields: list[str] | None = None,
                                    search_params: dict | None = None,
                                    timeout: float | None = None,
                                    vector_field_name: str | None = "vector",
                                    distance_cutoff: float | None = None,
                                    **kwargs):
        """
        Retrieve document chunks from a Milvus vectorstore using a search iterator, allowing for the retrieval of more
        results.
        """
        logger.debug("MilvusRetriever searching query: %s, for collection: %s. Returning max %s results",
                     query,
                     collection_name,
                     top_k)

        if not await self._validate_collection(collection_name):
            raise CollectionNotFoundError(f"Collection: {collection_name} does not exist")

        # If no output fields are specified, return all of them
        if not output_fields:
            if self._is_async:
                collection_schema = await self._client.describe_collection(collection_name)
            else:
                collection_schema = self._client.describe_collection(collection_name)
            output_fields = [
                field["name"] for field in collection_schema.get("fields") if field["name"] != vector_field_name
            ]

        search_vector = await self._embedder.aembed_query(query)

        # Create search iterator
        if self._is_async:
            search_iterator = await self._client.search_iterator(
                collection_name=collection_name,
                data=[search_vector],
                batch_size=kwargs.get("batch_size", 1000),
                filter=filters,
                limit=top_k,
                output_fields=output_fields,
                search_params=search_params if search_params else {"metric_type": "L2"},
                timeout=timeout,
                anns_field=vector_field_name,
                round_decimal=kwargs.get("round_decimal", -1),
                partition_names=kwargs.get("partition_names", None),
            )
        else:
            search_iterator = self._client.search_iterator(
                collection_name=collection_name,
                data=[search_vector],
                batch_size=kwargs.get("batch_size", 1000),
                filter=filters,
                limit=top_k,
                output_fields=output_fields,
                search_params=search_params if search_params else {"metric_type": "L2"},
                timeout=timeout,
                anns_field=vector_field_name,
                round_decimal=kwargs.get("round_decimal", -1),
                partition_names=kwargs.get("partition_names", None),
            )

        results = []
        try:
            while True:
                if self._is_async:
                    _res = await search_iterator.next()
                else:
                    _res = search_iterator.next()
                res = _res.get_res()
                if len(_res) == 0:
                    if self._is_async:
                        await search_iterator.close()
                    else:
                        search_iterator.close()
                    break

                if distance_cutoff and res[0][-1].distance > distance_cutoff:
                    for i in range(len(res[0])):
                        if res[0][i].distance > distance_cutoff:
                            break
                        results.append(res[0][i])
                    break
                results.extend(res[0])

                return _wrap_milvus_results(results, content_field=self.content_field)

        except Exception as e:
            logger.error("Exception when retrieving results from milvus for query %s: %s", query, e)
            raise RetrieverError(f"Error when retrieving documents from {collection_name} for query '{query}'") from e

    async def _search(self,
                      query: str,
                      *,
                      collection_name: str,
                      top_k: int,
                      filters: str | None = None,
                      output_fields: list[str] | None = None,
                      search_params: dict | None = None,
                      timeout: float | None = None,
                      vector_field_name: str | None = "vector",
                      **kwargs):
        """
        Retrieve document chunks from a Milvus vectorstore
        """
        logger.debug("MilvusRetriever searching query: %s, for collection: %s. Returning max %s results",
                     query,
                     collection_name,
                     top_k)

        if not await self._validate_collection(collection_name):
            raise CollectionNotFoundError(f"Collection: {collection_name} does not exist")

        # Get collection schema
        if self._is_async:
            collection_schema = await self._client.describe_collection(collection_name)
        else:
            collection_schema = self._client.describe_collection(collection_name)

        available_fields = [v.get("name") for v in collection_schema.get("fields", [])]

        if self.content_field not in available_fields:
            raise ValueError(f"The specified content field: {self.content_field} is not part of the schema.")

        if vector_field_name not in available_fields:
            raise ValueError(f"The specified vector field name: {vector_field_name} is not part of the schema.")

        # If no output fields are specified, return all of them
        if not output_fields:
            output_fields = [field for field in available_fields if field != vector_field_name]

        if self.content_field not in output_fields:
            output_fields.append(self.content_field)

        search_vector = await self._embedder.aembed_query(query)

        # Perform search
        if self._is_async:
            res = await self._client.search(
                collection_name=collection_name,
                data=[search_vector],
                filter=filters,
                output_fields=output_fields,
                search_params=search_params if search_params else {"metric_type": "L2"},
                timeout=timeout,
                anns_field=vector_field_name,
                limit=top_k,
            )
        else:
            res = self._client.search(
                collection_name=collection_name,
                data=[search_vector],
                filter=filters,
                output_fields=output_fields,
                search_params=search_params if search_params else {"metric_type": "L2"},
                timeout=timeout,
                anns_field=vector_field_name,
                limit=top_k,
            )

        return _wrap_milvus_results(res[0], content_field=self.content_field)


def _wrap_milvus_results(res: list[Hit], content_field: str):
    return RetrieverOutput(results=[_wrap_milvus_single_results(r, content_field=content_field) for r in res])


def _wrap_milvus_single_results(res: Hit | dict, content_field: str) -> Document:
    if not isinstance(res, Hit | dict):
        raise ValueError(f"Milvus search returned object of type {type(res)}. Expected 'Hit' or 'dict'.")

    if isinstance(res, Hit):
        metadata = {k: v for k, v in res.fields.items() if k != content_field}
        metadata.update({"distance": res.distance})
        return Document(page_content=res.fields[content_field], metadata=metadata, document_id=res.id)

    fields = res["entity"]
    metadata = {k: v for k, v in fields.items() if k != content_field}
    metadata.update({"distance": res.get("distance")})
    return Document(page_content=fields.get(content_field), metadata=metadata, document_id=res["id"])
