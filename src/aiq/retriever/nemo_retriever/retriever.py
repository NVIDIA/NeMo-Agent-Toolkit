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

import json
import logging
import os
import typing
from functools import partial
from urllib.parse import urljoin

import httpx
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl

from aiq.retriever.interface import AIQRetriever
from aiq.retriever.models import AIQDocument
from aiq.retriever.models import RetrieverError
from aiq.retriever.models import RetrieverOutput

logger = logging.getLogger(__name__)


class Collection(BaseModel):
    id: str
    name: str
    meta: typing.Any
    pipeline: str
    created_at: str


class RetrieverPayload(BaseModel):
    query: str
    top_k: int = Field(le=50, gt=0)


class CollectionUnavailableError(RetrieverError):
    pass


class NemoRetriever(AIQRetriever):
    """
    Client for retrieving document chunks from a Nemo Retriever service.
    """

    def __init__(self, uri: str | HttpUrl, timeout: int = 60, nvidia_api_key: str = None, **kwargs):

        self.base_url = str(uri)
        self.timeout = timeout
        self._search_func = self._search
        self.api_key = nvidia_api_key if nvidia_api_key else os.getenv('NVIDIA_API_KEY')
        self._bound_params = []
        if not self.api_key:
            logger.warning("No API key was specified as part of configuration or as an environment variable.")

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
        return [param for param in ["query", "collection_name", "top_k"] if param not in self._bound_params]

    async def get_collections(self, client) -> list[Collection]:
        """
        Get a list of all available collections as pydantic `Collection` objects
        """
        collection_response = await client.get(urljoin(self.base_url, "/v1/collections"))
        collection_response.raise_for_status()
        if not collection_response or len(collection_response.json().get('collections', [])) == 0:
            raise CollectionUnavailableError(f"No collections available at {self.base_url}")

        collections = [
            Collection.model_validate(collection) for collection in collection_response.json()["collections"]
        ]

        return collections

    async def get_collection_by_name(self, collection_name, client) -> Collection:
        """
        Retrieve a collection using it's name. Will return the first collection found if the name is ambiguous.
        """
        collections = await self.get_collections(client)
        if (collection := next((c for c in collections if c.name == collection_name), None)) is None:
            raise CollectionUnavailableError(f"Collection {collection_name} not found")
        return collection

    async def search(self, query: str, **kwargs):
        return await self._search_func(query=query, **kwargs)

    async def _search(
        self,
        query: str,
        collection_name: str,
        top_k: str,
        output_fields: list[str] = None,
    ):
        """
        Retrieve document chunks from the configured Nemo Retriever Service.
        """
        output = []
        try:
            async with httpx.AsyncClient(headers={"Authorization": f"Bearer {self.api_key}"},
                                         timeout=self.timeout) as client:
                collection = await self.get_collection_by_name(collection_name, client)
                url = urljoin(self.base_url, f"/v1/collections/{collection.id}/search")

                payload = RetrieverPayload(query=query, top_k=top_k)
                response = await client.post(url, content=json.dumps(payload.model_dump(mode="python")))

                logger.debug("response.status_code=%s", response.status_code)

                response.raise_for_status()
                output = response.json().get("chunks")

                # Handle output fields
                output = [_flatten(chunk, output_fields) for chunk in output]

                return _wrap_nemo_results(output=output, content_field="content")

        except Exception as e:
            logger.exception("Encountered an error when retrieving results from Nemo Retriever: %s", e)
            raise CollectionUnavailableError(
                f"Error when retrieving documents from {collection_name} for query '{query}'") from e


def _wrap_nemo_results(output: list[dict], content_field: str):
    return RetrieverOutput(results=[_wrap_nemo_single_results(o, content_field=content_field) for o in output])


def _wrap_nemo_single_results(output: dict, content_field: str):
    return AIQDocument(page_content=output[content_field],
                       metadata={
                           k: v
                           for k, v in output.items() if k != content_field
                       })


def _flatten(obj: dict, output_fields: list[str]) -> list[str]:
    base_fields = [
        "format",
        "id",
    ]
    if not output_fields:
        output_fields = [
            "format",
            "id",
        ]
        output_fields.extend(list(obj["metadata"].keys()))
    data = {"content": obj.get("content")}
    for field in base_fields:
        if field in output_fields:
            data.update({field: obj[field]})

    data.update({k: v for k, v in obj['metadata'].items() if k in output_fields})
    return data


class NemoLangchainRetriever(BaseRetriever, BaseModel):
    client: NemoRetriever

    def _get_relevant_documents(self, query, *, run_manager, **kwargs):
        raise NotImplementedError

    async def _aget_relevant_documents(self, query, *, run_manager, **kwargs):
        return await self.client.search(query, **kwargs)
