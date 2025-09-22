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

from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)


class ElasticsearchMixin:
    """Mixin for elasticsearch exporters.

    This mixin provides elasticsearch-specific functionality for SpanExporter exporters.
    It handles elasticsearch-specific resource tagging and uses the AsyncElasticsearch client.
    """

    def __init__(self,
                 *args,
                 endpoint: str,
                 index: str,
                 elasticsearch_auth: tuple[str, str],
                 headers: dict[str, str] | None = None,
                 **kwargs):
        """Initialize the elasticsearch exporter.

        Args:
            endpoint (str): The elasticsearch endpoint.
            index (str): The elasticsearch index.
            elasticsearch_auth (tuple[str, str]): The elasticsearch authentication credentials.
            headers (dict[str, str] | None): The elasticsearch headers.
        """
        if headers is None:
            headers = {"Accept": "application/vnd.elasticsearch+json; compatible-with=8"}

        self._elastic_client = AsyncElasticsearch(endpoint, basic_auth=elasticsearch_auth, headers=headers)
        self._index = index
        super().__init__(*args, **kwargs)

    async def export_processed(self, item: dict | list[dict]) -> None:
        """Export a batch of spans.

        Args:
            item (dict | list[dict]): Dictionary or list of dictionaries to export to Elasticsearch.
        """
        if isinstance(item, list):
            if not item:  # Empty list
                return
            if not all(isinstance(doc, dict) for doc in item):
                raise ValueError("All items in list must be dictionaries")

            # Format for bulk operations: each document needs an action/metadata line
            bulk_operations = []
            for doc in item:
                bulk_operations.append({"index": {"_index": self._index}})  # action/metadata with index
                bulk_operations.append(doc)  # document

            await self._elastic_client.bulk(operations=bulk_operations)
        elif isinstance(item, dict):
            # Single document export
            await self._elastic_client.index(index=self._index, document=item)
        else:
            raise ValueError(f"Invalid item type: {type(item)}. Expected dict or list[dict]")
