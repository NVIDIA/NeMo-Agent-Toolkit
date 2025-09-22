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
import secrets

import numpy as np
import redis.asyncio as redis
import redis.exceptions as redis_exceptions
from langchain_core.embeddings import Embeddings
from redis.commands.search.query import Query

from nat.memory.interfaces import MemoryEditor
from nat.memory.models import MemoryItem

logger = logging.getLogger(__name__)

INDEX_NAME = "memory_idx"


class RedisEditor(MemoryEditor):
    """
    Wrapper class that implements NAT interfaces for Redis memory storage.
    """

    def __init__(self, redis_client: redis.Redis, key_prefix: str, embedder: Embeddings):
        """
        Initialize Redis client for memory storage.

        Args:
            redis_client: (redis.Redis) Redis client
            key_prefix: (str) Redis key prefix
            embedder: (Embeddings) Embedder for semantic search functionality
        """

        self._client: redis.Redis = redis_client
        self._key_prefix: str = key_prefix
        self._embedder: Embeddings = embedder

    async def add_items(self, items: list[MemoryItem]) -> None:
        """
        Insert Multiple MemoryItems into Redis.
        Each MemoryItem is stored with its metadata and tags.
        """
        logger.debug("Attempting to add %d items to Redis", len(items))

        for memory_item in items:
            item_meta = memory_item.metadata
            conversation = memory_item.conversation
            user_id = memory_item.user_id
            tags = memory_item.tags
            memory_id = secrets.token_hex(4)  # e.g. 02ba3fe9

            # Create a unique key for this memory item
            memory_key = f"{self._key_prefix}:memory:{memory_id}"
            logger.debug("Generated memory key: %s", memory_key)

            # Prepare memory data
            memory_data = {
                "conversation": conversation,
                "user_id": user_id,
                "tags": tags,
                "metadata": item_meta,
                "memory": memory_item.memory or ""
            }
            logger.debug("Prepared memory data for key %s", memory_key)

            # If we have memory, compute and store the embedding
            if memory_item.memory:
                logger.debug("Computing embedding for memory text")
                search_vector = await self._embedder.aembed_query(memory_item.memory)
                logger.debug("Generated embedding vector of length: %d", len(search_vector))
                memory_data["embedding"] = search_vector

            try:
                # Store as JSON in Redis
                logger.debug("Attempting to store memory data in Redis for key: %s", memory_key)
                await self._client.json().set(memory_key, "$", memory_data)
                logger.debug("Successfully stored memory data for key: %s", memory_key)

                # Verify the data was stored
                stored_data = await self._client.json().get(memory_key)
                logger.debug("Verified data storage for key %s: %s", memory_key, bool(stored_data))

            except redis_exceptions.ResponseError as e:
                logger.error("Failed to store memory item: %s", e)
                raise
            except redis_exceptions.ConnectionError as e:
                logger.error("Redis connection error while storing memory item: %s", e)
                raise

    async def search(self, query: str, top_k: int = 5, **kwargs) -> list[MemoryItem]:
        """
        Retrieve items relevant to the given query.

        Args:
            query (str): The query string to match.
            top_k (int): Maximum number of items to return.
            kwargs (dict): Keyword arguments to pass to the search method.

        Returns:
            list[MemoryItem]: The most relevant MemoryItems for the given query.
        """
        logger.debug("Search called with query: %s, top_k: %d, kwargs: %s", query, top_k, kwargs)

        user_id = kwargs.get("user_id", "redis")  # TODO: remove this fallback username
        logger.debug("Using user_id: %s", user_id)

        # Perform vector search using Redis search
        logger.debug("Using embedder for vector search")
        try:
            logger.debug("Generating embedding for query: '%s'", query)
            query_vector = await self._embedder.aembed_query(query)
            logger.debug("Generated embedding vector of length: %d", len(query_vector))
        except Exception as e:
            logger.error("Failed to generate embedding: %s", e)
            raise

        # Create vector search query
        search_query = (
            Query(f"(@user_id:{user_id})=>[KNN {top_k} @embedding $vec AS score]").sort_by("score").return_fields(
                "conversation", "user_id", "tags", "metadata", "memory", "score").dialect(2))
        logger.debug("Created search query: %s", search_query)
        logger.debug("Query string: %s", search_query.query_string())

        # Convert query vector to bytes
        try:
            logger.debug("Converting query vector to bytes")
            query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
            logger.debug("Converted vector to bytes of length: %d", len(query_vector_bytes))
        except Exception as e:
            logger.error("Failed to convert vector to bytes: %s", e)
            raise

        try:
            # Execute search with vector parameters
            logger.debug("Executing Redis search with vector parameters")
            logger.debug("Search query parameters: vec length=%d", len(query_vector_bytes))

            # Log the actual query being executed
            logger.debug("Full search query: %s", search_query.query_string())

            # Check if there are any documents in the index
            try:
                total_docs = await self._client.ft(INDEX_NAME).info()
                logger.debug("Total documents in index: %d", total_docs.get('num_docs', 0))
            except Exception as e:
                logger.exception("Failed to get index info: %s", e)

            # Execute the search
            results = await self._client.ft(INDEX_NAME).search(search_query, query_params={"vec": query_vector_bytes})

            # Log detailed results information
            logger.debug("Search returned %d results", len(results.docs))
            logger.debug("Total results found: %d", results.total)

            # Convert results to MemoryItems
            memories = []
            for i, doc in enumerate(results.docs):
                try:
                    logger.debug("Processing result %d/%d", i + 1, len(results.docs))
                    logger.debug("Similarity score: %d", getattr(doc, 'score', 0))

                    # Get the full document data
                    full_doc = await self._client.json().get(doc.id)
                    logger.debug("Extracted data for result %d: %s", i + 1, full_doc)
                    memory_item = self._create_memory_item(dict(full_doc), user_id)
                    memories.append(memory_item)
                    logger.debug("Successfully created MemoryItem for result %d", i + 1)
                except Exception as e:
                    logger.error("Failed to process result %d: %s", i + 1, e)
                    raise

            logger.debug("Successfully processed all %d results", len(memories))
            return memories
        except redis_exceptions.ResponseError as e:
            logger.error("Search failed with ResponseError: %s", e)
            raise
        except redis_exceptions.ConnectionError as e:
            logger.error("Search failed with ConnectionError: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error during search: %s", e)
            raise

    def _create_memory_item(self, memory_data: dict, user_id: str) -> MemoryItem:
        """Helper method to create a MemoryItem from Redis data."""
        # Ensure tags is always a list
        tags = memory_data.get("tags", [])
        # Not sure why but sometimes the tags are retrieved as a string
        if isinstance(tags, str):
            tags = [tags]
        elif not isinstance(tags, list):
            tags = []

        return MemoryItem(conversation=memory_data.get("conversation", []),
                          user_id=user_id,
                          memory=memory_data.get("memory", ""),
                          tags=tags,
                          metadata=memory_data.get("metadata", {}))

    async def remove_items(self, **kwargs):
        """
        Remove memory items based on provided criteria.
        """
        try:
            pattern = f"{self._key_prefix}:memory:*"
            keys = await self._client.keys(pattern)
            if keys:
                await self._client.delete(*keys)
        except redis_exceptions.ResponseError as e:
            logger.error("Failed to remove items: %s", e)
            raise
        except redis_exceptions.ConnectionError as e:
            logger.error("Redis connection error while removing items: %s", e)
            raise
