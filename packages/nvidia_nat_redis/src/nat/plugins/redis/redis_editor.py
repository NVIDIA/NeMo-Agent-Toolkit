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

from nat.builder.context import Context
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

    async def add_items(self, items: list[MemoryItem], user_id: str, **kwargs) -> None:
        """
        Insert Multiple MemoryItems into Redis.
        Each MemoryItem is stored with its metadata and tags.

        Args:
            items (list[MemoryItem]): The items to be added.
            user_id (str): The user ID for which to add memories.
            kwargs (dict): Provider-specific keyword arguments.
        """
        logger.debug("Attempting to add %d items to Redis", len(items))

        for memory_item in items:
            item_meta = memory_item.metadata
            conversation = memory_item.conversation
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

    async def retrieve_memory(self, query: str, user_id: str, **kwargs) -> str:
        """
        Retrieve formatted memory from Redis using vector similarity search.

        Formats search results into structured memory with memory content,
        tags, and similarity scores.

        Args:
            query (str): The query string to match.
            user_id (str): The user ID for which to retrieve memory.
            kwargs (dict): Redis-specific keyword arguments.
                - top_k (int, optional): Maximum number of memories to include. Defaults to 5.
                - include_scores (bool, optional): Whether to include similarity scores. Defaults to True.

        Returns:
            str: Formatted memory string with relevant memories, or empty string if no results.
        """
        top_k = kwargs.pop("top_k", 5)
        include_scores = kwargs.pop("include_scores", True)

        logger.debug("retrieve_memory called with query: %s, user_id: %s, top_k: %d", query, user_id, top_k)

        # Generate embedding for query
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
        logger.debug("Created search query: %s", search_query.query_string())

        # Convert query vector to bytes
        try:
            query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
        except Exception as e:
            logger.error("Failed to convert vector to bytes: %s", e)
            raise

        try:
            # Execute the search
            results = await self._client.ft(INDEX_NAME).search(search_query, query_params={"vec": query_vector_bytes})

            logger.debug("Search returned %d results", len(results.docs))

            # Return empty string if no results
            if not results.docs:
                return ""

            # Format results into context string
            context_parts = ["Relevant memories (sorted by relevance):"]

            for i, doc in enumerate(results.docs, 1):
                try:
                    # Get the full document data
                    full_doc = await self._client.json().get(doc.id)
                    memory_text = full_doc.get("memory", "")
                    tags = full_doc.get("tags", [])
                    similarity_score = getattr(doc, 'score', None)

                    if not memory_text:
                        continue

                    # Format each memory with number
                    context_parts.append(f"\n{i}. {memory_text}")

                    # Add tags if present
                    if tags:
                        # Handle tags as string or list
                        if isinstance(tags, str):
                            context_parts.append(f"   (Tags: {tags})")
                        elif isinstance(tags, list) and tags:
                            context_parts.append(f"   (Tags: {', '.join(tags)})")

                    # Add similarity score if requested
                    if include_scores and similarity_score is not None:
                        context_parts.append(f"   (Similarity: {similarity_score:.3f})")

                    logger.debug("Formatted result %d", i)
                except Exception as e:
                    logger.error("Failed to process result %d: %s", i, e)
                    # Continue with other results
                    continue

            # If only header remains (no actual memories), return empty string
            if len(context_parts) == 1:
                return ""

            return "\n".join(context_parts)

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
        """Helper method to create a MemoryItem from Redis data.

        Note: user_id parameter is kept for backwards compatibility but is not used
        in MemoryItem construction since user_id is obtained from Context.
        """
        # Ensure tags is always a list
        tags = memory_data.get("tags", [])
        # Not sure why but sometimes the tags are retrieved as a string
        if isinstance(tags, str):
            tags = [tags]
        elif not isinstance(tags, list):
            tags = []

        return MemoryItem(conversation=memory_data.get("conversation", []),
                          memory=memory_data.get("memory", ""),
                          tags=tags,
                          metadata=memory_data.get("metadata", {}))

    async def remove_items(self, user_id: str, **kwargs):
        """
        Remove memory items for a specific user.

        Args:
            user_id (str): The user ID for which to remove memories (not currently used in filtering).
            kwargs: Additional parameters.
        """
        try:
            # Currently removes all memories with the key prefix
            # TODO: Filter by user_id if needed
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
