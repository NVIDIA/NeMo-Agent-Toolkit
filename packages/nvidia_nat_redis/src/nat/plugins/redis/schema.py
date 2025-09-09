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

import logging

import redis.asyncio as redis
import redis.exceptions as redis_exceptions
from redis.commands.search.field import TagField
from redis.commands.search.field import TextField
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition
from redis.commands.search.indexDefinition import IndexType

logger = logging.getLogger(__name__)

INDEX_NAME = "memory_idx"
DEFAULT_DIM = 384  # Default embedding dimension


def create_schema(embedding_dim: int = DEFAULT_DIM):
    """
    Create the Redis search schema for redis_memory.

    Args:
        embedding_dim (int): Dimension of the embedding vectors

    Returns:
        tuple: Schema definition for Redis search
    """
    logger.info("Creating schema with embedding dimension: %d", embedding_dim)

    embedding_field = VectorField("$.embedding",
                                  "HNSW",
                                  {
                                      "TYPE": "FLOAT32",
                                      "DIM": embedding_dim,
                                      "DISTANCE_METRIC": "L2",
                                      "INITIAL_CAP": 100,
                                      "M": 16,
                                      "EF_CONSTRUCTION": 200,
                                      "EF_RUNTIME": 10
                                  },
                                  as_name="embedding")
    logger.info("Created embedding field with dimension %d", embedding_dim)

    schema = (
        TextField("$.user_id", as_name="user_id"),
        TagField("$.tags[*]", as_name="tags"),
        TextField("$.memory", as_name="memory"),
        # TextField("$.conversations[*]", as_name="conversations"), # TODO: figure out if/how this should be done
        embedding_field)

    # Log the schema details
    logger.info("Schema fields:")
    for field in schema:
        logger.info("  - %s: %s", field.name, type(field).__name__)

    return schema


async def ensure_index_exists(client: redis.Redis, key_prefix: str, embedding_dim: int | None) -> None:
    """
    Ensure the Redis search index exists, creating it if necessary.

    Args:
        client (redis.Redis): Redis client instance
        key_prefix (str): Prefix for keys to be indexed
        embedding_dim (Optional[int]): Dimension of embedding vectors. If None, uses default.
    """
    try:
        # Check if index exists
        logger.info("Checking if index '%s' exists...", INDEX_NAME)
        info = await client.ft(INDEX_NAME).info()
        logger.info("Redis search index '%s' exists.", INDEX_NAME)

        # Verify the schema
        schema = info.get('attributes', [])

        return
    except redis_exceptions.ResponseError as ex:
        error_msg = str(ex)
        if "no such index" not in error_msg.lower() and "Index needs recreation" not in error_msg:
            logger.error("Unexpected Redis error: %s", error_msg)
            raise

        # Index doesn't exist or needs recreation
        logger.info("Creating Redis search index '%s' with prefix '%s'", INDEX_NAME, key_prefix)

        # Drop any existing index
        try:
            logger.info("Attempting to drop existing index '%s' if it exists", INDEX_NAME)
            await client.ft(INDEX_NAME).dropindex()
            logger.info("Successfully dropped existing index '%s'", INDEX_NAME)
        except redis_exceptions.ResponseError as e:
            if "no such index" not in str(e).lower():
                logger.warning("Error while dropping index: %s", str(e))

        # Create new schema and index
        schema = create_schema(embedding_dim or DEFAULT_DIM)
        logger.info("Created schema with embedding dimension: %d", embedding_dim or DEFAULT_DIM)

        try:
            # Create the index
            logger.info("Creating new index '%s' with schema", INDEX_NAME)
            await client.ft(INDEX_NAME).create_index(schema,
                                                     definition=IndexDefinition(prefix=[f"{key_prefix}:"],
                                                                                index_type=IndexType.JSON))

            # Verify index was created
            info = await client.ft(INDEX_NAME).info()
            logger.info("Successfully created Redis search index '%s'", INDEX_NAME)
            logger.debug("Redis search index info: %s", info)

            # Verify the schema
            schema = info.get('attributes', [])
            logger.debug("New index schema: %s", schema)

        except redis_exceptions.ResponseError as e:
            logger.error("Failed to create index: %s", str(e))
            raise
        except redis_exceptions.ConnectionError as e:
            logger.error("Redis connection error while creating index: %s", str(e))
            raise
