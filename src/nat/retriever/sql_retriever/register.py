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

"""Registration module for SQL Retriever."""

import logging

from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_retriever
from nat.data_models.retriever import RetrieverBaseConfig
from nat.retriever.sql_retriever.sql_retriever import SQLRetriever

logger = logging.getLogger(__name__)


class SQLRetrieverConfig(RetrieverBaseConfig, name="sql_retriever"):
    """
    Configuration for SQL Retriever.

    This retriever uses Vanna AI with NVIDIA NIM to convert natural language queries
    into SQL and retrieve data from SQL databases.

    Supported database types:
    - sqlite: SQLite databases (local file-based)
    - postgres/postgresql: PostgreSQL databases
    - sql: Generic SQL databases via SQLAlchemy (MySQL, SQL Server, Oracle, etc.)
    """

    llm_name: str = Field(description="Name of the LLM to use for SQL generation")
    embedding_name: str = Field(
        description="Name of the embedding model to use for Vanna training data"
    )
    vector_store_path: str = Field(
        description="Path to ChromaDB vector store for Vanna training data"
    )
    db_connection_string: str = Field(
        description=(
            "Database connection string. Format depends on db_type:\n"
            "- sqlite: Path to .db file (e.g., '/path/to/database.db')\n"
            "- postgres: Connection string (e.g., 'postgresql://user:pass@host:port/db')\n"
            "- sql: SQLAlchemy connection string (e.g., 'mysql+pymysql://user:pass@host/db')"
        )
    )
    db_type: str = Field(
        default="sqlite",
        description="Type of database: 'sqlite', 'postgres', or 'sql' (generic SQL via SQLAlchemy)",
    )
    training_data_path: str | None = Field(
        default=None,
        description="Path to YAML file containing Vanna training data (DDL, documentation, question-SQL pairs)",
    )
    max_results: int = Field(
        default=100,
        description="Maximum number of results to return from SQL queries",
    )
    nvidia_api_key: str | None = Field(
        default=None,
        description="NVIDIA API key (optional, defaults to NVIDIA_API_KEY environment variable)",
    )


@register_retriever(config_type=SQLRetrieverConfig)
async def create_sql_retriever(config: SQLRetrieverConfig, builder: Builder):
    """
    Create and register a SQL Retriever instance.

    Args:
        config: SQLRetrieverConfig containing all necessary parameters
        builder: Builder instance for accessing LLM and embedder configurations

    Returns:
        SQLRetriever: Configured SQL retriever instance

    Example YAML configuration:
        ```yaml
        retrievers:
          - name: sql_retriever
            type: sql_retriever
            llm_name: nim_llm
            embedding_name: nim_embeddings
            vector_store_path: ./vanna_vector_store
            db_connection_string: ./database.db
            db_type: sqlite
            training_data_path: ./training_data.yaml
            max_results: 100
        ```
    """
    logger.info(f"Creating SQL Retriever with config: {config.name}")

    # Get LLM and embedder configurations from builder
    llm_config = builder.get_llm_config(config.llm_name)
    embedder_config = builder.get_embedder_config(config.embedding_name)

    # Create SQL retriever instance
    retriever = SQLRetriever(
        llm_config=llm_config,
        embedder_config=embedder_config,
        vector_store_path=config.vector_store_path,
        db_connection_string=config.db_connection_string,
        db_type=config.db_type,
        training_data_path=config.training_data_path,
        nvidia_api_key=config.nvidia_api_key,
        max_results=config.max_results,
    )

    logger.info(f"SQL Retriever '{config.name}' created successfully")
    return retriever

