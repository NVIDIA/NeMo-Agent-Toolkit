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

"""SQL Retriever implementation using Vanna for text-to-SQL generation."""

import json
import logging
from typing import Any

import pandas as pd

from nat.retriever.interface import Retriever
from nat.retriever.models import Document
from nat.retriever.models import RetrieverError
from nat.retriever.models import RetrieverOutput
from nat.retriever.sql_retriever.vanna_manager import VannaManager

logger = logging.getLogger(__name__)


class SQLRetriever(Retriever):
    """
    SQL Retriever that converts natural language queries to SQL and executes them.

    This retriever uses Vanna AI with NVIDIA NIM for text-to-SQL generation.
    It supports multiple database types: SQLite, PostgreSQL, and generic SQL databases via SQLAlchemy.

    Example:
        >>> from nat.retriever.sql_retriever import SQLRetriever
        >>> retriever = SQLRetriever(
        ...     llm_config=llm_config,
        ...     embedder_config=embedder_config,
        ...     vector_store_path="/path/to/vector_store",
        ...     db_connection_string="/path/to/database.db",
        ...     db_type="sqlite",
        ...     training_data_path="/path/to/training_data.yaml"
        ... )
        >>> results = await retriever.search("What are the top 10 customers by revenue?")
    """

    def __init__(
        self,
        llm_config: Any,
        embedder_config: Any,
        vector_store_path: str,
        db_connection_string: str,
        db_type: str = "sqlite",
        training_data_path: str = None,
        nvidia_api_key: str = None,
        max_results: int = 100,
        **kwargs,
    ):
        """
        Initialize the SQL Retriever.

        Args:
            llm_config: LLM configuration object with model_name attribute
            embedder_config: Embedder configuration object with model_name attribute
            vector_store_path: Path to ChromaDB vector store for Vanna training data
            db_connection_string: Database connection string:
                - SQLite: Path to .db file (e.g., "/path/to/database.db")
                - PostgreSQL: Connection string (e.g., "postgresql://user:pass@host:port/db")
                - Generic SQL: SQLAlchemy connection string (e.g., "mysql+pymysql://user:pass@host/db")
            db_type: Type of database - 'sqlite', 'postgres', or 'sql' (default: 'sqlite')
            training_data_path: Path to YAML file containing training data for Vanna
            nvidia_api_key: NVIDIA API key (optional, defaults to NVIDIA_API_KEY env var)
            max_results: Maximum number of results to return (default: 100)
            **kwargs: Additional keyword arguments
        """
        self.llm_config = llm_config
        self.embedder_config = embedder_config
        self.vector_store_path = vector_store_path
        self.db_connection_string = db_connection_string
        self.db_type = db_type
        self.training_data_path = training_data_path
        self.nvidia_api_key = nvidia_api_key
        self.max_results = max_results

        # Create VannaManager instance
        self.vanna_manager = VannaManager.create_with_config(
            vanna_llm_config=llm_config,
            vanna_embedder_config=embedder_config,
            vector_store_path=vector_store_path,
            db_connection_string=db_connection_string,
            db_type=db_type,
            training_data_path=training_data_path,
            nvidia_api_key=nvidia_api_key,
        )

        logger.info(
            f"SQLRetriever initialized with {db_type} database at {db_connection_string}"
        )

    async def search(self, query: str, **kwargs) -> RetrieverOutput:
        """
        Retrieve data from SQL database by converting natural language query to SQL.

        Args:
            query: Natural language query to convert to SQL
            **kwargs: Additional search parameters:
                - top_k: Maximum number of results to return (overrides max_results)
                - return_sql: If True, include the generated SQL in metadata (default: True)

        Returns:
            RetrieverOutput: Retrieved results with documents containing:
                - page_content: JSON string of result rows
                - metadata: Contains 'sql' (generated SQL), 'row_count', 'columns'

        Raises:
            RetrieverError: If SQL generation or execution fails
        """
        try:
            logger.info(f"SQLRetriever: Processing query: {query}")

            # Get parameters
            top_k = kwargs.get("top_k", self.max_results)
            return_sql = kwargs.get("return_sql", True)

            # Get Vanna instance
            vn_instance = self.vanna_manager.get_instance()

            # Generate SQL from natural language query
            logger.debug("Generating SQL query...")
            sql = self.vanna_manager.generate_sql_safe(question=query)
            logger.info(f"Generated SQL: {sql}")

            # Check if database is connected
            if not vn_instance.run_sql_is_set:
                raise RetrieverError(
                    f"Database is not connected. Cannot execute SQL: {sql}"
                )

            # Execute SQL query
            logger.debug("Executing SQL query...")
            df = vn_instance.run_sql(sql)

            if df is None:
                raise RetrieverError(f"SQL execution returned None for query: {sql}")

            if df.empty:
                logger.warning(f"No results found for query: {query}")
                return RetrieverOutput(
                    results=[
                        Document(
                            page_content=json.dumps([]),
                            metadata={
                                "sql": sql if return_sql else None,
                                "row_count": 0,
                                "columns": [],
                                "query": query,
                            },
                        )
                    ]
                )

            # Limit results
            if top_k and top_k < len(df):
                logger.debug(f"Limiting results to top {top_k} rows")
                df = df.head(top_k)

            # Convert DataFrame to documents
            results = self._dataframe_to_documents(
                df=df,
                sql=sql if return_sql else None,
                query=query,
            )

            logger.info(f"SQLRetriever: Retrieved {len(df)} rows")
            return results

        except Exception as e:
            logger.error(f"Error in SQLRetriever.search: {e}", exc_info=True)
            raise RetrieverError(f"Failed to retrieve data from SQL database: {e}") from e

    def _dataframe_to_documents(
        self, df: pd.DataFrame, sql: str = None, query: str = None
    ) -> RetrieverOutput:
        """
        Convert a pandas DataFrame to RetrieverOutput format.

        Args:
            df: Pandas DataFrame containing query results
            sql: Generated SQL query (optional)
            query: Original natural language query (optional)

        Returns:
            RetrieverOutput: Formatted retriever output
        """
        # Convert DataFrame to JSON
        results_json = df.to_json(orient="records")
        results_list = json.loads(results_json)

        # Create metadata
        metadata = {
            "row_count": len(df),
            "columns": df.columns.tolist(),
        }

        if sql:
            metadata["sql"] = sql
        if query:
            metadata["query"] = query

        # Create a single document containing all results
        # For better integration, we could also create one document per row
        # but that might be overwhelming for large result sets
        document = Document(
            page_content=json.dumps(results_list, indent=2),
            metadata=metadata,
        )

        return RetrieverOutput(results=[document])

    def get_stats(self) -> dict:
        """
        Get statistics about the SQLRetriever instance.

        Returns:
            dict: Statistics including VannaManager info
        """
        return {
            "db_type": self.db_type,
            "db_connection": self.db_connection_string,
            "vector_store_path": self.vector_store_path,
            "max_results": self.max_results,
            "vanna_manager": self.vanna_manager.get_stats(),
        }

