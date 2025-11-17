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

import asyncio
import json
import logging
import re
import typing
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import PlainSerializer
from pydantic import SecretStr

logger = logging.getLogger(__name__)


def _serialize_secret(v: SecretStr) -> str:
    """Serialize SecretStr to plain string for required secret fields."""
    return v.get_secret_value()


# Required SecretStr that follows OptionalSecretStr pattern
RequiredSecretStr = typing.Annotated[SecretStr, PlainSerializer(_serialize_secret)]


class SupportedDatabase(str, Enum):
    """Supported database types for Vanna text-to-SQL."""

    DATABRICKS = "databricks"


class QueryResult(BaseModel):
    """Result from executing a database query."""

    results: list[tuple[Any, ...]] = Field(description="List of tuples representing rows returned from the query")
    column_names: list[str] = Field(description="List of column names for the result set")

    def to_dataframe(self) -> Any:
        """Convert query results to a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.results, columns=self.column_names)

    def to_records(self) -> list[dict[str, Any]]:
        """Convert query results to a list of dictionaries."""
        return [dict(zip(self.column_names, row, strict=False)) for row in self.results]

    @property
    def row_count(self) -> int:
        """Get the number of rows in the result set.

        Returns:
            Number of rows
        """
        return len(self.results)


def extract_sql_from_message(sql_query: str | Any) -> str:
    """Extract clean SQL query from various input formats.

    Handles:
    1. Direct SQL strings (passes through)
    2. BaseModel objects with 'sql' field (Text2SQLOutput)
    3. Dictionaries with 'sql' key
    4. Tool message format with content attribute
    5. String representations of tool messages

    Args:
        sql_query: SQL query in various formats

    Returns:
        Clean SQL query string
    """

    # Handle BaseModel objects (e.g., Text2SQLOutput)
    if isinstance(sql_query, BaseModel):
        # Try to get 'sql' field from BaseModel
        if hasattr(sql_query, "sql"):
            return sql_query.sql
        # Fall back to model_dump_json if no sql field
        sql_query = sql_query.model_dump_json()

    # Handle dictionaries with 'sql' key
    if isinstance(sql_query, dict):
        return sql_query.get("sql", str(sql_query))

    # Handle objects with content attribute (ToolMessage)
    if not isinstance(sql_query, str):
        if hasattr(sql_query, "content"):
            content = sql_query.content
            # Content might be a dict or list
            if isinstance(content, dict):
                return content.get("sql", str(content))
            if isinstance(content, list) and len(content) > 0:
                first_item = content[0]
                if isinstance(first_item, dict):
                    return first_item.get("sql", str(first_item))
            sql_query = str(content)
        else:
            sql_query = str(sql_query)

    # Extract from tool message format (legacy)
    if isinstance(sql_query, str) and 'content="' in sql_query:
        match = re.search(r'content="((?:[^"\\\\]|\\\\.)*)"', sql_query)
        if match:
            sql_query = match.group(1)
            sql_query = sql_query.replace("\\'", "'").replace('\\"', '"')

    # Try to parse as JSON if it looks like JSON
    if isinstance(sql_query, str) and sql_query.strip().startswith("{"):
        try:
            parsed = json.loads(sql_query)
            if isinstance(parsed, dict) and "sql" in parsed:
                return parsed["sql"]
        except json.JSONDecodeError:
            pass

    # Handle format: sql='...' explanation='...'
    if isinstance(sql_query, str) and "sql=" in sql_query:
        # Match sql='...' or sql="..." (non-greedy to stop at first closing quote before explanation)
        match = re.search(r"sql=['\"](.+?)['\"](?:\s+explanation=|$)", sql_query)
        if match:
            return match.group(1)

    return sql_query


def connect_to_databricks(connection_url: str) -> Any:
    """Connect to Databricks SQL Warehouse.

    Args:
        connection_url: Database connection string

    Returns:
        Databricks connection object
    """
    try:
        from sqlalchemy import create_engine

        connection = create_engine(url=connection_url, echo=False)
        logger.info("Connected to Databricks")
        return connection
    except Exception as e:
        logger.error(f"Failed to connect to Databricks: {e}")
        raise


def connect_to_database(
    database_type: str | SupportedDatabase,
    connection_url: str,
    **kwargs,
) -> Any:
    """Connect to a database based on type.

    Currently only Databricks is supported.

    Args:
        database_type: Type of database (currently only 'databricks' is supported)
        connection_url: Database connection string
        kwargs: Additional database-specific parameters

    Returns:
        Database connection object

    Raises:
        ValueError: If database_type is not supported
    """
    # Convert string to enum for validation
    if isinstance(database_type, str):
        try:
            db_type = SupportedDatabase(database_type.lower())
        except ValueError:
            supported = ", ".join([f"'{db.value}'" for db in SupportedDatabase])
            msg = f"Unsupported database type: '{database_type}'. Supported types: {supported}"
            raise ValueError(msg) from None
    else:
        db_type = database_type

    # Route to appropriate database connector
    if db_type == SupportedDatabase.DATABRICKS:
        return connect_to_databricks(connection_url=connection_url)

    # This should never be reached if enum is properly defined
    msg = f"Database type '{db_type.value}' has no connector implementation"
    raise NotImplementedError(msg)


def execute_query(connection: Any, query: str) -> QueryResult:
    """Execute a query and return results.

    Args:
        connection: Database connection object
        query: SQL query to execute

    Returns:
        QueryResult object containing results and column names
    """
    from sqlalchemy import text
    try:
        with connection.connect() as conn:
            logger.info(f"Executing query: {query}")
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = list(result.keys()) if result.keys() else []

            logger.info(f"Query completed, retrieved {len(rows)} rows")
            return QueryResult(results=rows, column_names=columns)

    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise


async def async_execute_query(connection: Any, query: str) -> QueryResult:
    """Execute query asynchronously and return QueryResult.

    Args:
        connection: Database connection object
        query: SQL query to execute

    Returns:
        QueryResult object containing results and column names
    """

    # Run synchronous query in executor
    loop = asyncio.get_event_loop()
    query_result = await loop.run_in_executor(None, execute_query, connection, query)

    return query_result


def setup_vanna_db_connection(
    vn: Any,
    database_type: str | SupportedDatabase,
    connection_url: str,
    **kwargs,
) -> None:
    """Set up database connection for Vanna instance.

    Currently only Databricks is supported.

    The database Engine is stored in the Vanna instance (vn.db_engine) and will
    persist for the lifetime of the Vanna singleton. The Engine will be disposed
    when the Vanna singleton is reset.

    Args:
        vn: Vanna instance
        database_type: Type of database (currently only 'databricks' is supported)
        connection_url: Database connection string
        kwargs: Additional connection parameters

    Raises:
        ValueError: If database_type is not supported
    """

    # Reuse existing engine if already connected to same URL
    if hasattr(vn, "db_engine") and vn.db_engine is not None:
        logger.info("Reusing existing database engine from Vanna instance")
        engine = vn.db_engine
    else:
        # Connect to database (validation handled by connect_to_database)
        engine = connect_to_database(database_type=database_type, connection_url=connection_url)
        # Store engine in Vanna instance - lifecycle matches singleton
        vn.db_engine = engine
        logger.info(f"Created and stored database engine in Vanna instance for {database_type}")

    # Define async run_sql function for Vanna
    async def run_sql(sql_query: str) -> Any:
        """Execute SQL asynchronously and return DataFrame."""
        try:
            query_result = await async_execute_query(engine, sql_query)
            return query_result.to_dataframe()
        except Exception:
            logger.exception("Error executing SQL")
            raise

    # Set up Vanna
    vn.run_sql = run_sql
    vn.run_sql_is_set = True

    logger.info(f"Database connection configured for {database_type}")
