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
import re
from typing import Any

from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


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
    from pydantic import BaseModel

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
        import json
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


def connect_to_databricks(server_hostname: str, http_path: str, access_token: str) -> Any:
    """Connect to Databricks SQL Warehouse.

    Args:
        server_hostname: Databricks server hostname
        http_path: HTTP path to SQL warehouse
        access_token: Access token for authentication

    Returns:
        Databricks connection object
    """
    try:
        from databricks import sql

        connection = sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token,
        )
        logger.info("Connected to Databricks")
        return connection
    except ImportError:
        logger.error("databricks-sql-connector not installed")
        raise
    except Exception as e:
        logger.error(f"Failed to connect to Databricks: {e}")
        raise


def connect_to_database(
    database_type: str,
    host: str | None = None,
    port: int | None = None,
    database: str | None = None,
    username: str | None = None,
    password: str | None = None,
    **kwargs,
) -> Any:
    """Connect to a database based on type.

    Args:
        database_type: Type of database ('databricks', 'postgres', 'mysql', etc.)
        host: Database host
        port: Database port
        database: Database name
        username: Username for authentication
        password: Password for authentication
        **kwargs: Additional database-specific parameters

    Returns:
        Database connection object
    """
    database_type = database_type.lower()

    if database_type == "databricks":
        return connect_to_databricks(
            server_hostname=kwargs.get("server_hostname", host),
            http_path=kwargs.get("http_path", ""),
            access_token=kwargs.get("access_token", password),
        )

    elif database_type in {"postgres", "postgresql"}:
        try:
            import psycopg2

            connection = psycopg2.connect(
                host=host,
                port=port or 5432,
                database=database,
                user=username,
                password=password,
            )
            logger.info("Connected to PostgreSQL")
            return connection
        except ImportError:
            logger.error("psycopg2 not installed")
            raise

    elif database_type == "mysql":
        try:
            import mysql.connector

            connection = mysql.connector.connect(
                host=host,
                port=port or 3306,
                database=database,
                user=username,
                password=password,
            )
            logger.info("Connected to MySQL")
            return connection
        except ImportError:
            logger.error("mysql-connector-python not installed")
            raise

    elif database_type == "sqlite":
        try:
            import sqlite3

            connection = sqlite3.connect(database or ":memory:")
            logger.info("Connected to SQLite")
            return connection
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise

    else:
        msg = f"Unsupported database type: {database_type}"
        raise ValueError(msg)


def execute_query(
    connection: Any,
    query: str,
    catalog: str | None = None,
    schema: str | None = None,
    database_type: str | None = None,
) -> QueryResult:
    """Execute a query and return results.

    Args:
        connection: Database connection object
        query: SQL query to execute
        catalog: Optional catalog to use (Databricks only)
        schema: Optional schema to use
        database_type: Type of database for proper catalog/schema handling

    Returns:
        QueryResult object containing results and column names
    """
    try:
        with connection.cursor() as cursor:
            # Database-specific catalog and schema handling
            if database_type:
                db_type = database_type.lower()

                if db_type == "databricks":
                    if catalog:
                        cursor.execute(f"USE CATALOG {catalog}")
                    if schema:
                        cursor.execute(f"USE SCHEMA {schema}")

                elif db_type in ("postgres", "postgresql"):
                    if schema:
                        cursor.execute(f"SET search_path TO {schema}")
                    # PostgreSQL doesn't have catalog concept

                elif db_type == "mysql":
                    if schema:
                        cursor.execute(f"USE {schema}")
                    # MySQL uses database, not catalog

                elif db_type in ("mssql", "sqlserver"):
                    # SQL Server uses database.schema notation
                    # Schema is typically set in the query itself or connection string
                    pass

            logger.info(f"Executing query: {query}")
            cursor.execute(query)

            results = cursor.fetchall()
            columns = ([desc[0] for desc in cursor.description] if cursor.description else [])

            logger.info(f"Query completed, retrieved {len(results)} rows")
            return QueryResult(results=results, column_names=columns)

    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise


async def async_execute_query(
    connection: Any,
    query: str,
    catalog: str | None = None,
    schema: str | None = None,
    database_type: str | None = None,
):
    """Execute query asynchronously and return DataFrame.

    Args:
        connection: Database connection object
        query: SQL query to execute
        catalog: Optional catalog to use
        schema: Optional schema to use
        database_type: Type of database for proper catalog/schema handling

    Returns:
        DataFrame with query results
    """
    import asyncio

    # Run synchronous query in executor
    loop = asyncio.get_event_loop()
    query_result = await loop.run_in_executor(None, execute_query, connection, query, catalog, schema, database_type)

    return query_result.to_dataframe()


def setup_vanna_db_connection(
    vn: Any,
    database_type: str,
    host: str | None = None,
    port: int | None = None,
    database: str | None = None,
    username: str | None = None,
    password: str | None = None,
    catalog: str | None = None,
    schema: str | None = None,
    **kwargs,
) -> Any:
    """Set up database connection for Vanna instance.

    Args:
        vn: Vanna instance
        database_type: Type of database
        host: Database host
        port: Database port
        database: Database name
        username: Username
        password: Password
        catalog: Catalog name (for Databricks)
        schema: Schema name
        **kwargs: Additional connection parameters

    Returns:
        Database connection object (must be closed by caller)
    """
    import pandas as pd

    # Connect to database
    if database_type == "databricks":
        connection = connect_to_databricks(
            server_hostname=kwargs.get("server_hostname", host),
            http_path=kwargs.get("http_path", ""),
            access_token=kwargs.get("access_token", password),
        )
    else:
        connection = connect_to_database(
            database_type=database_type,
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            **kwargs,
        )

    # Define async run_sql function for Vanna
    async def run_sql(sql_query: str) -> pd.DataFrame:
        """Execute SQL asynchronously and return DataFrame."""
        try:
            return await async_execute_query(connection, sql_query, catalog, schema, database_type)
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            raise

    # Set up Vanna
    vn.run_sql = run_sql
    vn.run_sql_is_set = True

    logger.info(f"Database connection configured for {database_type}")
    return connection
