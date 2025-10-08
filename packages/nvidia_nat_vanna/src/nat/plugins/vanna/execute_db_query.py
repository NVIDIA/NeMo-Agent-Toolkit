"""Database query execution function for NeMo Agent Toolkit."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

import pandas as pd
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field, TypeAdapter

from nat.plugins.vanna.db_utils import (
    add_table_prefix,
    connect_to_database,
    execute_query,
    extract_sql_from_message,
)

logger = logging.getLogger(__name__)


class ExecuteDBQueryInput(BaseModel):
    """Input schema for execute DB query function."""

    sql_query: str = Field(description="SQL query to execute")


class DataFrameInfo(BaseModel):
    """DataFrame structure information."""

    shape: list[int] = Field(description="Shape [rows, columns]")
    dtypes: dict[str, str] = Field(description="Column data types")
    columns: list[str] = Field(description="Column names")


class ExecuteDBQueryOutput(BaseModel):
    """Output schema for execute DB query function."""

    success: bool = Field(description="Whether query executed successfully")
    columns: list[str] = Field(default_factory=list, description="Column names")
    row_count: int = Field(default=0, description="Total rows returned")
    sql_query: str = Field(description="Original SQL query")
    query_executed: str | None = Field(
        default=None, description="Actual SQL query executed (with prefixes)"
    )
    dataframe_records: list[dict[str, Any]] = Field(
        default_factory=list, description="Results as list of dicts"
    )
    dataframe_info: DataFrameInfo | None = Field(
        default=None, description="DataFrame metadata"
    )
    error: str | None = Field(default=None, description="Error message if failed")
    limited_to: int | None = Field(
        default=None, description="Number of rows limited to"
    )
    truncated: bool | None = Field(default=None, description="Whether truncated")


class ExecuteDBQueryConfig(FunctionBaseConfig, name="execute_db_query"):
    """Database query execution configuration."""

    _type: str = "execute_db_query"

    # Database configuration
    database_type: str = Field(default="databricks", description="Database type")
    db_host: str | None = Field(default=None, description="Database host")
    db_port: int | None = Field(default=None, description="Database port")
    db_name: str | None = Field(default=None, description="Database name")
    db_username: str | None = Field(default=None, description="Database username")
    db_password: str | None = Field(default=None, description="Database password")
    db_catalog: str | None = Field(default=None, description="Database catalog")
    db_schema: str | None = Field(default=None, description="Database schema")

    # Databricks-specific
    databricks_server_hostname: str | None = Field(
        default=None, description="Databricks server hostname"
    )
    databricks_http_path: str | None = Field(
        default=None, description="Databricks HTTP path"
    )
    databricks_access_token: str | None = Field(
        default=None, description="Databricks access token"
    )

    # Query configuration
    max_rows: int = Field(default=100, description="Maximum rows to return")
    add_table_prefix_enabled: bool = Field(
        default=True, description="Add catalog.schema prefix to tables"
    )


@register_function(
    config_type=ExecuteDBQueryConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def execute_db_query(
    config: ExecuteDBQueryConfig,
    builder: Builder,
):
    """Register the Execute DB Query function."""
    logger.info("Initializing Execute DB Query function")

    # Streaming version
    async def _execute_sql_query_stream(
        input_data: ExecuteDBQueryInput,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream SQL query execution progress and results."""
        sql_query = extract_sql_from_message(input_data.sql_query)
        logger.info(f"Executing SQL: {sql_query}")

        yield {
            "type": "status",
            "message": "Starting SQL query execution...",
            "node": "execute_db_query",
        }

        try:
            # Clean up query
            sql_query = sql_query.strip()
            if sql_query.startswith('"') and sql_query.endswith('"'):
                sql_query = sql_query[1:-1]
            if sql_query.startswith("'") and sql_query.endswith("'"):
                sql_query = sql_query[1:-1]

            yield {
                "type": "status",
                "message": "Connecting to database...",
                "node": "execute_db_query",
            }

            # Connect to database
            if config.database_type == "databricks":
                if not all(
                    [
                        config.databricks_server_hostname,
                        config.databricks_http_path,
                        config.databricks_access_token,
                    ]
                ):
                    error_response = ExecuteDBQueryOutput(
                        success=False,
                        error="Missing Databricks connection parameters",
                        sql_query=sql_query,
                        dataframe_info=DataFrameInfo(shape=[0, 0], dtypes={}, columns=[]),
                    )
                    yield {
                        "type": "result",
                        "query_result": error_response.model_dump(),
                        "node": "execute_db_query",
                    }
                    return

                connection = connect_to_database(
                    database_type=config.database_type,
                    host=config.databricks_server_hostname,
                    server_hostname=config.databricks_server_hostname,
                    http_path=config.databricks_http_path,
                    access_token=config.databricks_access_token,
                )
            else:
                if not all([config.db_host, config.db_name]):
                    error_response = ExecuteDBQueryOutput(
                        success=False,
                        error="Missing database connection parameters",
                        sql_query=sql_query,
                        dataframe_info=DataFrameInfo(shape=[0, 0], dtypes={}, columns=[]),
                    )
                    yield {
                        "type": "result",
                        "query_result": error_response.model_dump(),
                        "node": "execute_db_query",
                    }
                    return

                connection = connect_to_database(
                    database_type=config.database_type,
                    host=config.db_host,
                    port=config.db_port,
                    database=config.db_name,
                    username=config.db_username,
                    password=config.db_password,
                )

            if connection is None:
                error_response = ExecuteDBQueryOutput(
                    success=False,
                    error="Failed to connect to database",
                    sql_query=sql_query,
                    dataframe_info=DataFrameInfo(shape=[0, 0], dtypes={}, columns=[]),
                )
                yield {
                    "type": "result",
                    "query_result": error_response.model_dump(),
                    "node": "execute_db_query",
                }
                return

            # Add table prefix if configured
            original_query = sql_query
            if config.add_table_prefix_enabled and (
                config.db_catalog or config.db_schema
            ):
                yield {
                    "type": "status",
                    "message": "Adding table prefix...",
                    "node": "execute_db_query",
                }
                prefix_parts = []
                if config.db_catalog:
                    prefix_parts.append(config.db_catalog)
                if config.db_schema:
                    prefix_parts.append(config.db_schema)
                table_prefix = ".".join(prefix_parts)
                sql_query = add_table_prefix(sql_query, table_prefix)
                logger.info(f"Modified query: {sql_query}")

            yield {
                "type": "status",
                "message": "Executing SQL query...",
                "node": "execute_db_query",
            }

            # Execute query
            results, columns = execute_query(
                connection, sql_query, config.db_catalog, config.db_schema
            )

            # Close connection
            connection.close()

            yield {
                "type": "status",
                "message": "Processing results...",
                "node": "execute_db_query",
            }

            # Limit results
            limited_results = (
                results[: config.max_rows]
                if len(results) > config.max_rows
                else results
            )

            # Convert to serializable format
            serializable_results = []
            for row in limited_results:
                serializable_row = []
                for value in row:
                    if value is None:
                        serializable_row.append(None)
                    elif isinstance(value, (int, float, str, bool)):
                        serializable_row.append(value)
                    else:
                        serializable_row.append(str(value))
                serializable_results.append(serializable_row)

            # Create DataFrame
            if serializable_results and columns:
                df = pd.DataFrame(serializable_results, columns=columns)
            else:
                df = pd.DataFrame()

            # Create response
            dataframe_info = DataFrameInfo(
                shape=[len(df), len(df.columns)] if not df.empty else [0, 0],
                dtypes=(
                    {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
                    if not df.empty
                    else {}
                ),
                columns=df.columns.tolist() if not df.empty else [],
            )

            response = ExecuteDBQueryOutput(
                success=True,
                columns=[str(col) for col in columns] if columns else [],
                row_count=len(results),
                sql_query=original_query,
                query_executed=sql_query,
                dataframe_records=df.to_dict("records") if not df.empty else [],
                dataframe_info=dataframe_info,
            )

            if len(results) > config.max_rows:
                response.limited_to = config.max_rows
                response.truncated = True

            yield {
                "type": "result",
                "query_result": response.model_dump(),
                "node": "execute_db_query",
            }

        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            error_response = ExecuteDBQueryOutput(
                success=False,
                error=str(e),
                sql_query=sql_query,
                dataframe_info=DataFrameInfo(shape=[0, 0], dtypes={}, columns=[]),
            )
            yield {
                "type": "result",
                "query_result": error_response.model_dump(),
                "node": "execute_db_query",
            }

        logger.info("Execute DB Query completed")

    # Non-streaming version
    async def _execute_sql_query(input_data: ExecuteDBQueryInput) -> str:
        """Execute SQL query and return results."""
        result = None

        async for update in _execute_sql_query_stream(input_data):
            if update["type"] == "result":
                result = update["query_result"]

        if result and isinstance(result, dict):
            return TypeAdapter(dict).dump_json(result).decode("utf-8")

        return str(result) if result else "{}"

    description = (
        f"Execute SQL queries on {config.database_type} and return results. "
        "Connects to the database, executes the provided SQL query, "
        "and returns results in a structured format."
    )

    if config.add_table_prefix_enabled:
        description += " Automatically adds table prefixes if configured."

    yield FunctionInfo.create(
        single_fn=_execute_sql_query,
        stream_fn=_execute_sql_query_stream,
        description=description,
        input_schema=ExecuteDBQueryInput,
    )

