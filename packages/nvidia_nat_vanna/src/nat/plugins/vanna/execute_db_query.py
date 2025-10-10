"""Database query execution function for NeMo Agent Toolkit."""

import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ResponseIntermediateStep
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field, TypeAdapter

logger = logging.getLogger(__name__)


class StatusPayload(BaseModel):
    """Payload for status intermediate steps."""
    message: str


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


@register_function(
    config_type=ExecuteDBQueryConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def execute_db_query(
    config: ExecuteDBQueryConfig,
    builder: Builder,
):
    """Register the Execute DB Query function."""
    # Import implementation details inside the registration function
    import pandas as pd

    from nat.plugins.vanna.db_utils import (
        async_query,
        connect_to_database,
        extract_sql_from_message,
    )

    logger.info("Initializing Execute DB Query function")

    # Streaming version
    async def _execute_sql_query_stream(
        input_data: ExecuteDBQueryInput,
    ) -> AsyncGenerator[ResponseIntermediateStep | ExecuteDBQueryOutput, None]:
        """Stream SQL query execution progress and results."""
        sql_query = extract_sql_from_message(input_data.sql_query)
        logger.info(f"Executing SQL: {sql_query}")

        # Generate parent_id for this function call
        parent_id = str(uuid.uuid4())

        yield ResponseIntermediateStep(
            id=str(uuid.uuid4()),
            parent_id=parent_id,
            type="markdown",
            name="execute_db_query_status",
            payload=StatusPayload(message="Starting SQL query execution...").model_dump_json(),
        )

        try:
            # Clean up query
            sql_query = sql_query.strip()
            if sql_query.startswith('"') and sql_query.endswith('"'):
                sql_query = sql_query[1:-1]
            if sql_query.startswith("'") and sql_query.endswith("'"):
                sql_query = sql_query[1:-1]

            yield ResponseIntermediateStep(
                id=str(uuid.uuid4()),
                parent_id=parent_id,
                type="markdown",
                name="execute_db_query_status",
                payload=StatusPayload(message="Connecting to database...").model_dump_json(),
            )

            # Connect to database
            if config.database_type == "databricks":
                if not all(
                    [
                        config.databricks_server_hostname,
                        config.databricks_http_path,
                        config.databricks_access_token,
                    ]
                ):
                    yield ExecuteDBQueryOutput(
                        success=False,
                        error="Missing Databricks connection parameters",
                        sql_query=sql_query,
                        dataframe_info=DataFrameInfo(shape=[0, 0], dtypes={}, columns=[]),
                    )
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
                    yield ExecuteDBQueryOutput(
                        success=False,
                        error="Missing database connection parameters",
                        sql_query=sql_query,
                        dataframe_info=DataFrameInfo(shape=[0, 0], dtypes={}, columns=[]),
                    )
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
                yield ExecuteDBQueryOutput(
                    success=False,
                    error="Failed to connect to database",
                    sql_query=sql_query,
                    dataframe_info=DataFrameInfo(shape=[0, 0], dtypes={}, columns=[]),
                )
                return

            yield ResponseIntermediateStep(
                id=str(uuid.uuid4()),
                parent_id=parent_id,
                type="markdown",
                name="execute_db_query_status",
                payload=StatusPayload(message="Executing SQL query...").model_dump_json(),
            )

            # Execute query
            df = await async_query(
                connection, sql_query, config.db_catalog, config.db_schema, config.database_type
            )

            # Close connection
            connection.close()

            yield ResponseIntermediateStep(
                id=str(uuid.uuid4()),
                parent_id=parent_id,
                type="markdown",
                name="execute_db_query_status",
                payload=StatusPayload(message="Processing results...").model_dump_json(),
            )

            # Store original row count before limiting
            original_row_count = len(df)

            # Limit results
            if original_row_count > config.max_rows:
                df = df.head(config.max_rows)

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
                columns=df.columns.tolist() if not df.empty else [],
                row_count=original_row_count,
                sql_query=sql_query,
                query_executed=sql_query,
                dataframe_records=df.to_dict("records") if not df.empty else [],
                dataframe_info=dataframe_info,
            )

            if original_row_count > config.max_rows:
                response.limited_to = config.max_rows
                response.truncated = True

            # Yield final result as ExecuteDBQueryOutput
            yield response

        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            yield ExecuteDBQueryOutput(
                success=False,
                error=str(e),
                sql_query=sql_query,
                dataframe_info=DataFrameInfo(shape=[0, 0], dtypes={}, columns=[]),
            )

        logger.info("Execute DB Query completed")

    # Non-streaming version
    async def _execute_sql_query(input_data: ExecuteDBQueryInput) -> ExecuteDBQueryOutput:
        """Execute SQL query and return results."""
        async for update in _execute_sql_query_stream(input_data):
            # Skip ResponseIntermediateStep objects, only return ExecuteDBQueryOutput
            if isinstance(update, ExecuteDBQueryOutput):
                return update

        # Fallback if no result found
        return ExecuteDBQueryOutput(
            success=False,
            error="No result returned",
            sql_query="",
            dataframe_info=DataFrameInfo(shape=[0, 0], dtypes={}, columns=[]),
        )

    description = (
        f"Execute SQL queries on {config.database_type} and return results. "
        "Connects to the database, executes the provided SQL query, "
        "and returns results in a structured format."
    )

    yield FunctionInfo.create(
        single_fn=_execute_sql_query,
        stream_fn=_execute_sql_query_stream,
        description=description,
        input_schema=ExecuteDBQueryInput,
    )
