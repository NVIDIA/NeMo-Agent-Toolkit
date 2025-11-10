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
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ResponseIntermediateStep
from nat.data_models.function import FunctionBaseConfig
from nat.plugins.vanna.db_utils import RequiredSecretStr

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
    query_executed: str | None = Field(default=None, description="Actual SQL query executed (with prefixes)")
    dataframe_records: list[dict[str, Any]] = Field(default_factory=list, description="Results as list of dicts")
    dataframe_info: DataFrameInfo | None = Field(default=None, description="DataFrame metadata")
    failure_reason: str | None = Field(default=None, description="Reason for failure if query failed")
    limited_to: int | None = Field(default=None, description="Number of rows limited to")
    truncated: bool | None = Field(default=None, description="Whether truncated")


class ExecuteDBQueryConfig(FunctionBaseConfig, name="execute_db_query"):
    """
    Database query execution configuration.

    Currently only Databricks is supported.
    """

    # Database configuration
    database_type: str = Field(default="databricks",
                               description="Database type (currently only 'databricks' is supported)")
    connection_url: RequiredSecretStr = Field(description="Database connection string")

    # Query configuration
    max_rows: int = Field(default=100, description="Maximum rows to return")


@register_function(
    config_type=ExecuteDBQueryConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def execute_db_query(
    config: ExecuteDBQueryConfig,
    _builder: Builder,
):
    """Register the Execute DB Query function."""

    from nat.plugins.vanna.db_utils import async_execute_query
    from nat.plugins.vanna.db_utils import connect_to_database
    from nat.plugins.vanna.db_utils import extract_sql_from_message

    logger.info("Initializing Execute DB Query function")

    # Streaming version
    async def _execute_sql_query_stream(
        input_data: ExecuteDBQueryInput, ) -> AsyncGenerator[ResponseIntermediateStep | ExecuteDBQueryOutput, None]:
        """Stream SQL query execution progress and results."""
        sql_query = extract_sql_from_message(input_data.sql_query)
        logger.info(f"Executing SQL: {sql_query}")

        # Generate parent_id for this function call
        parent_id = str(uuid.uuid4())

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
                payload=StatusPayload(message="Connecting to database and executing query...").model_dump_json(),
            )

            # Validate database type
            if config.database_type.lower() != "databricks":
                yield ExecuteDBQueryOutput(
                    success=False,
                    failure_reason=f"Only Databricks is currently supported. Got database_type: {config.database_type}",
                    sql_query=sql_query,
                    dataframe_info=DataFrameInfo(shape=[0, 0], dtypes={}, columns=[]),
                )
                return

            connection_url_value = config.connection_url.get_secret_value()
            if not connection_url_value:
                yield ExecuteDBQueryOutput(
                    success=False,
                    failure_reason="Missing required connection URL",
                    sql_query=sql_query,
                    dataframe_info=DataFrameInfo(shape=[0, 0], dtypes={}, columns=[]),
                )
                return

            connection = connect_to_database(
                database_type=config.database_type,
                connection_url=connection_url_value,
            )

            if connection is None:
                yield ExecuteDBQueryOutput(
                    success=False,
                    failure_reason="Failed to connect to database",
                    sql_query=sql_query,
                    dataframe_info=DataFrameInfo(shape=[0, 0], dtypes={}, columns=[]),
                )
                return

            # Execute query
            query_result = await async_execute_query(connection, sql_query)
            df = query_result.to_dataframe()

            # Store original row count before limiting
            original_row_count = len(df)

            # Limit results
            if original_row_count > config.max_rows:
                df = df.head(config.max_rows)

            # Create response
            dataframe_info = DataFrameInfo(
                shape=[len(df), len(df.columns)] if not df.empty else [0, 0],
                dtypes=({
                    str(k): str(v)
                    for k, v in df.dtypes.to_dict().items()
                } if not df.empty else {}),
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
            # Note: Engine is left alive; connections are managed internally by SQLAlchemy pool

        except Exception as e:
            logger.error("Error executing SQL query", exc_info=e)
            yield ExecuteDBQueryOutput(
                success=False,
                failure_reason="SQL execution failed. Please check server logs for details.",
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
            failure_reason="No result returned",
            sql_query="",
            dataframe_info=DataFrameInfo(shape=[0, 0], dtypes={}, columns=[]),
        )

    description = (f"Execute SQL queries on {config.database_type} and return results. "
                   "Connects to the database, executes the provided SQL query, "
                   "and returns results in a structured format.")

    yield FunctionInfo.create(
        single_fn=_execute_sql_query,
        stream_fn=_execute_sql_query_stream,
        description=description,
        input_schema=ExecuteDBQueryInput,
    )
