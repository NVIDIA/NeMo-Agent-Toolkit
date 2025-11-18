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

from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ResponseIntermediateStep
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import RetrieverRef
from nat.data_models.function import FunctionBaseConfig
from nat.plugins.vanna.db_utils import RequiredSecretStr

logger = logging.getLogger(__name__)


class StatusPayload(BaseModel):
    """Payload for status intermediate steps."""
    message: str


class Text2SQLOutput(BaseModel):
    """Output schema for text2sql function."""
    sql: str = Field(description="Generated SQL query")
    explanation: str | None = Field(default=None, description="Explanation of the query")


class Text2SQLConfig(FunctionBaseConfig, name="text2sql"):
    """
    Text2SQL configuration with Vanna integration.

    Currently only Databricks is supported.
    """

    # LLM and Embedder
    llm_name: LLMRef = Field(description="LLM for SQL generation")
    embedder_name: EmbedderRef = Field(description="Embedder for vector operations")

    # Milvus retriever (required, must use async client)
    milvus_retriever: RetrieverRef = Field(description="Milvus retriever reference for vector operations. "
                                           "MUST be configured with use_async_client=true for text2sql function.")

    # Database configuration
    database_type: str = Field(default="databricks",
                               description="Database type (currently only 'databricks' is supported)")
    connection_url: RequiredSecretStr = Field(description="Database connection string")

    # Vanna Milvus configuration
    milvus_search_limit: int = Field(default=1000,
                                     description="Maximum limit size for vector search operations in Milvus")

    # Vanna configuration
    allow_llm_to_see_data: bool = Field(default=False, description="Allow LLM to see data for intermediate queries")
    execute_sql: bool = Field(default=False, description="Execute SQL or just return query string")
    train_on_startup: bool = Field(default=False, description="Train Vanna on startup")
    auto_training: bool = Field(default=False,
                                description=("Auto-train Vanna (auto-extract DDL and generate training data "
                                             "from database) or manually train Vanna (uses training data from "
                                             "training_db_schema.py)"))
    initial_prompt: str | None = Field(default=None, description="Custom system prompt")
    n_results: int = Field(default=5, description="Number of similar examples")
    sql_collection: str = Field(default="vanna_sql", description="Milvus collection for SQL examples")
    ddl_collection: str = Field(default="vanna_ddl", description="Milvus collection for DDL")
    doc_collection: str = Field(default="vanna_documentation", description="Milvus collection for docs")

    # Model-specific configuration
    reasoning_models: set[str] = Field(
        default={
            "nvidia/llama-3.1-nemotron-ultra-253b-v1",
            "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "deepseek-ai/deepseek-v3.1",
            "deepseek-ai/deepseek-r1",
        },
        description="Models that require special handling for think tags removal and JSON extraction")

    chat_models: set[str] = Field(default={"meta/llama-3.1-70b-instruct"},
                                  description="Models using standard response handling without think tags")


@register_function(config_type=Text2SQLConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def text2sql(config: Text2SQLConfig, builder: Builder):
    """Register the Text2SQL function with Vanna integration."""
    from nat.plugins.vanna.db_utils import setup_vanna_db_connection
    from nat.plugins.vanna.vanna_utils import VannaSingleton
    from nat.plugins.vanna.vanna_utils import train_vanna

    logger.info("Initializing Text2SQL function")

    # Check if singleton exists to avoid unnecessary client creation
    existing_instance = VannaSingleton.instance()
    if existing_instance is not None:
        logger.info("Reusing existing Vanna singleton instance")
        vanna_instance = existing_instance
    else:
        # Create all clients only when initializing new singleton
        logger.info("Creating new Vanna singleton instance")

        # Get LLM and embedder
        llm_client = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        embedder_client = await builder.get_embedder(config.embedder_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        # Get Milvus clients from retriever (expects async client)
        logger.info("Getting async Milvus client from milvus_retriever")
        retriever = await builder.get_retriever(config.milvus_retriever)

        # Vanna expects async client from retriever
        if not retriever._is_async:  # type: ignore[attr-defined]
            msg = (f"Milvus retriever '{config.milvus_retriever}' must be configured with "
                   "use_async_client=true for Vanna text2sql function")
            raise ValueError(msg)

        # Get async client from retriever
        async_milvus_client = retriever._client  # type: ignore[attr-defined]

        # Initialize Vanna instance (singleton pattern) with async client only
        vanna_instance = await VannaSingleton.get_instance(
            llm_client=llm_client,
            embedder_client=embedder_client,
            async_milvus_client=async_milvus_client,
            dialect=config.database_type,
            initial_prompt=config.initial_prompt,
            n_results=config.n_results,
            sql_collection=config.sql_collection,
            ddl_collection=config.ddl_collection,
            doc_collection=config.doc_collection,
            milvus_search_limit=config.milvus_search_limit,
            reasoning_models=config.reasoning_models,
            chat_models=config.chat_models,
            create_collections=config.train_on_startup,
        )

    # Validate database type
    if config.database_type.lower() != "databricks":
        msg = f"Only Databricks is currently supported. Got database_type: {config.database_type}"
        raise ValueError(msg)

    # Setup database connection (Engine stored in vanna_instance.db_engine)
    setup_vanna_db_connection(
        vn=vanna_instance,
        database_type=config.database_type,
        connection_url=config.connection_url.get_secret_value(),
    )

    # Train on startup if configured
    if config.train_on_startup:
        await train_vanna(vanna_instance, auto_train=config.auto_training)

    # Streaming version
    async def _generate_sql_stream(question: str, ) -> AsyncGenerator[ResponseIntermediateStep | Text2SQLOutput, None]:
        """Stream SQL generation progress and results."""
        logger.info(f"Text2SQL input: {question}")

        # Generate parent_id for this function call
        parent_id = str(uuid.uuid4())

        # Yield starting status as ResponseIntermediateStep
        yield ResponseIntermediateStep(
            id=str(uuid.uuid4()),
            parent_id=parent_id,
            type="markdown",
            name="text2sql_status",
            payload=StatusPayload(message="Starting SQL generation...").model_dump_json(),
        )

        try:
            # Generate SQL using Vanna (returns dict with sql and explanation)
            sql_result = await vanna_instance.generate_sql(
                question=question,
                allow_llm_to_see_data=config.allow_llm_to_see_data,
            )

            sql = str(sql_result.get("sql", ""))
            explanation: str | None = sql_result.get("explanation")

            # If execute_sql is enabled, run the query
            if config.execute_sql:
                yield ResponseIntermediateStep(
                    id=str(uuid.uuid4()),
                    parent_id=parent_id,
                    type="markdown",
                    name="text2sql_status",
                    payload=StatusPayload(message="Executing SQL query...").model_dump_json(),
                )
                # Execute SQL and propagate errors
                # Note: run_sql is dynamically set as async function in setup_vanna_db_connection
                df = await vanna_instance.run_sql(sql)  # type: ignore[misc]
                logger.info(f"SQL executed successfully: {len(df)} rows returned")

            # Yield final result as Text2SQLOutput
            yield Text2SQLOutput(sql=sql, explanation=explanation)

        except Exception as e:
            logger.error("SQL generation failed", exc_info=e)
            # Error status as ResponseIntermediateStep
            yield ResponseIntermediateStep(
                id=str(uuid.uuid4()),
                parent_id=parent_id,
                type="markdown",
                name="text2sql_error",
                payload=StatusPayload(
                    message="SQL generation failed. Please check server logs for details.").model_dump_json(),
            )
            raise

        logger.info("Text2SQL completed successfully")

    # Non-streaming version
    async def _generate_sql(question: str) -> Text2SQLOutput:
        """Generate SQL query from natural language."""
        async for update in _generate_sql_stream(question):
            # Skip ResponseIntermediateStep objects, only return Text2SQLOutput
            if isinstance(update, Text2SQLOutput):
                return update

        # Fallback if no result found
        return Text2SQLOutput(sql="", explanation=None)

    description = ("Generate SQL queries from natural language questions using AI. "
                   "Leverages similar question-SQL pairs, DDL information, and "
                   "documentation to generate accurate SQL queries. "
                   "Currently supports Databricks only.")

    if config.execute_sql:
        description += " Also executes queries and returns results."

    yield FunctionInfo.create(
        single_fn=_generate_sql,
        stream_fn=_generate_sql_stream,
        description=description,
    )
