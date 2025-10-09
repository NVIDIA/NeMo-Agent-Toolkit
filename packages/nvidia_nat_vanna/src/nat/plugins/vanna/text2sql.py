"""Text-to-SQL function for NeMo Agent Toolkit with Vanna integration."""

import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ResponseIntermediateStep
from nat.data_models.component_ref import EmbedderRef, LLMRef
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StatusPayload(BaseModel):
    """Payload for status intermediate steps."""
    message: str


class Text2SQLOutput(BaseModel):
    """Output schema for text2sql function."""
    sql: str = Field(description="Generated SQL query")
    explanation: str | None = Field(default=None, description="Explanation of the query")


class Text2SQLConfig(FunctionBaseConfig, name="text2sql"):
    """Text2SQL configuration with Vanna integration."""

    _type: str = "text2sql"

    # LLM and Embedder
    llm_name: LLMRef = Field(description="LLM for SQL generation")
    embedder_name: EmbedderRef = Field(description="Embedder for vector operations")

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

    # Milvus configuration
    milvus_host: str | None = Field(default=None, description="Milvus host")
    milvus_port: int | None = Field(default=None, description="Milvus port")
    milvus_user: str | None = Field(default=None, description="Milvus username")
    milvus_password: str | None = Field(default=None, description="Milvus password")
    milvus_db_name: str | None = Field(default=None, description="Milvus database")
    milvus_use_tls: bool = Field(default=True, description="Use TLS for Milvus")

    # Vanna configuration
    allow_llm_to_see_data: bool = Field(
        default=False, description="Allow LLM to see data for intermediate queries"
    )
    execute_sql: bool = Field(
        default=False, description="Execute SQL or just return query string"
    )
    train_on_startup: bool = Field(
        default=False, description="Train Vanna on startup"
    )
    training_ddl: list[str] | None = Field(
        default=None, description="DDL statements for training"
    )
    training_documentation: list[str] | None = Field(
        default=None, description="Documentation for training"
    )
    training_examples: list[dict[str, str]] | None = Field(
        default=None, description="Question-SQL examples for training"
    )
    initial_prompt: str | None = Field(
        default=None, description="Custom system prompt"
    )
    n_results: int = Field(default=5, description="Number of similar examples")
    sql_collection: str = Field(
        default="vanna_sql", description="Milvus collection for SQL examples"
    )
    ddl_collection: str = Field(
        default="vanna_ddl", description="Milvus collection for DDL"
    )
    doc_collection: str = Field(
        default="vanna_documentation", description="Milvus collection for docs"
    )


@register_function(
    config_type=Text2SQLConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def text2sql(config: Text2SQLConfig, builder: Builder):
    """Register the Text2SQL function with Vanna integration."""
    # Import implementation details inside the registration function
    from nat.plugins.vanna.db_utils import setup_vanna_db_connection
    from nat.plugins.vanna.milvus_utils import create_milvus_client
    from nat.plugins.vanna.vanna_utils import get_vanna_instance, train_vanna

    logger.info("Initializing Text2SQL function")

    # Get LLM and embedder
    llm_client = await builder.get_llm(
        config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    embedder_client = await builder.get_embedder(
        config.embedder_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )

    # Create Milvus clients
    milvus_client = create_milvus_client(
        host=config.milvus_host,
        port=config.milvus_port,
        user=config.milvus_user,
        password=config.milvus_password,
        db_name=config.milvus_db_name,
        use_tls=config.milvus_use_tls,
        is_async=False,
    )
    async_milvus_client = create_milvus_client(
        host=config.milvus_host,
        port=config.milvus_port,
        user=config.milvus_user,
        password=config.milvus_password,
        db_name=config.milvus_db_name,
        use_tls=config.milvus_use_tls,
        is_async=True,
    )

    # Initialize Vanna instance
    vanna_instance = await get_vanna_instance(
        llm_client=llm_client,
        embedder_client=embedder_client,
        milvus_client=milvus_client,
        async_milvus_client=async_milvus_client,
        initial_prompt=config.initial_prompt,
        n_results=config.n_results,
        sql_collection=config.sql_collection,
        ddl_collection=config.ddl_collection,
        doc_collection=config.doc_collection,
    )

    # Setup database connection if execute_sql is enabled
    if config.execute_sql:
        setup_vanna_db_connection(
            vn=vanna_instance,
            database_type=config.database_type,
            host=config.db_host or config.databricks_server_hostname,
            port=config.db_port,
            database=config.db_name,
            username=config.db_username,
            password=config.db_password or config.databricks_access_token,
            catalog=config.db_catalog,
            schema=config.db_schema,
            server_hostname=config.databricks_server_hostname,
            http_path=config.databricks_http_path,
            access_token=config.databricks_access_token,
        )

    # Train on startup if configured
    if config.train_on_startup:
        logger.info("Training Vanna on startup...")
        training_data = {
            "ddl": config.training_ddl or [],
            "documentation": config.training_documentation or [],
            "examples": config.training_examples or [],
        }
        await train_vanna(vanna_instance, training_data)

    # Streaming version
    async def _generate_sql_stream(
        question: str,
    ) -> AsyncGenerator[ResponseIntermediateStep | Text2SQLOutput, None]:
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

            sql = sql_result.get("sql", "")
            explanation = sql_result.get("explanation")

            # If execute_sql is enabled, run the query
            if config.execute_sql:
                yield ResponseIntermediateStep(
                    id=str(uuid.uuid4()),
                    parent_id=parent_id,
                    type="markdown",
                    name="text2sql_status",
                    payload=StatusPayload(message="Executing SQL query...").model_dump_json(),
                )
                try:
                    df = vanna_instance.run_sql(sql)
                    # Log execution success but don't change the output
                    logger.info(f"SQL executed successfully: {len(df)} rows returned")
                except Exception as e:
                    logger.error(f"SQL execution failed: {e}")
                    # Note: execution errors are logged but don't change the SQL output

            # Yield final result as Text2SQLOutput
            yield Text2SQLOutput(sql=sql, explanation=explanation)

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            # Error status as ResponseIntermediateStep
            yield ResponseIntermediateStep(
                id=str(uuid.uuid4()),
                parent_id=parent_id,
                type="markdown",
                name="text2sql_error",
                payload=StatusPayload(message=str(e)).model_dump_json(),
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

    description = (
        "Generate SQL queries from natural language questions using AI. "
        "Leverages similar question-SQL pairs, DDL information, and "
        "documentation to generate accurate SQL queries."
    )

    if config.execute_sql:
        description += " Also executes queries and returns results."

    try:
        yield FunctionInfo.create(
            single_fn=_generate_sql,
            stream_fn=_generate_sql_stream,
            description=description,
        )
    except GeneratorExit:
        logger.info("Text2SQL function exited early")
    finally:
        logger.info("Cleaning up Text2SQL function")
