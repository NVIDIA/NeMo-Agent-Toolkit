"""Text-to-SQL conversion function for NeMo Agent Toolkit.

This module provides the main NeMo Agent Toolkit registration for text-to-SQL
functionality using Vanna framework with NVIDIA NIM integration, including
authentication support.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import EmbedderRef, LLMRef
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field

logger = logging.getLogger(__name__)


class Text2sqlFunctionConfig(FunctionBaseConfig, name="text2sql"):
    """Text2SQL configuration function with auth support."""

    _type: str = "text2sql"
    llm_name: LLMRef = Field(description="Name of the LLM to use for SQL generation")
    embedder_name: EmbedderRef = Field(
        description="Name of the embedder to use for vector operations"
    )
    train_on_startup: bool = Field(
        default=False,
        description="Flag to train Vanna on startup with examples and DDL",
    )
    database_type: str = Field(
        default="databricks", description="Type of database to connect to"
    )
    allow_llm_to_see_data: bool = Field(
        default=False,
        description="Whether to allow LLM to see actual data for intermediate queries",
    )
    execute_sql: bool = Field(
        default=True,
        description="Whether to execute SQL or just return the query string",
    )
    authorize: bool = Field(
        default=True, description="Require Bearer token to run the function"
    )
    enable_followup_questions: bool = Field(
        default=False,
        description="Whether to generate follow-up questions along with SQL results",
    )
    vanna_remote: bool = Field(
        default=False,
        description="Whether to use remote Milvus cloud instance instead of local",
    )
    milvus_host: str | None = Field(
        default=None,
        description="Milvus cloud host URL (without https://)",
    )
    milvus_port: str | None = Field(
        default=None,
        description="Milvus cloud port",
    )
    milvus_user: str | None = Field(
        default=None,
        description="Milvus cloud username",
    )
    milvus_db_name: str | None = Field(
        default=None,
        description="Milvus database name",
    )
    training_analysis_filter: list[str] | None = Field(
        default=None,
        description=(
            "Filter examples by analysis type during training "
            "(e.g., ['pbr'], ['supply_gap'])"
        ),
    )


@register_function(
    config_type=Text2sqlFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def text2sql(config: Text2sqlFunctionConfig, builder: Builder):  # noqa: ARG001
    """Register the Text2SQL function with Vanna integration and authentication."""
    # Import implementation details inside the registration function
    from fastapi import HTTPException, status
    from nat.builder.context import Context

    from talk_to_supply_chain.functions.text2sql.sql_utils import (
        generate_sql_with_fallback,
        get_vanna_instance,
        train_vanna,
    )
    from talk_to_supply_chain.utils.auth import validate_token_with_helios

    logger.debug("Starting SQL generation...")

    # Get the configured LLM client from the builder
    llm_client = await builder.get_llm(
        config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )

    # Get the configured embedder client from the builder
    embedder_client = await builder.get_embedder(
        config.embedder_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )

    # Initialize Vanna instance on startup if needed
    vanna_instance = await get_vanna_instance(
        llm_client,
        embedder_client,
        config.vanna_remote,
        config.milvus_host,
        config.milvus_port,
        config.milvus_user,
        config.milvus_db_name,
        valid_analysis_types=config.training_analysis_filter,
    )

    # Train Vanna on startup if configured
    if config.train_on_startup:
        logger.info("Training Vanna on startup...")
        await train_vanna(
            vanna_instance,
            analysis_filter=config.training_analysis_filter,
        )

    if not config.authorize:
        logger.warning("Unauthorized API Endpoint")

    # Streaming version
    async def _generate_sql_stream(
        question: str,
        auth_token: str | None = None,
        analysis_type: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream SQL generation progress and results.

        Args:
            question: Natural language question to convert to SQL
            auth_token: Optional authentication token
            analysis_type: Optional filter for analysis type
                          (e.g., 'pbr' or 'supply_gap') to retrieve
                          only relevant few-shot examples
        """
        logger.debug("=" * 50)
        logger.debug("🔍 TEXT2SQL - generating SQL from natural language")
        logger.info(f"Input Query to text2sql: {question}")
        logger.debug("=" * 50)

        # Yield start status
        yield {
            "type": "status",
            "message": "Starting SQL generation...",
            "node": "text2sql",
        }

        # Auth flow
        if config.authorize:
            yield {
                "type": "status",
                "message": "Validating authentication...",
                "node": "text2sql",
            }

            authenticated = False

            if auth_token:
                # Use provided auth_token parameter
                authenticated = validate_token_with_helios(auth_token)
            else:
                # Fall back to getting auth token from headers
                headers = Context.get().metadata.headers
                header_token = headers.get("Authorization", None) if headers else None
                if header_token:
                    authenticated = validate_token_with_helios(header_token)

            if not authenticated:
                yield {
                    "type": "error",
                    "message": "Unauthorized access",
                    "node": "text2sql",
                }
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unauthorized",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        try:
            # Yield SQL generation status
            yield {
                "type": "status",
                "message": "Generating SQL query...",
                "node": "text2sql",
            }

            # Generate SQL using Vanna
            # Note: Invalid analysis_type values are logged and ignored
            # (graceful degradation)
            output_message = await generate_sql_with_fallback(
                question=question,
                allow_llm_to_see_data=config.allow_llm_to_see_data,
                execute_sql=config.execute_sql,
                enable_followup_questions=config.enable_followup_questions,
                analysis_type=analysis_type,
            )

            # Yield final result
            yield {
                "type": "result",
                "sql_result": output_message,
                "node": "text2sql",
            }

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            yield {"type": "error", "message": str(e), "node": "text2sql"}
            raise

        logger.debug("✅ TEXT2SQL - SQL generation completed successfully")
        logger.debug("=" * 50)

    # Non-streaming version
    async def _generate_sql(
        question: str,
        auth_token: str | None = None,
        analysis_type: str | None = None,
    ) -> str:
        """Generate SQL query from natural language using Vanna with auth.

        Args:
            question: Natural language question to convert to SQL
            auth_token: Optional authentication token
            analysis_type: Optional filter for analysis type
                          (e.g., 'pbr' or 'supply_gap') to retrieve
                          only relevant few-shot examples
        """
        result = None

        # Run streaming version and capture final result
        async for update in _generate_sql_stream(
            question, auth_token=auth_token, analysis_type=analysis_type
        ):
            if update["type"] == "result":
                result = update

        return result["sql_result"]

    # Create function description
    description = (
        "This tool generates SQL queries from natural language questions using "
        "AI. It leverages similar question-SQL pairs, DDL information, and "
        "documentation to generate accurate SQL queries."
    )

    if config.execute_sql:
        description += (
            " It also executes the queries through database connection and "
            "supports fallback on SQL error."
        )
    else:
        description += " It returns only the SQL query without executing it."

    if config.authorize:
        description += " Requires Bearer token authentication."

    try:
        yield FunctionInfo.create(
            single_fn=_generate_sql,
            stream_fn=_generate_sql_stream,
            description=description,
        )
    except GeneratorExit:
        logger.info("Text2SQL function exited early!")
    finally:
        logger.info("Cleaning up Text2SQL workflow...")
