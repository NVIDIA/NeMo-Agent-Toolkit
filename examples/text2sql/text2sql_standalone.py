"""Standalone Text-to-SQL tool for MCP Server deployment and load testing.

This is a simplified version of text2sql designed for:
- Independent MCP server deployment
- Load testing and memory leak detection
- Minimal dependencies for easier profiling
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


class Text2sqlStandaloneConfig(FunctionBaseConfig, name="text2sql_standalone"):
    """Standalone Text2SQL configuration for MCP server."""

    _type: str = "text2sql_standalone"
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
        default=False,
        description="Whether to execute SQL or just return the query string",
    )
    authorize: bool = Field(
        default=False, description="Require Bearer token to run the function"
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
    config_type=Text2sqlStandaloneConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def text2sql_standalone(
    config: Text2sqlStandaloneConfig, builder: Builder
):  # noqa: ARG001
    """Register standalone Text2SQL function for MCP server deployment."""
    # Import implementation details inside the registration function
    from talk_to_supply_chain.functions.text2sql.sql_utils import (
        generate_sql_with_fallback,
        get_vanna_instance,
        train_vanna,
    )

    logger.info("🚀 Starting standalone text2sql for MCP server deployment")
    logger.info(
        f"Configuration: execute_sql={config.execute_sql}, "
        f"authorize={config.authorize}"
    )

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
        logger.info("ℹ️  Authorization disabled for standalone version")

    # Streaming version
    async def _generate_sql_stream(
        question: str,
        analysis_type: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream SQL generation progress and results.

        Args:
            question: Natural language question to convert to SQL
            analysis_type: Optional filter for analysis type
                          (e.g., 'pbr' or 'supply_gap') to retrieve
                          only relevant few-shot examples
        """
        logger.debug("=" * 50)
        logger.debug("🔍 TEXT2SQL_STANDALONE - generating SQL from natural language")
        logger.info(f"Input Query to text2sql_standalone: {question}")
        logger.debug("=" * 50)

        # Yield start status
        yield {
            "type": "status",
            "message": "Starting SQL generation...",
            "node": "text2sql_standalone",
        }

        try:
            # Yield SQL generation status
            yield {
                "type": "status",
                "message": "Generating SQL query...",
                "node": "text2sql_standalone",
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
                "node": "text2sql_standalone",
            }

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            yield {"type": "error", "message": str(e), "node": "text2sql_standalone"}
            raise

        logger.debug("✅ TEXT2SQL_STANDALONE - SQL generation completed successfully")
        logger.debug("=" * 50)

    # Non-streaming version
    async def _generate_sql(
        question: str,
        analysis_type: str | None = None,
    ) -> str:
        """Generate SQL query from natural language using Vanna.

        Args:
            question: Natural language question to convert to SQL
            analysis_type: Optional filter for analysis type
                          (e.g., 'pbr' or 'supply_gap') to retrieve
                          only relevant few-shot examples
        """
        result = None

        # Run streaming version and capture final result
        async for update in _generate_sql_stream(
            question, analysis_type=analysis_type
        ):
            if update["type"] == "result":
                result = update

        return result["sql_result"]

    # Create function description
    description = (
        "Standalone text-to-SQL tool for MCP server deployment and load testing. "
        "This tool generates SQL queries from natural language questions using "
        "AI. It leverages similar question-SQL pairs, DDL information, and "
        "documentation to generate accurate SQL queries. "
        "Designed for independent deployment and memory profiling."
    )

    try:
        yield FunctionInfo.create(
            single_fn=_generate_sql,
            stream_fn=_generate_sql_stream,
            description=description,
        )
    except GeneratorExit:
        logger.info("Text2SQL standalone function exited early!")
    finally:
        logger.info("Cleaning up Text2SQL standalone workflow...")

