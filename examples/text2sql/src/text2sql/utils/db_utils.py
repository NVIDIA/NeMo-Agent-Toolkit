import asyncio
import logging
import os
import re
from typing import Any

import sqlglot
from databricks import sql
from databricks.sql.client import Connection
from sqlglot.expressions import Table

from text2sql.utils.db_schema import TTYSC_TABLES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_sql_from_tool_message(sql_query: str | Any) -> str:
    """Extract clean SQL query from various input formats.

    This function handles:
    1. Direct SQL strings (passes through unchanged)
    2. ReWOO tool message format where the entire ToolMessage is converted to string
       e.g., content="SELECT..." name='text2sql' tool_call_id='...'
    3. Objects with 'content' attribute

    Args:
        sql_query: The SQL query in various formats

    Returns:
        str: Clean SQL query string with unescaped quotes
    """
    # If it's not a string, try to get content attribute or convert to string
    if not isinstance(sql_query, str):
        if hasattr(sql_query, "content"):
            sql_query = sql_query.content
        else:
            sql_query = str(sql_query)

    # Handle ReWOO tool output format when placeholder is replaced with
    # entire tool message.
    # This happens when ReWOO's _replace_placeholder method converts the entire.
    # ToolMessage to string using str(tool_output) when replacing placeholders
    # in structured inputs.
    if (isinstance(sql_query, str) and 'content="' in sql_query and "tool_call_id=" in sql_query):
        # Extract just the SQL content from the tool message string
        # Look for content="..." pattern, handling escaped quotes
        match = re.search(r'content="((?:[^"\\]|\\.)*)"', sql_query)
        if match:
            sql_query = match.group(1)
            # Unescape any escaped quotes in the SQL
            sql_query = sql_query.replace("\\'", "'").replace('\\"', '"')

    return sql_query


def execute_query(connection: Connection | None, query: str) -> tuple[list[tuple[Any, ...]], list[str]]:
    """Execute a query on Databricks and return results."""
    try:
        if connection is None:
            _error_msg = "Connection is None"
            raise Exception(_error_msg)

        with connection.cursor() as cursor:
            logger.info(f"Executing query:\n{query}")
            cursor.execute(query)

            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            logger.info(f"Query completed, retrieved {len(results)} rows")
            return results, columns

    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise


def display_results(results, columns, max_rows=10):
    """Display query results in a formatted way."""
    if not results:
        logger.debug("No results returned")
        return

    # Show column headers
    logger.debug("\n" + "\t".join(columns))
    logger.debug("-" * 100)

    # Show rows
    for _, row in enumerate(results[:max_rows]):
        row_str = "\t".join(str(val) if val is not None else "NULL" for val in row)
        logger.debug(row_str)

    if len(results) > max_rows:
        logger.debug(f"\n... ({len(results) - max_rows} more rows)")


def connect_to_database(server_hostname: str | None, http_path: str | None,
                        access_token: str | None) -> Connection | None:
    if not server_hostname or not http_path or not access_token:
        msg_missingvalue = ("One of the db_variables (server_hostname, http_path, "
                            "access_token) is not set")
        raise Exception(msg_missingvalue)
    try:
        db_connection = sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token,
        )
        return db_connection
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        return None


def sync_query(query: str) -> tuple[list, list]:
    """Execute a query on Databricks and return results."""
    # TTYSC PAT Token
    server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    access_token = os.getenv("DATABRICKS_ACCESS_TOKEN")

    if not server_hostname or not http_path or not access_token:
        msg_missingvalue = ("One of the db_variables (server_hostname, http_path, "
                            "access_token) is not set")
        raise Exception(msg_missingvalue)
    try:
        with sql.connect(
                server_hostname=server_hostname,
                http_path=http_path,
                access_token=access_token,
        ) as connection:
            cursor = connection.cursor()
            logger.info(f"Executing query:\n{query}")
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            results = (rows, columns)
            logger.info(f"Query completed, retrieved {len(rows)} rows")
            return results

    except Exception as e:
        error_msg = f"Error executing query: {e}"
        logger.error(error_msg)
        raise


async def async_query(query: str) -> tuple[list, list]:
    """Execute a query on Databricks and return results."""
    return await asyncio.to_thread(sync_query, query)


def add_table_prefix(
    sql_query: str,
    prefix: str = "hive_metastore.silver_global_supply",
) -> str:
    """Add a prefix to all table names in a SQL query using sqlglot.

    Args:
        sql_query (str): The SQL query to modify
        prefix (str): The prefix to add to table names

    Returns:
        str: The modified SQL query with prefixed table names
    """
    # TODO(jiaxiangr): This is a temporary fix to avoid adding prefixes to CTEs
    return sql_query
    try:
        # Parse the SQL query
        parsed = sqlglot.parse_one(sql_query)

        # Extract CTE names to avoid adding prefixes to them
        cte_names = set()
        if parsed.args.get("with"):
            for cte in parsed.args["with"].expressions:
                if hasattr(cte, "alias") and cte.alias:
                    cte_names.add(str(cte.alias).lower())

        # Find all table references and add prefix if not already present
        def transform_table(node: sqlglot.Expression) -> sqlglot.Expression:
            if isinstance(node, Table):
                table_name = str(node.name).lower()

                # Skip CTEs - they shouldn't have database prefixes
                if table_name in cte_names:
                    return node

                current_db = str(node.db) if node.db else None
                current_catalog = str(node.catalog) if node.catalog else None

                # Parse the desired prefix
                prefix_parts = prefix.split(".")
                desired_catalog = prefix_parts[0] if len(prefix_parts) > 0 else None
                desired_schema = prefix_parts[1] if len(prefix_parts) > 1 else None

                # Check if table already has the exact prefix we want
                if current_catalog == desired_catalog and current_db == desired_schema:
                    logger.info("Table already has desired prefix.")
                elif current_catalog:
                    # Has existing catalog, leave as-is to avoid overriding
                    logger.info("Table has existing catalog, leaving as-is.")
                else:
                    # Add missing catalog and/or schema
                    node.set("catalog", desired_catalog)
                    if not current_db:
                        node.set("db", desired_schema)
                    logger.info("Added prefix to table.")

            return node

        # Transform the parsed query
        transformed = parsed.transform(transform_table)

        # Convert back to SQL string
        modified_query = transformed.sql(pretty=True)

        return modified_query

    except Exception as e:
        logger.error(f"Error processing SQL query: {e}")
        logger.error(f"Original query: {sql_query}")
        raise


# TODO(apourhabib): The default is currently PBR but should be DEMAND_DLT
# and that is the main use case
def infer_table_name_from_sql(sql_query: str) -> TTYSC_TABLES:
    """Infer the primary table name from a SQL query.

    Args:
        sql_query: The SQL query to analyze

    Returns:
        TTYSC_TABLES enum value for the primary table, defaults to PBR if uncertain
    """
    try:
        # Parse the SQL query
        parsed = sqlglot.parse_one(sql_query)

        # Extract all table names from the query
        table_names = []
        for table in parsed.find_all(Table):
            table_name = str(table.name).lower()
            # Remove any prefixes (schema.table_name -> table_name)
            if "." in table_name:
                table_name = table_name.split(".")[-1]
            table_names.append(table_name)

        logger.info(f"Extracted table names from SQL: {table_names}")

        # Map table names to TTYSC_TABLES enum
        for table_name in table_names:
            if "pbr" in table_name or "shortage" in table_name:
                return TTYSC_TABLES.PBR
            elif "supply" in table_name or "demand" in table_name:
                return TTYSC_TABLES.DEMAND_DLT

        # If no specific match found, check against enum values directly
        for table_name in table_names:
            for ttysc_table in TTYSC_TABLES:
                if table_name == ttysc_table.value:
                    return ttysc_table

        # Default to PBR if we can't determine the table
        # TODO(apourhabib): Default should be DEMAND_DLT instead of PBR
        # as it's the main use case
        logger.info(f"Could not infer table type from {table_names}, defaulting to PBR")
        return TTYSC_TABLES.PBR

    except Exception as e:
        logger.warning(f"Error parsing SQL for table inference: {e}. Defaulting to PBR")
        # TODO(apourhabib): Default should be DEMAND_DLT instead of PBR
        # as it's the main use case
        return TTYSC_TABLES.PBR
