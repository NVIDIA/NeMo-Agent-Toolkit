"""Mock Vanna instance for memory leak testing.

This module provides a lightweight mock of the Vanna interface
to help isolate memory leak sources during testing.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MockVannaInstance:
    """
    Mock Vanna instance that simulates the interface without Milvus connections.

    This mock helps identify if memory leaks are in:
    - Vanna/Milvus layer (leak disappears with mock)
    - Other parts of the system (leak persists with mock)
    """

    def __init__(self, **kwargs):
        """Initialize mock Vanna instance."""
        logger.info("Creating MockVannaInstance (no actual connections)")
        self.config = kwargs
        # Simulate having clients without actually creating them
        self.milvus_client = None
        self.async_milvus_client = None
        self._call_count = 0

    async def generate_sql(self, question: str, allow_llm_to_see_data: bool = False, **kwargs) -> str:
        """
        Mock SQL generation that returns a simple query.

        Args:
            question: The natural language question
            allow_llm_to_see_data: Whether to allow LLM to see data
            **kwargs: Additional arguments

        Returns:
            A mock SQL query string
        """
        self._call_count += 1
        logger.debug(f"MockVanna.generate_sql called (count: {self._call_count})")

        # Return a simple mock SQL query
        return f"SELECT * FROM mock_table WHERE question = '{question[:20]}...' LIMIT 10;"

    async def get_related_ddl(self, question: str, **kwargs) -> list[str]:
        """Mock method to get related DDL."""
        return ["CREATE TABLE mock_table (id INT, data TEXT);"]

    async def get_related_documentation(self, question: str, **kwargs) -> list[str]:
        """Mock method to get related documentation."""
        return ["Mock documentation for the table"]

    async def get_similar_question_sql(self, question: str, **kwargs) -> list[tuple]:
        """Mock method to get similar questions."""
        return [
            ("Similar question 1", "SELECT * FROM table1;"),
            ("Similar question 2", "SELECT * FROM table2;"),
        ]

    def train(self, **kwargs):
        """Mock training method."""
        logger.info("MockVanna.train called (no-op)")
        return True

    def add_question_sql(self, question: str, sql: str, **kwargs):
        """Mock method to add training data."""
        logger.debug(f"MockVanna.add_question_sql called: {question[:30]}")
        return True

    def add_ddl(self, ddl: str, **kwargs):
        """Mock method to add DDL."""
        logger.debug("MockVanna.add_ddl called")
        return True

    def add_documentation(self, doc: str, **kwargs):
        """Mock method to add documentation."""
        logger.debug("MockVanna.add_documentation called")
        return True

    async def run_sql(self, sql: str) -> Any:
        """Mock SQL execution."""
        logger.debug(f"MockVanna.run_sql called: {sql[:50]}")
        # Return mock results
        return {
            "columns": ["id", "data"],
            "rows": [[1, "mock_data_1"], [2, "mock_data_2"]],
        }

    def connect_to_databricks(self, **kwargs):
        """Mock Databricks connection."""
        logger.info("MockVanna.connect_to_databricks called (no-op)")
        return True

    def __repr__(self):
        return f"<MockVannaInstance calls={self._call_count}>"


async def get_mock_vanna_instance(**kwargs) -> MockVannaInstance:
    """
    Factory function to create a mock Vanna instance.

    This replaces get_vanna_instance() for testing purposes.

    Returns:
        MockVannaInstance that simulates Vanna without connections
    """
    logger.info("Creating mock Vanna instance for testing")
    return MockVannaInstance(**kwargs)


async def train_mock_vanna(vanna_instance: MockVannaInstance, **kwargs):
    """
    Mock training function.

    Args:
        vanna_instance: Mock Vanna instance
        **kwargs: Training arguments (ignored)
    """
    logger.info("Mock training called (no-op)")
    return True


async def generate_sql_with_mock(
    question: str,
    allow_llm_to_see_data: bool = False,
    execute_sql: bool = False,
    enable_followup_questions: bool = False,
    analysis_type: str | None = None,
    vanna_instance: MockVannaInstance | None = None,
) -> dict[str, Any]:
    """
    Mock SQL generation function that mimics the real one.

    Args:
        question: Natural language question
        allow_llm_to_see_data: Whether to show data to LLM
        execute_sql: Whether to execute the SQL
        enable_followup_questions: Whether to generate follow-ups
        analysis_type: Optional analysis type filter
        vanna_instance: Mock Vanna instance to use

    Returns:
        Dictionary with mock SQL results
    """
    if vanna_instance is None:
        vanna_instance = await get_mock_vanna_instance()

    logger.info(f"Generating mock SQL for: {question[:50]}...")

    # Generate mock SQL
    sql = await vanna_instance.generate_sql(
        question=question,
        allow_llm_to_see_data=allow_llm_to_see_data,
    )

    result = {
        "sql": sql,
        "explanation": f"Mock SQL generated for: {question}",
        "confidence": "high",
        "method": "mock_vanna",
        "mock_mode": True,
    }

    # Mock execution if requested
    if execute_sql:
        mock_data = await vanna_instance.run_sql(sql)
        result["results"] = {
            "row_count": len(mock_data.get("rows", [])),
            "columns": mock_data.get("columns", []),
            "data": mock_data.get("rows", []),
        }

    # Mock follow-up questions if requested
    if enable_followup_questions:
        result["followup_questions"] = [
            "Mock follow-up question 1?",
            "Mock follow-up question 2?",
            "Mock follow-up question 3?",
        ]

    return result
