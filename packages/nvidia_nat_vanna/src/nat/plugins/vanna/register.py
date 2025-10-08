"""Register Vanna-based functions with NeMo Agent Toolkit.

This module registers text2sql and execute_db_query functions
for use in NAT workflows.
"""

# Import functions for automatic registration
from nat.plugins.vanna.execute_db_query import execute_db_query
from nat.plugins.vanna.text2sql import text2sql

__all__ = ["text2sql", "execute_db_query"]

