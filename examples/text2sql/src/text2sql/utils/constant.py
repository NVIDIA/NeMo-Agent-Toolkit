"""Constants for the text2sql module."""

import os

# TODO: config remote vector database.
# Vector database path
VDB_PATH = os.environ.get("SUPPLY_CHAIN_VDB_PATH", "./milvus_vanna.db")

# Maximum limit size for queries
MAX_LIMIT_SIZE = 10_000

# Maximum length for Milvus fields
MILVUS_MAX_LEN = 65535

# Maximum number of rows to return from SQL query
MAX_SQL_ROWS = 2000
