import os
from enum import Enum
from typing import Any


class Flag(Enum):
    DYNAMIC_PROMPT = "dynamic_prompt"
    ENABLE_TABLEFILTER = "enable_tablefilter"
    VANNA_SQL_COLLECION = "vanna_sql_collection"
    VANNA_DOC_COLLECION = "vanna_doc_collection"
    VANNA_DDL_COLLECION = "vanna_ddl_collection"
    VANNA_COMMENT_COLLECION = "vanna_comment_collection"
    VANNA_REMOTE = "vanna_remote"  # Remote Vanna instance or local.
    MILVUS_VANNA_THRESH = "milvus_vanna_thresh"


# Define default values for each flag
DEFAULT_FLAG_VALUES = {
    Flag.DYNAMIC_PROMPT: True,
    Flag.ENABLE_TABLEFILTER: True,
    Flag.VANNA_SQL_COLLECION: "vannasql_v3_0",
    Flag.VANNA_DOC_COLLECION: "vannadoc_v3_0",
    Flag.VANNA_DDL_COLLECION: "vannaddl_v3_0",
    Flag.VANNA_COMMENT_COLLECION: "vannacomment_v3_0",
    Flag.VANNA_REMOTE: True,
    Flag.MILVUS_VANNA_THRESH: 2.0,
}


def get_flag_value(flag: Flag) -> Any:
    """Retrieve the value of a flag from the environment variable."""
    # Check environment variable
    env_var_name = f"FLAG_{flag.value.upper()}"
    env_value = os.getenv(env_var_name)
    if env_value is not None:
        return env_value

    # Fallback to default flag value
    return DEFAULT_FLAG_VALUES.get(flag)
