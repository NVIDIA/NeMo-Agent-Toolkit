"""Registration of text2sql standalone functions for NeMo Agent Toolkit."""

from text2sql.functions.text2sql_standalone import Text2sqlStandaloneConfig
from text2sql.functions.text2sql_standalone import text2sql_standalone

__all__ = ["text2sql_standalone", "Text2sqlStandaloneConfig"]
