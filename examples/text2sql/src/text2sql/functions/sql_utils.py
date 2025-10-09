"""Vanna-based Text-to-SQL implementation utilities.

This module contains all the Vanna framework integration logic, database connections,
training utilities, and helper functions for SQL generation.
"""

import asyncio
import contextlib
import json
import logging
import os
import re
import uuid

import pandas as pd
from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi import status
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from pymilvus import DataType
from pymilvus import MilvusClient
from vanna.base import VannaBase
from vanna.milvus import Milvus_VectorStore
from vanna.utils import deterministic_uuid

from text2sql.resources.followup_resources import FOLLOWUP_GUIDELINES
from text2sql.resources.followup_resources import TABLE_USE_CASES
from text2sql.utils.constant import MAX_LIMIT_SIZE
from text2sql.utils.constant import MAX_SQL_ROWS
from text2sql.utils.constant import MILVUS_MAX_LEN
from text2sql.utils.db_schema import DEMAND_DLT_EXAMPLES
from text2sql.utils.db_schema import INSTRUCTION_PROMPT
from text2sql.utils.db_schema import PBR_EXAMPLES
from text2sql.utils.db_schema import RESPONSE_GUIDELINES
from text2sql.utils.db_schema import TABLES
from text2sql.utils.db_schema import TTYSC_TABLES
from text2sql.utils.db_schema import generate_table_description
from text2sql.utils.db_utils import add_table_prefix
from text2sql.utils.db_utils import async_query
from text2sql.utils.db_utils import infer_table_name_from_sql
from text2sql.utils.feature_flag import Flag
from text2sql.utils.feature_flag import get_flag_value
from text2sql.utils.milvus_utils import create_milvus_client

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Global instances for singleton pattern
_vanna_instance = None
_init_lock = None


def validate_analysis_type(analysis_type: str | None, valid_types: list[str] | None = None) -> str | None:
    """Validate and sanitize analysis_type parameter.

    Args:
        analysis_type: The analysis type to validate
        valid_types: List of valid analysis type strings from config
                    (e.g., from training_analysis_filter in config.yml)

    Returns:
        Validated analysis_type or None if invalid/not provided.
        If invalid, logs error and returns None to allow query without filtering.

    Note:
        This function does NOT raise exceptions. Invalid values are logged
        and the function returns None, allowing the workflow to continue
        without filtering rather than failing.
    """
    if analysis_type is None or analysis_type == "":
        return None

    # If no valid types configured, filtering is disabled
    if not valid_types or len(valid_types) == 0:
        if analysis_type:
            logger.warning(f"analysis_type '{analysis_type}' provided but "
                           "training_analysis_filter not configured - filtering disabled")
        return None

    # Convert valid types to set and normalize to lowercase
    valid_types_set = {vtype.lower() for vtype in valid_types}

    # Strip whitespace and convert to lowercase for case-insensitive comparison
    cleaned_type = analysis_type.strip().lower()

    if cleaned_type not in valid_types_set:
        error_msg = (f"Invalid analysis_type: '{analysis_type}'. "
                     f"Must be one of: {', '.join(sorted(valid_types_set))}. "
                     "Continuing without filtering.")
        logger.error(error_msg)
        return None  # Return None instead of raising, allowing graceful degradation

    return cleaned_type


def build_sql_result(
    sql: str,
    question: str,
    rows: list,
    columns: list,
    records: list,
    followup_questions: list | None = None,
    confidence: str = "high",
    method: str = "vanna",
    original_error: str | None = None,
) -> str:
    """Build a standardized SQL result dictionary and return as JSON string.

    Args:
        sql: The SQL query that was executed
        question: The original user question
        rows: List of result rows
        columns: List of column names
        records: List of record dictionaries
        followup_questions: Optional list of follow-up questions
        confidence: Confidence level ("high", "medium", "low")
        method: Method used ("vanna", "vanna_retry", etc.)
        original_error: Optional error message for retry scenarios

    Returns:
        JSON string representation of the result
    """
    # Build base result dictionary
    result = {
        "sql": sql,
        "explanation": (f"{'Regenerated' if method == 'vanna_retry' else 'Generated'} "
                        f"SQL query for: {question}"),
        "confidence": confidence,
        "method": method,
        "results": {
            "row_count": len(rows),
            "column_count": len(columns),
            "columns": columns,
            "data": records[:MAX_SQL_ROWS] if records else [],
            "truncated": len(records) > MAX_SQL_ROWS,
        },
    }

    # Add original error for retry scenarios
    if original_error is not None:
        result["original_error"] = original_error

    # Add follow-up questions if provided
    if followup_questions:
        # Ensure followup questions are properly formatted for JSON
        cleaned_questions = []
        for q in followup_questions:
            if isinstance(q, str):
                # Remove any problematic characters and normalize whitespace
                cleaned_q = re.sub(r"\s+", " ", str(q).strip())
                if cleaned_q:
                    cleaned_questions.append(cleaned_q)
        result["followup_questions"] = cleaned_questions

    return json.dumps(result)


class VannaChatNVIDIA(VannaBase):
    """NVIDIA NIM integration for Vanna framework."""

    def __init__(self, client=None, config=None):
        if config is None:
            msg = "For VannaChatNVIDIA, llm must be provided with an api_key and model"
            raise ValueError(msg)
        if "model" not in config:
            msg = "config must contain a ChatNVIDIA model"
            raise ValueError(msg)

        self.client = client
        self.model = config["model"]

    def system_message(self, message: str) -> dict:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> dict:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> dict:
        return {"role": "assistant", "content": message}

    def get_sql_prompt(
        self,
        initial_prompt: str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        error_message: dict | None = None,
    ) -> list:
        """Generate a prompt for the LLM to generate SQL."""
        if initial_prompt is None:
            initial_prompt = (f"You are a {self.dialect} expert. "
                              "Please help to generate a SQL query to answer the question. "
                              "Your response should ONLY be based on the given context "
                              "and follow the response guidelines and format instructions. ")

        initial_prompt = self.add_ddl_to_prompt(initial_prompt, ddl_list, max_tokens=self.max_tokens)

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(initial_prompt, doc_list, max_tokens=self.max_tokens)

        initial_prompt += RESPONSE_GUIDELINES
        initial_prompt += (f"5. Ensure that the output SQL is {self.dialect}-compliant "
                           "and executable, and free of syntax errors. \n")

        # If there is an error in previously generated SQL, add it to the prompt
        if error_message is not None:
            initial_prompt += (f"6. For question: {question}. "
                               "\tPrevious SQL attempt failed with error: "
                               f"{error_message['sql_error']}\n"
                               f"\tPrevious SQL was: {error_message['previous_sql']}\n"
                               f"\tPlease fix the SQL syntax/logic error and regenerate.")

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                logger.info("example is None")
            else:
                if "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))
        return message_log

    async def submit_prompt(self, prompt) -> str:
        response = await self.client.ainvoke(str(prompt))
        return response.content

    async def generate_sql(
        self,
        question: str,
        allow_llm_to_see_data=False,
        error_message: dict | None = None,
        analysis_type: str | None = None,
        **kwargs,
    ) -> str:
        """Generate SQL using the LLM.

        Args:
            question: Natural language question to convert to SQL
            allow_llm_to_see_data: Whether to allow LLM to see actual data
            error_message: Optional error message from previous SQL execution
            analysis_type: Optional filter for analysis type
                          (e.g., 'pbr' or 'supply_gap').
                          Invalid values are logged and ignored.
            **kwargs: Additional keyword arguments
        """
        logger.info("Starting SQL Generation with Vanna")

        # Validate and sanitize analysis_type for security
        # Note: Invalid values return None and log error, allowing graceful degradation
        validated_analysis_type = validate_analysis_type(analysis_type, valid_types=self.valid_analysis_types)

        if validated_analysis_type:
            logger.info(f"Generating SQL with analysis_type filter: {validated_analysis_type}")
        elif analysis_type and not validated_analysis_type:
            # Log when an invalid type was provided and ignored
            logger.error(f"Invalid analysis_type '{analysis_type}' - proceeding without filter")
        else:
            logger.info("Generating SQL (no analysis_type filter)")

        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None

        retrieval_in_parallel = []
        retrieval_in_parallel.append(
            self.get_similar_question_sql(question, analysis_type=validated_analysis_type, **kwargs))
        retrieval_in_parallel.append(self.get_related_ddl(question, **kwargs))
        retrieval_in_parallel.append(self.get_related_documentation(question, **kwargs))

        # TODO: Commented out SQL comments retrieval - currently it is empty
        # retrieval_in_parallel.append(
        #     self.get_similar_sql_comments(
        #         question, analysis_type=validated_analysis_type, **kwargs
        #     )
        # )

        question_sql_list, ddl_list, doc_list = await asyncio.gather(*retrieval_in_parallel)

        # SQL comments are disabled for now
        sql_comments = []

        # Note: SQL comments retrieval is currently disabled
        if sql_comments:
            doc_list.append("\n===Few-Shot Examples with Reasoning\n")
            for comment in sql_comments:
                doc_list.append(f"Question: {comment['question']}\n"
                                f"SQL: {comment['sql']}\n"
                                f"Reasoning: {comment['explanation']}\n")

        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            error_message=error_message,
            **kwargs,
        )

        try:
            response = await self.submit_prompt(prompt, **kwargs)
            llm_response = response
            self.log(title="LLM Response", message=llm_response)
        except Exception as e:
            logger.error(f"Error calling LLM during SQL query generation: {e}")
            error_message = "Error calling LLM during SQL query generation"
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_message,
            ) from e

        if "intermediate_sql" in llm_response:
            if not allow_llm_to_see_data:
                return ("The LLM is not allowed to see the data in your database. "
                        "Your question requires database introspection to generate "
                        "the necessary SQL. Please set allow_llm_to_see_data=True "
                        "to enable this.")

            intermediate_sql = self.extract_sql(llm_response)

            try:
                self.log(title="Running Intermediate SQL", message=intermediate_sql)
                df = self.run_sql(intermediate_sql)

                prompt = self.get_sql_prompt(
                    initial_prompt=initial_prompt,
                    question=question,
                    question_sql_list=question_sql_list,
                    ddl_list=ddl_list,
                    doc_list=doc_list + [
                        "The following is a pandas DataFrame with the results of "
                        f"the intermediate SQL query {intermediate_sql}: \n" + df.to_markdown()
                    ],
                    **kwargs,
                )
                response = await self.submit_prompt(prompt, **kwargs)
                llm_response = response
                self.log(title="LLM Response", message=llm_response)
            except Exception as e:
                return f"Error running intermediate SQL: {e}"

        sql = self.extract_sql(llm_response)
        return sql.replace("\\_", "_")

    def get_followup_questions_prompt(
        self,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        table_preview: str | None = None,
        **kwargs,
    ) -> list:
        """Generate a prompt for the LLM to generate follow-up questions."""
        # Get table-specific use cases for follow-up questions
        table_name = kwargs.get("table_name", TTYSC_TABLES.PBR)
        use_cases = TABLE_USE_CASES.get(table_name, TABLE_USE_CASES[TTYSC_TABLES.PBR])
        use_cases_text = "\n".join([f"- {case}" for case in use_cases])

        # Format the follow-up guidelines with the actual values
        formatted_prompt = FOLLOWUP_GUIDELINES.format(
            user_prompt=question,
            table_preview=table_preview or "No data preview available",
            use_cases_text=use_cases_text,
        )

        # Add DDL context
        initial_prompt = self.add_ddl_to_prompt(formatted_prompt, ddl_list, max_tokens=self.max_tokens)

        # Add documentation context
        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(initial_prompt, doc_list, max_tokens=self.max_tokens)

        # Add SQL examples context
        initial_prompt = self.add_sql_to_prompt(initial_prompt, question_sql_list, max_tokens=self.max_tokens)

        message_log = [self.system_message(initial_prompt)]
        message_log.append(
            self.user_message("Generate a list of followup questions that the user might ask "
                              "about this data. Respond with a list of questions, one per line. "
                              "Do not answer with any explanations -- just the questions."))

        return message_log

    # TODO(apourhabib): This is repetitive and I need to import it from followup tool.
    def _validate_questions(self, questions_text: str, applicable_use_cases: list[str], user_prompt: str) -> str:
        """Validate that generated questions align with supported use cases.

        Args:
            questions_text: The raw generated questions
            applicable_use_cases: List of supported use cases
            user_prompt: Original user prompt for fallback questions

        Returns:
            Validated and potentially filtered questions
        """
        if not questions_text:
            return self._generate_fallback_questions(applicable_use_cases, user_prompt)

        lines = questions_text.strip().split("\n")
        validated_questions = []

        # Forbidden patterns that indicate unsupported question types
        forbidden_patterns = [
            r"average.*over.*\d+\s*(months?|weeks?|days?)",  # time-based averages
            r"(last|past|previous)\s*\d+\s*(months?|weeks?|days?)",  # time periods
            r"which\s+dates?",  # date-specific queries
            r"on\s+what\s+date",  # date-specific queries
            r"trend.*over.*time",  # complex trend analysis
            r"compare.*across.*time",  # time-based comparisons
            r"when.*will.*be",  # future predictions
            r"forecast",  # forecasting
            r"predict",  # predictions
        ]

        for line in lines:
            line = line.strip()
            if not line or not re.match(r"^\d+\.", line):
                continue

            # Check if the question contains forbidden patterns
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in forbidden_patterns):
                continue

            # Check if question matches use cases or parameter substitution
            if self._is_valid_question(line, applicable_use_cases, user_prompt):
                validated_questions.append(line)

        # If we don't have enough valid questions, generate fallbacks
        while len(validated_questions) < 4:
            fallback = self._get_fallback_question(len(validated_questions) + 1, applicable_use_cases, user_prompt)
            if fallback:
                validated_questions.append(fallback)
            else:
                break

        return ("\n".join(validated_questions) if validated_questions else self._generate_fallback_questions(
            applicable_use_cases, user_prompt))

    def _is_valid_question(self, question: str, applicable_use_cases: list[str], user_prompt: str) -> bool:
        """Check if question is valid by comparing to use cases and user prompt."""
        question_lower = question.lower()
        user_prompt_lower = user_prompt.lower()

        # Check if it's a simple parameter substitution of the user prompt
        if self._is_parameter_substitution(question_lower, user_prompt_lower):
            return True

        # Check if it matches patterns from applicable use cases
        for use_case in applicable_use_cases:
            if self._question_matches_use_case_pattern(question_lower, use_case.lower()):
                return True

        return False

    def _is_parameter_substitution(self, question: str, user_prompt: str) -> bool:
        """Check if question is a simple parameter substitution of user prompt."""
        # Remove question numbers
        question_clean = re.sub(r"^\d+\.\s*", "", question)

        # Define common parameter patterns
        nvpn_pattern = (r"(?:[A-Z]{3}\d{6}|\d{3}[-\.][A-Z0-9]{4,5}[-\.][A-Z0-9]{3,4}"
                        r"(?:\.[A-Z])?|\d{2,3}\.[\dA-Z\.]{4,})")
        cm_pattern = r"[A-Z0-9_]{3,}"
        sku_pattern = r"\d{3}-[A-Z0-9]{5}-\d{4}-[A-Z0-9]{3}"

        # Replace parameters with placeholders for comparison
        question_normalized = re.sub(nvpn_pattern, "NVPN_PLACEHOLDER", question_clean)
        question_normalized = re.sub(cm_pattern, "CM_PLACEHOLDER", question_normalized)
        question_normalized = re.sub(sku_pattern, "SKU_PLACEHOLDER", question_normalized)

        user_normalized = re.sub(nvpn_pattern, "NVPN_PLACEHOLDER", user_prompt)
        user_normalized = re.sub(cm_pattern, "CM_PLACEHOLDER", user_normalized)
        user_normalized = re.sub(sku_pattern, "SKU_PLACEHOLDER", user_normalized)

        # Calculate similarity (simple word overlap)
        question_words = set(question_normalized.split())
        user_words = set(user_normalized.split())
        overlap = len(question_words.intersection(user_words))
        total_words = len(question_words.union(user_words))

        return overlap / total_words > 0.6 if total_words > 0 else False

    def _question_matches_use_case_pattern(self, question: str, use_case: str) -> bool:
        """Check if the question follows a pattern from the use cases."""
        # Remove question numbers
        question_clean = re.sub(r"^\d+\.\s*", "", question)

        # Extract key pattern words from use case
        use_case_words = set(use_case.split())
        question_words = set(question_clean.split())

        # Check for significant word overlap with use case patterns
        overlap = len(use_case_words.intersection(question_words))
        return overlap >= 3  # At least 3 words should match

    def _get_fallback_question(self, question_num: int, applicable_use_cases: list[str],
                               _user_prompt: str) -> str | None:
        """Generate a safe fallback question."""
        if not applicable_use_cases:
            return None

        # Use the first applicable use case as a safe fallback
        fallback_use_case = applicable_use_cases[0]
        return f"{question_num}. {fallback_use_case}"

    def _generate_fallback_questions(self, applicable_use_cases: list[str], _user_prompt: str) -> str:
        """Generate safe fallback questions when validation fails."""
        if not applicable_use_cases:
            return "No supported follow-up questions available."

        fallback_questions = []
        for i, use_case in enumerate(applicable_use_cases[:4], 1):
            fallback_questions.append(f"{i}. {use_case}")

        return "\n".join(fallback_questions)


class MilvusVectorStore(Milvus_VectorStore):
    """Extended Milvus vector store with SQL comment support."""

    def __init__(self, config=None):
        try:
            VannaBase.__init__(self, config=config)

            self.milvus_client = config["milvus_client"]
            self.async_milvus_client = config["async_milvus_client"]

            # Store valid analysis types from config for validation
            self.valid_analysis_types = config.get("valid_analysis_types", None)

            self.n_results_sql_comment = config.get("n_results_sql_comment", config.get("n_results", 5))
            self.n_results = config.get("n_results", 5)

            # Use configured embedder if provided, otherwise fallback to default
            if config.get("embedder_client") is not None:
                logger.info("Using configured embedder client")
                self.embedder = config["embedder_client"]
            else:
                logger.info("No embedder client provided, using default embedder")
                self.embedder = get_embedder()

            try:
                self._embedding_dim = len(self.embedder.embed_documents(["foo"])[0])
                logger.info(f"embedding_dim: {self._embedding_dim}")
            except Exception as e:
                logger.error(f"Error calling embedder client during Milvus initialization: {e}")
                error_msg = "Error calling embedder client during Milvus initialization"
                raise Exception(error_msg) from e

            self._create_collections()
        except Exception as e:
            logger.error(f"Error initializing MilvusVectorStore: {e}")
            raise

    def _check_collections_empty(self) -> bool:
        """Check if all collections are empty."""
        collections_empty = True
        for coll in [
                get_flag_value(Flag.VANNA_SQL_COLLECION),
                get_flag_value(Flag.VANNA_DDL_COLLECION),
                get_flag_value(Flag.VANNA_DOC_COLLECION),
                get_flag_value(Flag.VANNA_COMMENT_COLLECION),
        ]:
            if self.milvus_client.has_collection(collection_name=coll):
                stats = self.milvus_client.get_collection_stats(collection_name=coll)
                if int(stats["row_count"]) > 0:
                    collections_empty = False
                    return collections_empty
        return collections_empty

    def _create_collections(self):
        """Create all necessary Milvus collections."""
        self._create_sql_collection(get_flag_value(Flag.VANNA_SQL_COLLECION))
        self._create_ddl_collection(get_flag_value(Flag.VANNA_DDL_COLLECION))
        self._create_doc_collection(get_flag_value(Flag.VANNA_DOC_COLLECION))
        self._create_sqlcomment_collection(get_flag_value(Flag.VANNA_COMMENT_COLLECION))

    def _create_sql_collection(self, name: str):
        """Create SQL collection with analysis metadata field."""
        if not self.milvus_client.has_collection(collection_name=name):
            vannasql_schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            vannasql_schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=MILVUS_MAX_LEN,
            )
            vannasql_schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                max_length=MILVUS_MAX_LEN,
            )
            vannasql_schema.add_field(
                field_name="sql",
                datatype=DataType.VARCHAR,
                max_length=MILVUS_MAX_LEN,
            )
            vannasql_schema.add_field(
                field_name="analysis",
                datatype=DataType.VARCHAR,
                max_length=50,
            )
            vannasql_schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self._embedding_dim,
            )

            vannasql_index_params = MilvusClient.prepare_index_params()
            vannasql_index_params.add_index(
                field_name="vector",
                index_type="AUTOINDEX",
                metric_type="L2",
            )
            self.milvus_client.create_collection(
                collection_name=name,
                schema=vannasql_schema,
                index_params=vannasql_index_params,
                consistency_level="Strong",
            )

    def _create_sqlcomment_collection(self, name: str):
        """Create SQL comment collection."""
        if not self.milvus_client.has_collection(collection_name=name):
            vannacomment_schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            vannacomment_schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                max_length=MILVUS_MAX_LEN,
                is_primary=True,
            )
            vannacomment_schema.add_field(
                field_name="question",
                datatype=DataType.VARCHAR,
                max_length=MILVUS_MAX_LEN,
            )
            vannacomment_schema.add_field(field_name="sql", datatype=DataType.VARCHAR, max_length=MILVUS_MAX_LEN)
            vannacomment_schema.add_field(
                field_name="explanation",
                datatype=DataType.VARCHAR,
                max_length=MILVUS_MAX_LEN,
            )
            vannacomment_schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self._embedding_dim,
            )

            vannacomment_index_params = self.milvus_client.prepare_index_params()
            vannacomment_index_params.add_index(
                field_name="vector",
                index_name="vector",
                index_type="AUTOINDEX",
                metric_type="L2",
            )
            self.milvus_client.create_collection(
                collection_name=name,
                schema=vannacomment_schema,
                index_params=vannacomment_index_params,
                consistency_level="Strong",
            )

    def add_question_sql(self, question: str, sql: str, analysis: str | None = None) -> str:
        """Add question-SQL pair to collection with optional analysis metadata."""
        if len(question) == 0 or len(sql) == 0:
            msg = "pair of question and sql can not be null"
            raise Exception(msg)
        _id = str(uuid.uuid4()) + "-sql"
        embedding = self.embedder.embed_documents([question])[0]
        data = {
            "id": _id,
            "text": question,
            "sql": sql,
            "analysis": analysis if analysis else "",
            "vector": embedding,
        }
        self.milvus_client.insert(
            collection_name=get_flag_value(Flag.VANNA_SQL_COLLECION),
            data=data,
        )
        return _id

    def add_ddl(self, ddl: str) -> str:
        """Add DDL to collection."""
        logger.info(f"add_ddl: {ddl}")
        if len(ddl) == 0:
            msg = "ddl can not be null"
            raise Exception(msg)
        _id = str(uuid.uuid4()) + "-ddl"
        embedding = self.embedder.embed_documents([ddl])[0]
        self.milvus_client.insert(
            collection_name=get_flag_value(Flag.VANNA_DDL_COLLECION),
            data={
                "id": _id, "ddl": ddl, "vector": embedding
            },
        )
        return _id

    def add_documentation(self, documentation: str) -> str:
        """Add documentation to collection."""
        if len(documentation) == 0:
            msg = "documentation can not be null"
            raise Exception(msg)
        _id = str(uuid.uuid4()) + "-doc"
        embedding = self.embedder.embed_documents([documentation])[0]
        self.milvus_client.insert(
            collection_name=get_flag_value(Flag.VANNA_DOC_COLLECION),
            data={
                "id": _id, "doc": documentation, "vector": embedding
            },
        )
        return _id

    def add_sql_comment(self, question: str, sql: str, explanation: str) -> str:
        """Add SQL comment to collection."""
        sql_comment_json = json.dumps(
            {
                "question": question, "sql": sql, "explanation": explanation
            },
            ensure_ascii=False,
        )

        _id = deterministic_uuid(sql_comment_json) + "-sql-comment"
        embedding = self.embedder.embed_documents([question])[0]
        self.milvus_client.insert(
            collection_name=get_flag_value(Flag.VANNA_COMMENT_COLLECION),
            data={
                "id": _id,
                "question": question,
                "sql": sql,
                "explanation": explanation,
                "vector": embedding,
            },
        )
        return _id

    async def get_related_ddl(self, question: str, **kwargs) -> list:  # noqa: ARG002
        """Retrieve related DDL."""
        list_ddl = []
        try:
            res = await self.async_milvus_client.query(
                collection_name=get_flag_value(Flag.VANNA_DDL_COLLECION),
                output_fields=["ddl"],
                limit=MAX_LIMIT_SIZE,
            )
            for doc in res:
                list_ddl.append(doc["ddl"])
        except Exception as e:
            logger.error(f"Error during milvus client query in get_related_ddl: {e}")
        return list_ddl

    async def get_related_documentation(self, question: str) -> list:  # noqa: ARG002
        """Retrieve related documents."""
        list_doc = []
        try:
            res = await self.async_milvus_client.query(
                collection_name=get_flag_value(Flag.VANNA_DOC_COLLECION),
                output_fields=["doc"],
                limit=MAX_LIMIT_SIZE,
            )
            for doc in res:
                list_doc.append(doc["doc"])
        except Exception as e:
            logger.error(f"Error during milvus client query in get_related_documentation: {e}")
        return list_doc

    async def get_similar_question_sql(
            self,
            question: str,
            analysis_type: str | None = None,
            **kwargs,  # noqa: ARG002
    ) -> list:
        """Get similar question-SQL pairs, optionally filtered.

        Args:
            question: The question to search for similar examples
            analysis_type: Optional filter for analysis type
                          (e.g., 'pbr' or 'supply_gap').
                          NOTE: This should be pre-validated.
            **kwargs: Additional keyword arguments

        Returns:
            List of similar question-SQL pairs with their analysis type
        """
        search_params = {
            "metric_type": "L2",
            "params": {
                "nprobe": 128
            },
        }
        list_sql = []
        try:
            embeddings = [await self.embedder.aembed_query(question)]

            # Build filter expression if analysis_type is provided
            # NOTE: analysis_type should already be validated
            # to prevent Milvus filter injection attacks
            filter_expr = None
            if analysis_type:
                filter_expr = f'analysis == "{analysis_type}"'
                logger.debug(f"Applying Milvus filter for similar_question_sql: {filter_expr}")

            search_kwargs = {
                "collection_name": get_flag_value(Flag.VANNA_SQL_COLLECION),
                "anns_field": "vector",
                "data": embeddings,
                "limit": self.n_results,
                "output_fields": ["text", "sql", "analysis"],
                "search_params": search_params,
            }

            if filter_expr:
                search_kwargs["filter"] = filter_expr

            res = await self.async_milvus_client.search(**search_kwargs)
            res = res[0]
            thresh = get_flag_value(Flag.MILVUS_VANNA_THRESH)
            for doc in res:
                score = doc.get("distance", 1.0)
                if isinstance(thresh, float) and score > thresh:
                    continue
                dict = {}
                dict["question"] = doc["entity"]["text"]
                dict["sql"] = doc["entity"]["sql"]
                dict["analysis"] = doc["entity"].get("analysis", "")
                list_sql.append(dict)
            if len(list_sql) == 0 and len(res) > 0:
                doc = res[0]
                dict = {}
                dict["question"] = doc["entity"]["text"]
                dict["sql"] = doc["entity"]["sql"]
                dict["analysis"] = doc["entity"].get("analysis", "")
                list_sql.append(dict)

            logger.info(f"Retrieved {len(list_sql)} similar SQL examples" +
                        (f" (filtered by {analysis_type})" if analysis_type else ""))

            # Warn if filtering resulted in no examples
            if analysis_type and len(list_sql) == 0:
                logger.warning(f"No SQL examples found with filter: analysis={analysis_type}. "
                               "SQL generation may be less accurate.")
        except Exception as e:
            logger.error(
                f"Error during milvus client search in get_similar_question_sql: {e}",
                exc_info=True,
            )
        return list_sql

    async def get_similar_sql_comments(
            self,
            question: str,
            analysis_type: str | None = None,
            **kwargs,  # noqa: ARG002
    ) -> list:
        """Get similar SQL comments, optionally filtered.

        Args:
            question: The question to search for similar examples
            analysis_type: Optional filter for analysis type
                          (e.g., 'pbr' or 'supply_gap').
                          NOTE: This should be pre-validated.
            **kwargs: Additional keyword arguments

        Returns:
            List of similar SQL comments with explanations
        """
        search_params = {
            "metric_type": "L2",
            "params": {
                "nprobe": 128
            },
        }
        list_sql = []
        try:
            embeddings = [await self.embedder.aembed_query(question)]

            # Build filter expression if analysis_type is provided
            # NOTE: analysis_type should already be validated
            # to prevent Milvus filter injection attacks
            filter_expr = None
            if analysis_type:
                filter_expr = f'analysis == "{analysis_type}"'
                logger.debug(f"Applying Milvus filter for similar_sql_comments: {filter_expr}")

            search_kwargs = {
                "collection_name": get_flag_value(Flag.VANNA_COMMENT_COLLECION),
                "anns_field": "vector",
                "data": embeddings,
                "limit": self.n_results_sql_comment,
                "output_fields": ["question", "sql", "explanation", "analysis"],
                "search_params": search_params,
            }

            if filter_expr:
                search_kwargs["filter"] = filter_expr

            res = await self.async_milvus_client.search(**search_kwargs)
            res = res[0]
            thresh = get_flag_value(Flag.MILVUS_VANNA_THRESH)
            for doc in res:
                score = doc.get("distance", 1.0)
                if isinstance(thresh, float) and score > thresh:
                    continue
                dict = {}
                dict["question"] = doc["entity"]["question"]
                dict["sql"] = doc["entity"]["sql"]
                dict["explanation"] = doc["entity"]["explanation"]
                dict["analysis"] = doc["entity"].get("analysis", "")
                list_sql.append(dict)
            if len(list_sql) == 0 and len(res) > 0:
                doc = res[0]
                dict = {}
                dict["question"] = doc["entity"]["question"]
                dict["sql"] = doc["entity"]["sql"]
                dict["explanation"] = doc["entity"]["explanation"]
                dict["analysis"] = doc["entity"].get("analysis", "")
                list_sql.append(dict)

            logger.info(f"Retrieved {len(list_sql)} similar SQL comments" +
                        (f" (filtered by {analysis_type})" if analysis_type else ""))

            # Warn if filtering resulted in no examples
            if analysis_type and len(list_sql) == 0:
                logger.warning(f"No SQL comment examples found with filter: "
                               f"analysis={analysis_type}. "
                               "SQL generation may be less accurate.")
        except Exception as e:
            logger.error(
                f"Error during milvus client search in get_similar_sql_comments: {e}",
                exc_info=True,
            )
        return list_sql

    def get_training_data(self) -> pd.DataFrame:
        """Get all training data."""
        sql_data = self.milvus_client.query(
            collection_name=get_flag_value(Flag.VANNA_SQL_COLLECION),
            output_fields=["*"],
            limit=MAX_LIMIT_SIZE,
        )
        df = pd.DataFrame()
        df_sql = pd.DataFrame({
            "id": [doc["id"] for doc in sql_data],
            "question": [doc["text"] for doc in sql_data],
            "content": [doc["sql"] for doc in sql_data],
        })
        df_sql["training_data_type"] = "sql"
        df = pd.concat([df, df_sql])

        ddl_data = self.milvus_client.query(
            collection_name=get_flag_value(Flag.VANNA_DDL_COLLECION),
            output_fields=["*"],
            limit=MAX_LIMIT_SIZE,
        )

        df_ddl = pd.DataFrame({
            "id": [doc["id"] for doc in ddl_data],
            "question": [None for doc in ddl_data],
            "content": [doc["ddl"] for doc in ddl_data],
        })
        df_ddl["training_data_type"] = "ddl"
        df = pd.concat([df, df_ddl])

        doc_data = self.milvus_client.query(
            collection_name=get_flag_value(Flag.VANNA_DOC_COLLECION),
            output_fields=["*"],
            limit=MAX_LIMIT_SIZE,
        )

        df_doc = pd.DataFrame({
            "id": [doc["id"] for doc in doc_data],
            "question": [None for doc in doc_data],
            "content": [doc["doc"] for doc in doc_data],
        })
        df_doc["training_data_type"] = "documentation"
        df = pd.concat([df, df_doc])

        comment_data = self.milvus_client.query(
            collection_name=get_flag_value(Flag.VANNA_COMMENT_COLLECION),
            output_fields=["*"],
            limit=MAX_LIMIT_SIZE,
        )

        df_sql_comment = pd.DataFrame({
            "id": [doc["id"] for doc in comment_data],
            "question": [doc["question"] for doc in comment_data],
            "content": [doc["sql"] for doc in comment_data],
            "explanation": [doc["explanation"] for doc in comment_data],
        })
        df_sql_comment["training_data_type"] = "sql_comment"
        df = pd.concat([df, df_sql_comment])

        return df

    def remove_training_data(self, id: str, **kwargs) -> bool:
        """Remove training data by ID."""
        if id.endswith("-sql-comment"):
            self.milvus_client.delete(collection_name=get_flag_value(Flag.VANNA_COMMENT_COLLECION), ids=[id])
            return True
        return super().remove_training_data(id, **kwargs)


class VannaChat(MilvusVectorStore, VannaChatNVIDIA):
    """Combined Vanna implementation with Milvus and NVIDIA NIM."""

    def __init__(self, client, v_config, config=None):
        MilvusVectorStore.__init__(self, config=v_config)
        VannaChatNVIDIA.__init__(self, client=client, config=config)


def get_model():
    """Get NVIDIA model configuration."""
    client_config = {
        "model": "meta/llama-3.1-70b-instruct",
        "temperature": 0.0,
        "top_p": 0.1,
        "max_completion_tokens": 1024,
    }
    if os.getenv("NVIDIA_API_KEY"):
        client_config["api_key"] = os.getenv("NVIDIA_API_KEY")
    return ChatNVIDIA(**client_config)


def get_embedder():
    """Get NVIDIA embedder configuration."""
    return NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key=os.getenv("NVIDIA_API_KEY"),
        truncate="NONE",
    )


def generate_ddl():
    """Generate DDL statements for all tables."""
    ddls = []
    for table in TABLES:
        columns = table["schema"]
        columns_spec = []
        for c in columns:
            columns_spec.append(f"{c['field']} {c['type']}")
        columns_spec_str = ",".join(columns_spec)
        ddl = f"create table {table['name']} ({columns_spec_str})"
        ddls.append(ddl)
    return ddls


def generate_all_table_description() -> str:
    """Generate descriptions for all tables."""
    descriptions = []
    for table in TABLES:
        tbl_desc = f"Table {generate_table_description(table)}"
        descriptions.append(tbl_desc)
    return "\n".join(descriptions)


def get_examples(analysis_filter: list[str] | None = None, ):
    """Get training examples with optional metadata filtering.

    Args:
        analysis_filter: Filter by analysis type
            (e.g., ["pbr"], ["supply_gap"], or both)

    Returns:
        List of examples matching the filter criteria
    """
    examples = [PBR_EXAMPLES, DEMAND_DLT_EXAMPLES]
    all_examples = []
    for ex in examples:
        all_examples = all_examples + ex

    # Apply metadata filter if provided
    if analysis_filter:
        filtered_examples = []
        for ex in all_examples:
            metadata = ex.get("metadata", {})

            # Check analysis filter
            if metadata.get("analysis") in analysis_filter:
                filtered_examples.append(ex)

        all_examples = filtered_examples
        logger.info(f"Filtered examples: {len(all_examples)} (analysis={analysis_filter})")

    logger.info(f"Vanna num sql: {len(all_examples)}")
    return all_examples


def connect_to_databricks(
    vn,
    server_hostname: str | None = None,
    http_path: str | None = None,
    access_token: str | None = None,
    catalog: str | None = None,
    schema: str | None = None,
):
    """Connect Vanna to Databricks SQL Warehouse."""
    from databricks import sql

    # Use environment variables if not provided
    if not server_hostname:
        server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
    if not http_path:
        http_path = os.getenv("DATABRICKS_HTTP_PATH")
    if not access_token:
        access_token = os.getenv("DATABRICKS_ACCESS_TOKEN")

    if not all([server_hostname, http_path, access_token]):
        msg_missingvalue = "Missing required Databricks connection parameters"
        raise ValueError(msg_missingvalue)

    # Create connection
    conn = sql.connect(server_hostname=server_hostname, http_path=http_path, access_token=access_token)

    def run_sql_databricks(sql_query: str) -> pd.DataFrame:
        """Execute SQL on Databricks and return results as DataFrame."""
        try:
            with conn.cursor() as cursor:
                # Set catalog and schema if provided
                if catalog:
                    cursor.execute(f"USE CATALOG {catalog}")
                if schema:
                    cursor.execute(f"USE SCHEMA {schema}")

                # Execute the query
                cursor.execute(sql_query)

                # Fetch results
                results = cursor.fetchall()

                # Get column names
                columns = ([desc[0] for desc in cursor.description] if cursor.description else [])

                # Create DataFrame
                df = pd.DataFrame(results, columns=columns)
                return df

        except Exception as e:
            logger.error(f"Error executing query on Databricks: {e}")
            raise

    # Set the run_sql function and dialect
    vn.run_sql = run_sql_databricks
    vn.run_sql_is_set = True
    vn.dialect = "databricks"

    logger.info("Successfully connected to Databricks")
    return vn


async def train_vanna(
    vn,
    analysis_filter: list[str] | None = None,
):
    """Train Vanna with DDL, documentation, and examples.

    Args:
        vn: Vanna instance
        analysis_filter: Filter by analysis type
            (e.g., ["pbr"], ["supply_gap"], or both)
    """
    # Check if the target collection already exists
    if not vn._check_collections_empty():
        logger.info("Non-empty Vanna collections already exist. Skipping training.")
        return
    logger.info("Training vanna")

    # Log filtering configuration
    if analysis_filter:
        logger.info(f"Training with metadata filter: analysis={analysis_filter}")

    # Train with DDL
    ddls = generate_ddl()
    for ddl in ddls:
        vn.train(ddl=ddl)

    # Train with documentation
    table_description = generate_all_table_description()
    concepts = [table_description]
    for concept in concepts:
        vn.train(documentation=concept)

    # Train with sql and sql_comment (with optional filtering)
    examples = get_examples(analysis_filter=analysis_filter)
    logger.info(f"Vanna num sql: {len(examples)}")

    for ex in examples:
        # Get analysis metadata
        analysis = ex.get("metadata", {}).get("analysis", "")

        # Check if example has a comment
        if "Comment" in ex and ex["Comment"]:
            if len(ex["Query"]) == 0 or len(ex["SQL"]) == 0:
                logger.info("pair of question and sql can not be null")
                continue

            vn.add_sql_comment(question=ex["Query"], sql=ex["SQL"], explanation=ex["Comment"])
            logger.info(f"Added to sql_comment collection: {ex['Query']}")

        else:
            # Add to regular sql collection with analysis metadata
            vn.add_question_sql(question=ex["Query"], sql=ex["SQL"], analysis=analysis)
            logger.info(f"Added to sql collection: {ex['Query']} (analysis={analysis})")

    return


async def get_lock():
    """Get or create the initialization lock."""
    global _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    return _init_lock


async def get_vanna_instance(
    llm_client=None,
    embedder_client=None,
    vanna_remote=None,
    milvus_host=None,
    milvus_port=None,
    milvus_user=None,
    milvus_db_name=None,
    valid_analysis_types=None,
):
    """Get or create a singleton Vanna instance.

    Args:
        llm_client: LLM client for SQL generation
        embedder_client: Embedder client for vector operations
        vanna_remote: Whether to use remote Milvus
        milvus_host: Milvus host URL
        milvus_port: Milvus port
        milvus_user: Milvus username
        milvus_db_name: Milvus database name
        valid_analysis_types: List of valid analysis types from config
                             (e.g., ['pbr', 'supply_gap'])

    Note:
        The instance is cached globally. If you need to force recreation
        (e.g., DNS changed), call reset_vanna_instance() first.
    """
    global _vanna_instance

    logger.info("Setting up Vanna instance...")

    # Fast path - return existing instance without lock
    if _vanna_instance is not None:
        logger.info("Vanna instance already exists")
        return _vanna_instance

    # Slow path - create new instance if needed
    _init_lock = await get_lock()
    async with _init_lock:
        # Double check after acquiring lock
        if _vanna_instance is not None:
            logger.info("Vanna instance already exists")
            return _vanna_instance

        milvus_config = {
            "milvus_client":
                create_milvus_client(
                    is_async=False,
                    milvus_host=milvus_host,
                    milvus_port=milvus_port,
                    milvus_user=milvus_user,
                    milvus_db_name=milvus_db_name,
                    vanna_remote=vanna_remote,
                ),
            "async_milvus_client":
                create_milvus_client(
                    is_async=True,
                    milvus_host=milvus_host,
                    milvus_port=milvus_port,
                    milvus_user=milvus_user,
                    milvus_db_name=milvus_db_name,
                    vanna_remote=vanna_remote,
                ),
            "n_results":
                7,
            "embedder_client":
                embedder_client,  # Pass embedder to config
            "valid_analysis_types":
                valid_analysis_types,  # Pass from config
        }

        # Create and initialize new instance
        logger.info("Creating new Vanna instance: ChatNVIDIA")
        if llm_client is None:
            logger.info("No LLM client provided, using default model")
            llm_client = get_model()
        llm_config = {"model": llm_client.model, "initial_prompt": INSTRUCTION_PROMPT}

        logger.info("Initializing Vanna Chat instance")
        vn = VannaChat(
            llm_client,
            v_config=llm_config | milvus_config,
            config=llm_config | milvus_config,
        )

        # Connect to Databricks
        vn = connect_to_databricks(
            vn,
            catalog="hive_metastore",
            schema="silver_global_supply",
        )

        _vanna_instance = vn
        return _vanna_instance


async def generate_sql_vanna(query: str):
    """Generate SQL using Vanna instance."""
    vanna = await get_vanna_instance()

    try:
        sql = await vanna.generate_sql(query, allow_llm_to_see_data=True)
    except Exception as e:
        logger.error(f"Error during generate_sql: {e}")
        sql = None
    return sql


# TODO(apourhabib): Create a new tool independent from text2sql for follow ups
async def generate_followup_questions_if_enabled(
    vn,
    question: str,
    df=None,
    enable_followup_questions: bool = False,
    sql_query: str | None = None,
):
    """Generate follow-up questions if enabled.

    Args:
        vn: Vanna instance
        question: Original user question
        df: DataFrame with query results (optional)
        enable_followup_questions: Whether to generate follow-up questions
        sql_query: The SQL query that was executed (for table inference)

    Returns:
        List of follow-up questions or None if disabled/failed
    """
    if not enable_followup_questions:
        return None

    try:
        # Create table preview from the first few rows
        table_preview = ""
        if df is not None and not df.empty:
            # Get first 3 rows for preview
            preview_df = df.head(3)
            table_preview = preview_df.to_string(index=False)

        # Infer table name from SQL query
        table_name = TTYSC_TABLES.PBR  # Default fallback
        if sql_query:
            table_name = infer_table_name_from_sql(sql_query)
            logger.info(f"Inferred table name: {table_name}")

        # Get retrieval data for follow-up questions
        retrieval_in_parallel = []
        retrieval_in_parallel.append(vn.get_similar_question_sql(question))
        retrieval_in_parallel.append(vn.get_related_ddl(question))
        retrieval_in_parallel.append(vn.get_related_documentation(question))

        question_sql_list, ddl_list, doc_list = await asyncio.gather(*retrieval_in_parallel)

        # Generate follow-up questions prompt
        followup_prompt = vn.get_followup_questions_prompt(
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            table_preview=table_preview,
            table_name=table_name,
        )

        # Get follow-up questions from LLM
        followup_response = await vn.submit_prompt(followup_prompt)

        # Get applicable use cases for validation
        applicable_use_cases = TABLE_USE_CASES.get(table_name, TABLE_USE_CASES[TTYSC_TABLES.PBR])

        # Validate the response using the validation methods
        validated_response = vn._validate_questions(followup_response, applicable_use_cases, question)

        # Parse the validated response into a list of questions
        followup_questions = []
        if validated_response:
            lines = validated_response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):  # Skip empty lines and comments
                    # Remove numbering if present (e.g., "1. " or "1)")
                    clean_line = re.sub(r"^\d+[\.\)]\s*", "", line)
                    if clean_line:
                        followup_questions.append(clean_line)

        # TODO(apourhabib): Avoid hardcoding this value and make it configurable
        return followup_questions[:4]  # Return max 4 questions as specified in guidelines

    except Exception as e:
        logger.error(f"Error generating follow-up questions: {e}")
        return None


async def generate_sql_with_fallback(
    question: str,
    allow_llm_to_see_data: bool = False,
    execute_sql: bool = True,
    enable_followup_questions: bool = False,
    analysis_type: str | None = None,
) -> str:
    """Generate SQL with fallback for DB error handling.

    Args:
        question: Natural language question to convert to SQL
        allow_llm_to_see_data: Whether to allow LLM to see actual data
        execute_sql: Whether to execute the SQL query
        enable_followup_questions: Whether to generate follow-up questions
        analysis_type: Optional filter for analysis type (e.g., 'pbr' or 'supply_gap')
                      to retrieve only relevant few-shot examples
    """
    try:
        # Get Vanna instance
        vn = await get_vanna_instance()

        # Generate SQL using Vanna
        sql = await vn.generate_sql(
            question=question,
            allow_llm_to_see_data=allow_llm_to_see_data,
            analysis_type=analysis_type,
        )

        # If we're only generating SQL (not executing), return just the SQL query
        if not execute_sql:
            return sql

        # Try to execute SQL - with fallback on error
        try:
            (rows, columns) = await async_query(add_table_prefix(sql))

            # Convert results to DataFrame for processing
            df = pd.DataFrame(rows, columns=columns)

            # Convert DataFrame to JSON-friendly format for transmission
            records = df.to_dict("records")

            # Generate follow-up questions if enabled
            followup_questions = await generate_followup_questions_if_enabled(vn,
                                                                              question,
                                                                              df,
                                                                              enable_followup_questions,
                                                                              sql_query=sql)

            # Return successful result
            return build_sql_result(
                sql=sql,
                question=question,
                rows=rows,
                columns=columns,
                records=records,
                followup_questions=followup_questions,
                confidence="high",
                method="vanna",
            )

        except Exception as sql_error:
            # SQL execution failed - try to regenerate with error context
            error_message = {"sql_error": str(sql_error), "previous_sql": sql}
            log_error_message = (f"SQL execution failed: {error_message['sql_error']}. "
                                 "Attempting regeneration...")
            logger.error(log_error_message)

            try:
                # Regenerate SQL with error context
                retry_sql = await vn.generate_sql(
                    question=question,
                    allow_llm_to_see_data=allow_llm_to_see_data,
                    error_message=error_message,
                    analysis_type=analysis_type,
                )

                # Try executing the regenerated SQL
                (rows, columns) = await async_query(add_table_prefix(retry_sql))

                # Convert results to DataFrame for processing
                df = pd.DataFrame(rows, columns=columns)

                # Convert DataFrame to JSON-friendly format for transmission
                records = df.to_dict("records")

                # Generate follow-up questions if enabled
                followup_questions = await generate_followup_questions_if_enabled(vn,
                                                                                  question,
                                                                                  df,
                                                                                  enable_followup_questions,
                                                                                  sql_query=retry_sql)

                # Return successful result from retry
                return build_sql_result(
                    sql=retry_sql,
                    question=question,
                    rows=rows,
                    columns=columns,
                    records=records,
                    followup_questions=followup_questions,
                    confidence="medium",
                    method="vanna_retry",
                    original_error=str(sql_error),
                )

            except Exception as retry_error:
                # Both attempts failed
                logger.error(f"SQL regeneration also failed: {str(retry_error)}")
                explanation = (f"SQL execution failed twice. Original error: {str(sql_error)}."
                               f"Retry error: {str(retry_error)}")
                return json.dumps({
                    "sql": retry_sql if "retry_sql" in locals() else sql,
                    "error": str(retry_error),
                    "explanation": explanation,
                    "confidence": "low",
                    "original_error": str(sql_error),
                    "retry_error": str(retry_error),
                })

    except Exception as e:
        logger.error(f"Error generating SQL with Vanna: {str(e)}")

        if execute_sql:
            return json.dumps({
                "sql": None,
                "error": str(e),
                "explanation": f"An error occurred while generating SQL: {str(e)}",
                "confidence": "low",
            })
        else:
            raise


def check_client(client):
    """Check Milvus client collections."""
    collections = [
        get_flag_value(Flag.VANNA_DDL_COLLECION),
        get_flag_value(Flag.VANNA_SQL_COLLECION),
        get_flag_value(Flag.VANNA_DOC_COLLECION),
        get_flag_value(Flag.VANNA_COMMENT_COLLECION),
    ]

    for coll in collections:
        try:
            data = client.query(
                collection_name=coll,
                output_fields=["*"],
                limit=MAX_LIMIT_SIZE,
            )
        except Exception as e:
            logger.info(f"Failed to query {coll}: {e}")
            data = []

        logger.info(f"check_client {coll}: {len(data)}")


def drop_collection(client, collection_to_drop):
    """Drop a specific Milvus collection."""
    collections = [
        get_flag_value(Flag.VANNA_DDL_COLLECION),
        get_flag_value(Flag.VANNA_SQL_COLLECION),
        get_flag_value(Flag.VANNA_DOC_COLLECION),
        get_flag_value(Flag.VANNA_COMMENT_COLLECION),
    ]
    if collection_to_drop not in collections:
        logger.warning(f"{collection_to_drop} not in {collections}. "
                       "Please specify a valid collection name")

    with contextlib.suppress(Exception):
        client.drop_collection(collection_name=collection_to_drop)
