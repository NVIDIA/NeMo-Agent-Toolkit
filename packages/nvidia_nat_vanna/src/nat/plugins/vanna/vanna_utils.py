"""Vanna-based Text-to-SQL implementation utilities.

This module provides generic Vanna framework integration for text-to-SQL conversion,
supporting multiple database backends and vector stores.
"""

import asyncio
import logging
import uuid

from vanna.base import VannaBase
from vanna.milvus import Milvus_VectorStore

from nat.plugins.vanna.constants import REASONING_MODEL_VALUES, MAX_LIMIT_SIZE
from nat.plugins.vanna.db_schema import (
    RESPONSE_GUIDELINES,
    TRAINING_PROMPT,
    TRAINING_EXAMPLES,
    TRAINING_DDL,
    TRAINING_DOCUMENTATION,
)

logger = logging.getLogger(__name__)

# Global instances for singleton pattern
_vanna_instance = None
_init_lock = None


def extract_json_from_string(content: str) -> dict:
    """Extract JSON from a string that may contain additional content.

    Args:
        content: String containing JSON data

    Returns:
        Parsed JSON as dictionary

    Raises:
        ValueError: If no valid JSON found
    """
    import json

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            # Extract JSON from string that may contain additional content
            json_str = content
            # Try to find JSON between ``` markers
            if "```" in content:
                json_start = content.find("```")
                if json_start != -1:
                    json_start += len("```")
                    json_end = content.find("```", json_start)
                    if json_end != -1:
                        json_str = content[json_start:json_end]
                    else:
                        msg = "No JSON found in response"
                        raise ValueError(msg)
            else:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                json_str = content[json_start:json_end]

            return json.loads(json_str.strip())
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to extract JSON from content: {e}")
            raise ValueError("Could not extract valid JSON from response") from e


def remove_think_tags(text: str, model_name: str) -> str:
    """Remove think tags from reasoning model output based on model type.

    Args:
        text: Text potentially containing think tags
        model_name: Name of the model

    Returns:
        Text with think tags removed if applicable
    """
    if "openai/gpt-oss" in model_name:
        return text
    elif model_name in REASONING_MODEL_VALUES:
        from nat.utils.io.model_processing import remove_r1_think_tags

        return remove_r1_think_tags(text)
    else:
        return text

def to_langchain_msgs(msgs):
    """Convert message dicts to LangChain message objects."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    role2cls = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    return [role2cls[m["role"]](content=m["content"]) for m in msgs]


class VannaLangChainLLM(VannaBase):
    """LangChain LLM integration for Vanna framework."""

    def __init__(self, client=None, config=None):
        if client is None:
            msg = "LangChain client must be provided"
            raise ValueError(msg)

        self.client = client
        self.config = config or {}
        self.dialect = self.config.get("dialect", "SQL")
        self.model = getattr(self.client, "model", "unknown")

    def system_message(self, message: str) -> dict:
        """Create system message."""
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> dict:
        """Create user message."""
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> dict:
        """Create assistant message."""
        return {"role": "assistant", "content": message}

    def get_training_sql_prompt(
        self,
        ddl_list: list,
        doc_list: list,
    ) -> list:
        """Generate prompt for synthetic question-SQL pairs."""
        initial_prompt = (
            f"You are a {self.dialect} expert. "
            "Please generate diverse question–SQL pairs where each SQL statement starts with either `SELECT` or `WITH`. "
            "Your response should follow the response guidelines and format instructions."
        )

        # Add DDL information
        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        # Add documentation
        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        # Add response guidelines
        initial_prompt += TRAINING_PROMPT

        # Build message log
        message_log = [self.system_message(initial_prompt)]
        message_log.append(self.user_message('Begin:'))
        return message_log

    def get_sql_prompt(
        self,
        initial_prompt: str | None,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        error_message: dict | None = None,
        **kwargs,
    ) -> list:
        """Generate prompt for SQL generation."""
        if initial_prompt is None:
            initial_prompt = (
                f"You are a {self.dialect} expert. "
                "Please help to generate a SQL query to answer the question. "
                "Your response should ONLY be based on the given context "
                "and follow the response guidelines and format instructions."
            )

        # Add DDL information
        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        # Add documentation
        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        # Add response guidelines
        initial_prompt += RESPONSE_GUIDELINES
        initial_prompt += (
            f"3. Ensure that the output SQL is {self.dialect}-compliant "
            "and executable, and free of syntax errors.\n"
        )

        # Add error message if provided
        if error_message is not None:
            initial_prompt += (
                f"4. For question: {question}. "
                "\tPrevious SQL attempt failed with error: "
                f"{error_message['sql_error']}\n"
                f"\tPrevious SQL was: {error_message['previous_sql']}\n"
                "\tPlease fix the SQL syntax/logic error and regenerate."
            )

        # Build message log with examples
        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example and "question" in example and "sql" in example:
                message_log.append(self.user_message(example["question"]))
                message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))
        return message_log

    async def submit_prompt(self, prompt, **kwargs) -> str:
        """Submit prompt to LLM."""
        try:
            # Determine model name
            llm_name = getattr(
                self.client, 'model_name', None
                ) or getattr(self.client, 'model', 'unknown')

            # Get LLM response (with streaming for reasoning models)
            if llm_name in REASONING_MODEL_VALUES:
                llm_output = ""
                async for chunk in self.client.astream(prompt):
                    llm_output += chunk.content
                llm_response = remove_think_tags(llm_output, llm_name)
            else:
                llm_response = (await self.client.ainvoke(prompt)).content

            logger.debug(f"LLM Response: {llm_response}")
            return llm_response

        except Exception as e:
            logger.error(f"Error calling LLM during SQL query generation: {e}")
            raise


class MilvusVectorStore(Milvus_VectorStore):
    """Extended Milvus vector store for Vanna."""

    def __init__(self, config=None):
        try:
            VannaBase.__init__(self, config=config)

            self.milvus_client = config["milvus_client"]
            self.async_milvus_client = config["async_milvus_client"]
            self.n_results = config.get("n_results", 5)
            self._owns_sync_client = config.get("owns_sync_client", True)

            # Use configured embedder
            if config.get("embedder_client") is not None:
                logger.info("Using configured embedder client")
                self.embedder = config["embedder_client"]
            else:
                msg = "Embedder client must be provided in config"
                raise ValueError(msg)

            try:
                self._embedding_dim = len(self.embedder.embed_documents(["test"])[0])
                logger.info(f"Embedding dimension: {self._embedding_dim}")
            except Exception as e:
                logger.error(f"Error calling embedder during Milvus initialization: {e}")
                raise

            # Collection names
            self.sql_collection = config.get("sql_collection", "vanna_sql")
            self.ddl_collection = config.get("ddl_collection", "vanna_ddl")
            self.doc_collection = config.get("doc_collection", "vanna_documentation")

            self._create_collections()
        except Exception as e:
            logger.error(f"Error initializing MilvusVectorStore: {e}")
            raise

    def _create_collections(self):
        """Create all necessary Milvus collections."""
        self._create_sql_collection(self.sql_collection)
        self._create_ddl_collection(self.ddl_collection)
        self._create_doc_collection(self.doc_collection)

    def _create_sql_collection(self, name: str):
        """Create SQL collection."""
        from pymilvus import DataType, MilvusClient

        if not self.milvus_client.has_collection(collection_name=name):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=65535,
            )
            schema.add_field(
                field_name="text", datatype=DataType.VARCHAR, max_length=65535
            )
            schema.add_field(
                field_name="sql", datatype=DataType.VARCHAR, max_length=65535
            )
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self._embedding_dim,
            )

            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector", index_type="AUTOINDEX", metric_type="L2"
            )
            self.milvus_client.create_collection(
                collection_name=name,
                schema=schema,
                index_params=index_params,
                consistency_level="Strong",
            )

    def _create_ddl_collection(self, name: str):
        """Create DDL collection."""
        from pymilvus import DataType, MilvusClient

        if not self.milvus_client.has_collection(collection_name=name):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=65535,
            )
            schema.add_field(
                field_name="ddl", datatype=DataType.VARCHAR, max_length=65535
            )
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self._embedding_dim,
            )

            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector", index_type="AUTOINDEX", metric_type="L2"
            )
            self.milvus_client.create_collection(
                collection_name=name,
                schema=schema,
                index_params=index_params,
                consistency_level="Strong",
            )

    def _create_doc_collection(self, name: str):
        """Create documentation collection."""
        from pymilvus import DataType, MilvusClient

        if not self.milvus_client.has_collection(collection_name=name):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=65535,
            )
            schema.add_field(
                field_name="doc", datatype=DataType.VARCHAR, max_length=65535
            )
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self._embedding_dim,
            )

            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector", index_type="AUTOINDEX", metric_type="L2"
            )
            self.milvus_client.create_collection(
                collection_name=name,
                schema=schema,
                index_params=index_params,
                consistency_level="Strong",
            )

    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """Add question-SQL pair to collection."""
        if len(question) == 0 or len(sql) == 0:
            msg = "Question and SQL cannot be empty"
            raise ValueError(msg)
        _id = str(uuid.uuid4()) + "-sql"
        embedding = self.embedder.embed_documents([question])[0]
        data = {"id": _id, "text": question, "sql": sql, "vector": embedding}
        self.milvus_client.insert(collection_name=self.sql_collection, data=data)
        return _id

    def add_ddl(self, ddl: str, **kwargs) -> str:
        """Add DDL to collection."""
        if len(ddl) == 0:
            msg = "DDL cannot be empty"
            raise ValueError(msg)
        _id = str(uuid.uuid4()) + "-ddl"
        embedding = self.embedder.embed_documents([ddl])[0]
        self.milvus_client.insert(
            collection_name=self.ddl_collection,
            data={"id": _id, "ddl": ddl, "vector": embedding},
        )
        return _id

    def add_documentation(self, documentation: str, **kwargs) -> str:
        """Add documentation to collection."""
        if len(documentation) == 0:
            msg = "Documentation cannot be empty"
            raise ValueError(msg)
        _id = str(uuid.uuid4()) + "-doc"
        embedding = self.embedder.embed_documents([documentation])[0]
        self.milvus_client.insert(
            collection_name=self.doc_collection,
            data={"id": _id, "doc": documentation, "vector": embedding},
        )
        return _id

    async def get_related_record(self, collection_name: str) -> list:
        """Retrieve all related records."""

        if 'ddl' in collection_name:
            output_field = "ddl"
        elif 'doc' in collection_name:
            output_field = "doc"
        else:
            output_field = collection_name

        record_list = []
        try:
            records = await self.async_milvus_client.query(
                collection_name=collection_name,
                output_fields=[output_field],
                limit=MAX_LIMIT_SIZE,
            )
            for record in records:
                record_list.append(record[output_field])
        except Exception as e:
            logger.error(f"Error retrieving {collection_name}: {e}")
        return record_list

    async def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """Get similar question-SQL pairs."""
        search_params = {"metric_type": "L2", "params": {"nprobe": 128}}
        list_sql = []
        try:
            embeddings = [await self.embedder.aembed_query(question)]

            res = await self.async_milvus_client.search(
                collection_name=self.sql_collection,
                anns_field="vector",
                data=embeddings,
                limit=self.n_results,
                output_fields=["text", "sql"],
                search_params=search_params,
            )
            res = res[0]

            for doc in res:
                entry = {
                    "question": doc["entity"]["text"],
                    "sql": doc["entity"]["sql"],
                }
                list_sql.append(entry)

            logger.info(f"Retrieved {len(list_sql)} similar SQL examples")
        except Exception as e:
            logger.error(f"Error retrieving similar questions: {e}")
        return list_sql

    def get_training_data(self, **kwargs):
        """Get all training data."""
        import pandas as pd

        df = pd.DataFrame()

        # Get SQL data
        sql_data = self.milvus_client.query(
            collection_name=self.sql_collection, output_fields=["*"], limit=1000
        )
        if sql_data:
            df_sql = pd.DataFrame(
                {
                    "id": [doc["id"] for doc in sql_data],
                    "question": [doc["text"] for doc in sql_data],
                    "content": [doc["sql"] for doc in sql_data],
                }
            )
            df_sql["training_data_type"] = "sql"
            df = pd.concat([df, df_sql])

        # Get DDL data
        ddl_data = self.milvus_client.query(
            collection_name=self.ddl_collection, output_fields=["*"], limit=1000
        )
        if ddl_data:
            df_ddl = pd.DataFrame(
                {
                    "id": [doc["id"] for doc in ddl_data],
                    "question": [None for doc in ddl_data],
                    "content": [doc["ddl"] for doc in ddl_data],
                }
            )
            df_ddl["training_data_type"] = "ddl"
            df = pd.concat([df, df_ddl])

        # Get documentation data
        doc_data = self.milvus_client.query(
            collection_name=self.doc_collection, output_fields=["*"], limit=1000
        )
        if doc_data:
            df_doc = pd.DataFrame(
                {
                    "id": [doc["id"] for doc in doc_data],
                    "question": [None for doc in doc_data],
                    "content": [doc["doc"] for doc in doc_data],
                }
            )
            df_doc["training_data_type"] = "documentation"
            df = pd.concat([df, df_doc])

        return df

    async def close(self):
        """Close Milvus client connections."""
        try:
            # Close async client (always owned by us)
            if hasattr(self, 'async_milvus_client') and self.async_milvus_client is not None:
                await self.async_milvus_client.close()
                logger.info("Closed async Milvus client")
        except Exception as e:
            logger.warning(f"Error closing async Milvus client: {e}")

        try:
            # Close sync client only if we own it
            if hasattr(self, '_owns_sync_client') and self._owns_sync_client:
                if hasattr(self, 'milvus_client') and self.milvus_client is not None:
                    self.milvus_client.close()
                    logger.info("Closed sync Milvus client")
        except Exception as e:
            logger.warning(f"Error closing sync Milvus client: {e}")


class VannaLangChain(MilvusVectorStore, VannaLangChainLLM):
    """Combined Vanna implementation with Milvus and LangChain LLM."""

    def __init__(self, client, config=None):
        """Initialize VannaLangChain.

        Args:
            client: LangChain LLM client
            config: Configuration dict containing:
                - milvus_client: Sync Milvus client
                - async_milvus_client: Async Milvus client
                - embedder_client: LangChain embedder
                - initial_prompt: Optional custom prompt
                - n_results: Number of similar examples to retrieve
                - sql_collection: Collection name for SQL examples
                - ddl_collection: Collection name for DDL
                - doc_collection: Collection name for documentation
        """
        MilvusVectorStore.__init__(self, config=config)
        VannaLangChainLLM.__init__(self, client=client, config=config)

    async def generate_sql(
        self,
        question: str,
        allow_llm_to_see_data: bool = False,
        error_message: dict | None = None,
        **kwargs,
    ) -> dict[str, str | None]:
        """Generate SQL using the LLM.

        Args:
            question: Natural language question to convert to SQL
            allow_llm_to_see_data: Whether to allow LLM to see actual data
            error_message: Optional error message from previous SQL execution
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with 'sql' and optional 'explanation' keys
        """
        logger.info("Starting SQL Generation with Vanna")

        # Get initial prompt from config
        initial_prompt = self.config.get("initial_prompt", None)

        # Retrieve relevant context in parallel
        retrieval_tasks = [
            self.get_similar_question_sql(question, **kwargs),
            self.get_related_record(self.ddl_collection),
            self.get_related_record(self.doc_collection),
        ]

        question_sql_list, ddl_list, doc_list = await asyncio.gather(*retrieval_tasks)

        # Build prompt
        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            error_message=error_message,
            **kwargs,
        )

        llm_response = await self.submit_prompt(prompt)

        # Try to extract structured JSON response (sql + explanation)
        try:
            llm_response_json = extract_json_from_string(llm_response)
            sql_text = llm_response_json.get("sql", "")
            explanation_text = llm_response_json.get("explanation")
        except Exception:
            # Fallback: treat entire response as SQL without explanation
            sql_text = llm_response
            explanation_text = None

        sql = self.extract_sql(sql_text)
        return {
            "sql": sql.replace("\\_", "_"),
            "explanation": explanation_text
        }


async def get_lock():
    """Get or create the initialization lock."""
    global _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    return _init_lock


async def get_vanna_instance(
    llm_client,
    embedder_client,
    milvus_client,
    async_milvus_client,
    dialect: str = "SQLite",
    initial_prompt: str | None = None,
    n_results: int = 5,
    sql_collection: str = "vanna_sql",
    ddl_collection: str = "vanna_ddl",
    doc_collection: str = "vanna_documentation",
    owns_sync_client: bool = True,
) -> VannaLangChain:
    """Get or create a singleton Vanna instance.

    Args:
        llm_client: LangChain LLM client for SQL generation
        embedder_client: LangChain embedder for vector operations
        milvus_client: Sync Milvus client
        async_milvus_client: Async Milvus client
        dialect: SQL dialect (e.g., 'databricks', 'postgres', 'mysql')
        initial_prompt: Optional custom system prompt
        n_results: Number of similar examples to retrieve
        sql_collection: Collection name for SQL examples
        ddl_collection: Collection name for DDL
        doc_collection: Collection name for documentation
        owns_sync_client: Whether we own the sync client for cleanup

    Returns:
        Initialized Vanna instance
    """
    global _vanna_instance

    logger.info("Setting up Vanna instance...")

    # Fast path - return existing instance
    if _vanna_instance is not None:
        logger.info("Vanna instance already exists")
        return _vanna_instance

    # Slow path - create new instance
    init_lock = await get_lock()
    async with init_lock:
        # Double check after acquiring lock
        if _vanna_instance is not None:
            logger.info("Vanna instance already exists")
            return _vanna_instance

        config = {
            "milvus_client": milvus_client,
            "async_milvus_client": async_milvus_client,
            "embedder_client": embedder_client,
            "dialect": dialect,
            "initial_prompt": initial_prompt,
            "n_results": n_results,
            "sql_collection": sql_collection,
            "ddl_collection": ddl_collection,
            "doc_collection": doc_collection,
            "owns_sync_client": owns_sync_client,
        }

        logger.info(f"Creating new Vanna instance with LangChain (dialect: {dialect})")
        vn = VannaLangChain(client=llm_client, config=config)

        _vanna_instance = vn
        return _vanna_instance


async def reset_vanna_instance():
    """Reset the singleton Vanna instance.

    Useful for testing or when configuration changes.
    """
    global _vanna_instance
    if _vanna_instance is not None:
        try:
            await _vanna_instance.close()
        except Exception as e:
            logger.warning(f"Error closing Vanna instance: {e}")
    _vanna_instance = None


async def train_vanna(vn: VannaLangChain, auto_extract_ddl: bool = False):
    """Train Vanna with DDL, documentation, and question-SQL examples.

    Args:
        vn: Vanna instance
        auto_extract_ddl: Whether to automatically extract DDL from the database
    """
    logger.info("Training Vanna...")

    # Train with DDL
    if auto_extract_ddl:
        from nat.plugins.vanna.db_schema import ACTIVE_TABLES

        dialect = vn.dialect.lower()
        ddls = []

        if dialect == 'databricks':
            for table in ACTIVE_TABLES:
                ddl_sql = f"SHOW CREATE TABLE {table}"
                ddl = await vn.run_sql(ddl_sql)
                ddl = ddl.to_string()  # Convert DataFrame to string
                ddls.append(ddl)

        elif dialect == 'mysql':
            for table in ACTIVE_TABLES:
                ddl_sql = f"SHOW CREATE TABLE {table};"
                ddl = await vn.run_sql(ddl_sql)
                ddl = ddl.to_string()  # Convert DataFrame to string
                ddls.append(ddl)

        elif dialect == 'sqlite':
            for table in ACTIVE_TABLES:
                ddl_sql = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';"
                ddl = await vn.run_sql(ddl_sql)
                ddl = ddl.to_string()  # Convert DataFrame to string
                ddls.append(ddl)

        else:
            error_msg = (
                f"Auto-extraction of DDL is not implemented for dialect: {vn.dialect}. "
                "Supported dialects: 'databricks', 'mysql', 'sqlite'. "
                "Please either set auto_extract_ddl=False or use a supported dialect."
            )
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
    else:
        ddls = TRAINING_DDL

    for ddl in ddls:
        vn.train(ddl=ddl)

    # Train with documentation
    for doc in TRAINING_DOCUMENTATION:
        vn.train(documentation=doc)

    # Retrieve relevant context in parallel
    retrieval_tasks = [
        vn.get_related_record(vn.ddl_collection),
        vn.get_related_record(vn.doc_collection)
    ]

    ddl_list, doc_list = await asyncio.gather(*retrieval_tasks)

    prompt = vn.get_training_sql_prompt(
        ddl_list=ddl_list,
        doc_list=doc_list,
    )

    llm_response = await vn.submit_prompt(prompt)

    # Validate and collect all examples
    examples = []
    examples.extend(TRAINING_EXAMPLES)

    # Validate LLM-generated examples
    try:
        question_sql_list = extract_json_from_string(llm_response)
        for question_sql in question_sql_list:
            sql = question_sql.get("sql", "")
            if not sql:
                continue
            try:
                await vn.run_sql(sql)
                examples.append({
                    "question": question_sql.get("question", ""),
                    "sql": sql,
                })
            except Exception as e:
                logger.debug(f"Dropping invalid LLM-generated SQL: {e}")
    except Exception as e:
        logger.warning(f"Failed to parse LLM response for training examples: {e}")

    # Train with validated examples
    logger.info(f"Training Vanna with {len(examples)} validated examples")
    for example in examples:
        vn.train(question=example["question"], sql=example["sql"])
    df = vn.get_training_data()
    df.to_csv("vanna_training_data.csv", index=False)
    logger.info("Vanna training complete")
