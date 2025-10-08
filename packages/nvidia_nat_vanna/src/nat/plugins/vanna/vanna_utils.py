"""Vanna-based Text-to-SQL implementation utilities.

This module provides generic Vanna framework integration for text-to-SQL conversion,
supporting multiple database backends and vector stores.
"""

import asyncio
import json
import logging
import uuid
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pymilvus import DataType, MilvusClient
from vanna.base import VannaBase
from vanna.milvus import Milvus_VectorStore
from vanna.utils import deterministic_uuid

logger = logging.getLogger(__name__)

# Global instances for singleton pattern
_vanna_instance = None
_init_lock = None


# Default prompts
DEFAULT_RESPONSE_GUIDELINES = """
Response Guidelines:
1. Carefully analyze the question to understand the user’s intent, target columns, filters, and any aggregation or grouping requirements.
2. Retrieve only the relevant columns and tables needed to answer the question, avoiding unnecessary joins or selections.
"""


def to_langchain_msgs(msgs):
    """Convert message dicts to LangChain message objects."""
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
        self.model = getattr(client, "model", "unknown")

    def system_message(self, message: str) -> dict:
        """Create system message."""
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> dict:
        """Create user message."""
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> dict:
        """Create assistant message."""
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
        initial_prompt += DEFAULT_RESPONSE_GUIDELINES
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

    async def submit_prompt(self, prompt) -> str:
        """Submit prompt to LLM."""
        response = await self.client.ainvoke(prompt)
        return response.content

    async def generate_sql(
        self,
        question: str,
        allow_llm_to_see_data: bool = False,
        error_message: dict | None = None,
        **kwargs,
    ) -> str:
        """Generate SQL using the LLM.

        Args:
            question: Natural language question to convert to SQL
            allow_llm_to_see_data: Whether to allow LLM to see actual data
            error_message: Optional error message from previous SQL execution
            **kwargs: Additional keyword arguments

        Returns:
            Generated SQL query string
        """
        logger.info("Starting SQL Generation with Vanna")

        # Get initial prompt from config
        initial_prompt = self.config.get("initial_prompt", None)

        # Retrieve relevant context in parallel
        retrieval_tasks = [
            self.get_similar_question_sql(question, **kwargs),
            self.get_related_ddl(question, **kwargs),
            self.get_related_documentation(question, **kwargs),
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

        # Generate SQL
        try:
            response = await self.submit_prompt(prompt)
            llm_response = response
            self.log(title="LLM Response", message=llm_response)

        except Exception as e:
            logger.error(f"Error calling LLM during SQL query generation: {e}")
            raise

        # Handle intermediate SQL for data introspection
        if "intermediate_sql" in llm_response:
            if not allow_llm_to_see_data:
                return (
                    "The LLM is not allowed to see the data in your database. "
                    "Your question requires database introspection to generate "
                    "the necessary SQL. Please set allow_llm_to_see_data=True "
                    "to enable this."
                )

            intermediate_sql = self.extract_sql(llm_response)

            try:
                self.log(title="Running Intermediate SQL", message=intermediate_sql)
                df = self.run_sql(intermediate_sql)

                # Re-generate with intermediate results
                prompt = self.get_sql_prompt(
                    initial_prompt=initial_prompt,
                    question=question,
                    question_sql_list=question_sql_list,
                    ddl_list=ddl_list,
                    doc_list=doc_list
                    + [
                        f"The following is a pandas DataFrame with the results of "
                        f"the intermediate SQL query {intermediate_sql}:\n"
                        + df.to_markdown()
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


class MilvusVectorStore(Milvus_VectorStore):
    """Extended Milvus vector store for Vanna."""

    def __init__(self, config=None):
        try:
            VannaBase.__init__(self, config=config)

            self.milvus_client = config["milvus_client"]
            self.async_milvus_client = config["async_milvus_client"]
            self.n_results = config.get("n_results", 5)

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

    def add_question_sql(self, question: str, sql: str) -> str:
        """Add question-SQL pair to collection."""
        if len(question) == 0 or len(sql) == 0:
            msg = "Question and SQL cannot be empty"
            raise ValueError(msg)
        _id = str(uuid.uuid4()) + "-sql"
        embedding = self.embedder.embed_documents([question])[0]
        data = {"id": _id, "text": question, "sql": sql, "vector": embedding}
        self.milvus_client.insert(collection_name=self.sql_collection, data=data)
        return _id

    def add_ddl(self, ddl: str) -> str:
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

    def add_documentation(self, documentation: str) -> str:
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

    async def get_related_ddl(self, question: str, **kwargs) -> list:
        """Retrieve all DDL statements."""
        ddl_list = []
        try:
            res = await self.async_milvus_client.query(
                collection_name=self.ddl_collection,
                output_fields=["ddl"],
                limit=1000,
            )
            for doc in res:
                ddl_list.append(doc["ddl"])
        except Exception as e:
            logger.error(f"Error retrieving DDL: {e}")
        return ddl_list

    async def get_related_documentation(self, question: str) -> list:
        """Retrieve all documentation."""
        doc_list = []
        try:
            res = await self.async_milvus_client.query(
                collection_name=self.doc_collection,
                output_fields=["doc"],
                limit=1000,
            )
            for doc in res:
                doc_list.append(doc["doc"])
        except Exception as e:
            logger.error(f"Error retrieving documentation: {e}")
        return doc_list

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

    def get_training_data(self) -> pd.DataFrame:
        """Get all training data."""
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
    initial_prompt: str | None = None,
    n_results: int = 5,
    sql_collection: str = "vanna_sql",
    ddl_collection: str = "vanna_ddl",
    doc_collection: str = "vanna_documentation",
) -> VannaLangChain:
    """Get or create a singleton Vanna instance.

    Args:
        llm_client: LangChain LLM client for SQL generation
        embedder_client: LangChain embedder for vector operations
        milvus_client: Sync Milvus client
        async_milvus_client: Async Milvus client
        initial_prompt: Optional custom system prompt
        n_results: Number of similar examples to retrieve
        sql_collection: Collection name for SQL examples
        ddl_collection: Collection name for DDL
        doc_collection: Collection name for documentation

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
            "initial_prompt": initial_prompt,
            "n_results": n_results,
            "sql_collection": sql_collection,
            "ddl_collection": ddl_collection,
            "doc_collection": doc_collection,
        }

        logger.info("Creating new Vanna instance with LangChain")
        vn = VannaLangChain(client=llm_client, config=config)

        _vanna_instance = vn
        return _vanna_instance


def reset_vanna_instance():
    """Reset the singleton Vanna instance.

    Useful for testing or when configuration changes.
    """
    global _vanna_instance
    _vanna_instance = None


async def train_vanna(vn: VannaLangChain, training_data: dict):
    """Train Vanna with DDL, documentation, and question-SQL examples.

    Args:
        vn: Vanna instance
        training_data: Dict containing:
            - ddl: List of DDL statements
            - documentation: List of documentation strings
            - examples: List of dicts with 'question' and 'sql' keys
    """
    logger.info("Training Vanna...")

    # Train with DDL
    for ddl in training_data.get("ddl", []):
        vn.train(ddl=ddl)

    # Train with documentation
    for doc in training_data.get("documentation", []):
        vn.train(documentation=doc)

    # Train with examples
    for example in training_data.get("examples", []):
        if "question" in example and "sql" in example:
            vn.train(question=example["question"], sql=example["sql"])

    logger.info("Vanna training complete")

