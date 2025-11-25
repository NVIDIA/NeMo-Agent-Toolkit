# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import uuid

from nat.plugins.vanna.training_db_schema import VANNA_RESPONSE_GUIDELINES
from nat.plugins.vanna.training_db_schema import VANNA_TRAINING_DDL
from nat.plugins.vanna.training_db_schema import VANNA_TRAINING_DOCUMENTATION
from nat.plugins.vanna.training_db_schema import VANNA_TRAINING_EXAMPLES
from nat.plugins.vanna.training_db_schema import VANNA_TRAINING_PROMPT
from vanna.base import VannaBase
from vanna.milvus import Milvus_VectorStore

logger = logging.getLogger(__name__)


def extract_json_from_string(content: str) -> dict:
    """Extract JSON from a string that may contain additional content.

    Args:
        content: String containing JSON data

    Returns:
        Parsed JSON as dictionary

    Raises:
        ValueError: If no valid JSON found
    """
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


def remove_think_tags(text: str, model_name: str, reasoning_models: set[str]) -> str:
    """Remove think tags from reasoning model output based on model type.

    Args:
        text: Text potentially containing think tags
        model_name: Name of the model
        reasoning_models: Set of model names that require think tag removal

    Returns:
        Text with think tags removed if applicable
    """
    if "openai/gpt-oss" in model_name:
        return text
    elif model_name in reasoning_models:
        from nat.utils.io.model_processing import remove_r1_think_tags

        return remove_r1_think_tags(text)
    else:
        return text


def to_langchain_msgs(msgs):
    """Convert message dicts to LangChain message objects."""
    from langchain_core.messages import AIMessage
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import SystemMessage

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

        # Store configurable values
        self.milvus_search_limit = self.config.get("milvus_search_limit", 1000)
        self.reasoning_models = self.config["reasoning_models"]
        self.chat_models = self.config["chat_models"]

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
        initial_prompt = (f"You are a {self.dialect} expert. "
                          "Please generate diverse question-SQL pairs where each SQL "
                          "statement starts with either `SELECT` or `WITH`. "
                          "Your response should follow the response guidelines and format instructions.")

        # Add DDL information
        initial_prompt = self.add_ddl_to_prompt(initial_prompt, ddl_list, max_tokens=self.max_tokens)

        # Add documentation
        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(initial_prompt, doc_list, max_tokens=self.max_tokens)

        # Add response guidelines
        initial_prompt += VANNA_TRAINING_PROMPT

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
            initial_prompt = (f"You are a {self.dialect} expert. "
                              "Please help to generate a SQL query to answer the question. "
                              "Your response should ONLY be based on the given context "
                              "and follow the response guidelines and format instructions.")

        # Add DDL information
        initial_prompt = self.add_ddl_to_prompt(initial_prompt, ddl_list, max_tokens=self.max_tokens)

        # Add documentation
        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(initial_prompt, doc_list, max_tokens=self.max_tokens)

        # Add response guidelines
        initial_prompt += VANNA_RESPONSE_GUIDELINES
        initial_prompt += (f"3. Ensure that the output SQL is {self.dialect}-compliant "
                           "and executable, and free of syntax errors.\n")

        # Add error message if provided
        if error_message is not None:
            initial_prompt += (f"4. For question: {question}. "
                               "\tPrevious SQL attempt failed with error: "
                               f"{error_message['sql_error']}\n"
                               f"\tPrevious SQL was: {error_message['previous_sql']}\n"
                               "\tPlease fix the SQL syntax/logic error and regenerate.")

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
            llm_name = getattr(self.client, 'model_name', None) or getattr(self.client, 'model', 'unknown')

            # Get LLM response (with streaming for reasoning models)
            if llm_name in self.reasoning_models:
                llm_output = ""
                async for chunk in self.client.astream(prompt):
                    llm_output += chunk.content
                llm_response = remove_think_tags(llm_output, llm_name, self.reasoning_models)
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

            # Only use async client
            self.async_milvus_client = config["async_milvus_client"]
            self.n_results = config.get("n_results", 5)
            self.milvus_search_limit = config.get("milvus_search_limit", 1000)

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

            # Collection creation tracking
            self._collections_created = False
        except Exception as e:
            logger.error(f"Error initializing MilvusVectorStore: {e}")
            raise

    async def _ensure_collections_created(self):
        """Ensure all necessary Milvus collections are created (async)."""
        if self._collections_created:
            return

        logger.info("Creating Milvus collections if they don't exist...")
        await self._create_sql_collection(self.sql_collection)
        await self._create_ddl_collection(self.ddl_collection)
        await self._create_doc_collection(self.doc_collection)
        self._collections_created = True

    async def _create_sql_collection(self, name: str):
        """Create SQL collection using async client."""
        from pymilvus import DataType
        from pymilvus import MilvusClient
        from pymilvus import MilvusException

        # Check if collection already exists by attempting to load it
        try:
            await self.async_milvus_client.load_collection(collection_name=name)
            logger.debug(f"Collection {name} already exists, skipping creation")
            return
        except MilvusException as e:
            if "collection not found" not in str(e).lower():
                raise  # Unexpected error, re-raise
            # Collection doesn't exist, proceed to create it

        # Create the collection
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
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="sql", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self._embedding_dim,
        )

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="L2")
        await self.async_milvus_client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )
        logger.info(f"Created collection: {name}")

    async def _create_ddl_collection(self, name: str):
        """Create DDL collection using async client."""
        from pymilvus import DataType
        from pymilvus import MilvusClient
        from pymilvus import MilvusException

        # Check if collection already exists by attempting to load it
        try:
            await self.async_milvus_client.load_collection(collection_name=name)
            logger.debug(f"Collection {name} already exists, skipping creation")
            return
        except MilvusException as e:
            if "collection not found" not in str(e).lower():
                raise  # Unexpected error, re-raise
            # Collection doesn't exist, proceed to create it

        # Create the collection
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
        schema.add_field(field_name="ddl", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self._embedding_dim,
        )

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="L2")
        await self.async_milvus_client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )
        logger.info(f"Created collection: {name}")

    async def _create_doc_collection(self, name: str):
        """Create documentation collection using async client."""
        from pymilvus import DataType
        from pymilvus import MilvusClient
        from pymilvus import MilvusException

        # Check if collection already exists by attempting to load it
        try:
            await self.async_milvus_client.load_collection(collection_name=name)
            logger.debug(f"Collection {name} already exists, skipping creation")
            return
        except MilvusException as e:
            if "collection not found" not in str(e).lower():
                raise  # Unexpected error, re-raise
            # Collection doesn't exist, proceed to create it

        # Create the collection
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
        schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self._embedding_dim,
        )

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="L2")
        await self.async_milvus_client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )
        logger.info(f"Created collection: {name}")

    async def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """Add question-SQL pair to collection using async client."""
        if len(question) == 0 or len(sql) == 0:
            msg = "Question and SQL cannot be empty"
            raise ValueError(msg)
        _id = str(uuid.uuid4()) + "-sql"
        embedding = (await self.embedder.aembed_documents([question]))[0]
        data = {"id": _id, "text": question, "sql": sql, "vector": embedding}
        await self.async_milvus_client.insert(collection_name=self.sql_collection, data=data)
        return _id

    async def add_ddl(self, ddl: str, **kwargs) -> str:
        """Add DDL to collection using async client."""
        if len(ddl) == 0:
            msg = "DDL cannot be empty"
            raise ValueError(msg)
        _id = str(uuid.uuid4()) + "-ddl"
        embedding = self.embedder.embed_documents([ddl])[0]
        await self.async_milvus_client.insert(
            collection_name=self.ddl_collection,
            data={
                "id": _id, "ddl": ddl, "vector": embedding
            },
        )
        return _id

    async def add_documentation(self, documentation: str, **kwargs) -> str:
        """Add documentation to collection using async client."""
        if len(documentation) == 0:
            msg = "Documentation cannot be empty"
            raise ValueError(msg)
        _id = str(uuid.uuid4()) + "-doc"
        embedding = self.embedder.embed_documents([documentation])[0]
        await self.async_milvus_client.insert(
            collection_name=self.doc_collection,
            data={
                "id": _id, "doc": documentation, "vector": embedding
            },
        )
        return _id

    async def get_related_record(self, collection_name: str) -> list:
        """Retrieve all related records using async client."""

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
                limit=self.milvus_search_limit,
            )
            for record in records:
                record_list.append(record[output_field])
        except Exception as e:
            logger.exception(f"Error retrieving {collection_name}: {e}")
        return record_list

    async def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """Get similar question-SQL pairs using async client."""
        search_params = {"metric_type": "L2", "params": {"nprobe": 128}}
        list_sql = []
        try:
            # Use async embedder and async Milvus client
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
            logger.exception(f"Error retrieving similar questions: {e}")
        return list_sql

    async def get_training_data(self, **kwargs):
        """Get all training data using async client."""
        import pandas as pd

        df = pd.DataFrame()

        # Get SQL data
        sql_data = await self.async_milvus_client.query(collection_name=self.sql_collection,
                                                        output_fields=["*"],
                                                        limit=1000)
        if sql_data:
            df_sql = pd.DataFrame({
                "id": [doc["id"] for doc in sql_data],
                "question": [doc["text"] for doc in sql_data],
                "content": [doc["sql"] for doc in sql_data],
            })
            df_sql["training_data_type"] = "sql"
            df = pd.concat([df, df_sql])

        # Get DDL data
        ddl_data = await self.async_milvus_client.query(collection_name=self.ddl_collection,
                                                        output_fields=["*"],
                                                        limit=1000)
        if ddl_data:
            df_ddl = pd.DataFrame({
                "id": [doc["id"] for doc in ddl_data],
                "question": [None for doc in ddl_data],
                "content": [doc["ddl"] for doc in ddl_data],
            })
            df_ddl["training_data_type"] = "ddl"
            df = pd.concat([df, df_ddl])

        # Get documentation data
        doc_data = await self.async_milvus_client.query(collection_name=self.doc_collection,
                                                        output_fields=["*"],
                                                        limit=1000)
        if doc_data:
            df_doc = pd.DataFrame({
                "id": [doc["id"] for doc in doc_data],
                "question": [None for doc in doc_data],
                "content": [doc["doc"] for doc in doc_data],
            })
            df_doc["training_data_type"] = "documentation"
            df = pd.concat([df, df_doc])

        return df

    async def close(self):
        """Close async Milvus client connection."""
        if hasattr(self, 'async_milvus_client') and self.async_milvus_client is not None:
            try:
                await self.async_milvus_client.close()
                logger.info("Closed async Milvus client")
            except Exception as e:
                logger.warning(f"Error closing async Milvus client: {e}")


class VannaLangChain(MilvusVectorStore, VannaLangChainLLM):
    """Combined Vanna implementation with Milvus and LangChain LLM."""

    def __init__(self, client, config=None):
        """Initialize VannaLangChain.

        Args:
            client: LangChain LLM client
            config: Configuration dict for Milvus vector store and LLM settings
        """
        MilvusVectorStore.__init__(self, config=config)
        VannaLangChainLLM.__init__(self, client=client, config=config)
        # Store database engine (if any) - lifecycle matches Vanna singleton
        self.db_engine = None

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
            kwargs: Additional keyword arguments

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
        return {"sql": sql.replace("\\_", "_"), "explanation": explanation_text}


class VannaSingleton:
    """Singleton manager for Vanna instances."""

    _instance: VannaLangChain | None = None
    _lock: asyncio.Lock | None = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        """Get or create the lock in the current event loop."""
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    def instance(cls) -> VannaLangChain | None:
        """Get current instance without creating one.

        Returns:
            Current Vanna instance or None if not initialized
        """
        return cls._instance

    @classmethod
    async def get_instance(
        cls,
        llm_client,
        embedder_client,
        async_milvus_client,
        dialect: str = "SQLite",
        initial_prompt: str | None = None,
        n_results: int = 5,
        sql_collection: str = "vanna_sql",
        ddl_collection: str = "vanna_ddl",
        doc_collection: str = "vanna_documentation",
        milvus_search_limit: int = 1000,
        reasoning_models: set[str] | None = None,
        chat_models: set[str] | None = None,
        create_collections: bool = True,
    ) -> VannaLangChain:
        """Get or create a singleton Vanna instance.

        Args:
            llm_client: LangChain LLM client for SQL generation
            embedder_client: LangChain embedder for vector operations
            async_milvus_client: Async Milvus client
            dialect: SQL dialect (e.g., 'databricks', 'postgres', 'mysql')
            initial_prompt: Optional custom system prompt
            n_results: Number of similar examples to retrieve
            sql_collection: Collection name for SQL examples
            ddl_collection: Collection name for DDL
            doc_collection: Collection name for documentation
            milvus_search_limit: Maximum limit size for vector search operations
            reasoning_models: Models requiring special handling for think tags
            chat_models: Models using standard response handling
            create_collections: Whether to create Milvus collections if they don't exist (default True)

        Returns:
            Initialized Vanna instance
        """
        logger.info("Setting up Vanna instance...")

        # Fast path - return existing instance
        if cls._instance is not None:
            logger.info("Vanna instance already exists")
            return cls._instance

        # Slow path - create new instance
        async with cls._get_lock():
            # Double check after acquiring lock
            if cls._instance is not None:
                logger.info("Vanna instance already exists")
                return cls._instance

            config = {
                "async_milvus_client": async_milvus_client,
                "embedder_client": embedder_client,
                "dialect": dialect,
                "initial_prompt": initial_prompt,
                "n_results": n_results,
                "sql_collection": sql_collection,
                "ddl_collection": ddl_collection,
                "doc_collection": doc_collection,
                "milvus_search_limit": milvus_search_limit,
                "reasoning_models": reasoning_models,
                "chat_models": chat_models,
                "create_collections": create_collections,
            }

            logger.info(f"Creating new Vanna instance with LangChain (dialect: {dialect})")
            cls._instance = VannaLangChain(client=llm_client, config=config)

            # Create collections if requested
            if create_collections:
                await cls._instance._ensure_collections_created()  # type: ignore[attr-defined]

            return cls._instance

    @classmethod
    async def reset(cls):
        """Reset the singleton Vanna instance.

        Useful for testing or when configuration changes.
        Properly disposes of database engine if present.
        """
        if cls._instance is not None:
            try:
                # Dispose database engine if present
                if hasattr(cls._instance, "db_engine") and cls._instance.db_engine is not None:
                    try:
                        cls._instance.db_engine.dispose()
                        logger.info("Disposed database engine pool")
                    except Exception as e:
                        logger.warning(f"Error disposing database engine: {e}")

                await cls._instance.close()
            except Exception as e:
                logger.warning(f"Error closing Vanna instance: {e}")
        cls._instance = None


async def train_vanna(vn: VannaLangChain, auto_train: bool = False):
    """Train Vanna with DDL, documentation, and question-SQL examples.

    Args:
        vn: Vanna instance
        auto_train: Whether to automatically train Vanna (auto-extract DDL and generate training data from database)
    """
    logger.info("Training Vanna...")

    # Train with DDL
    if auto_train:
        from nat.plugins.vanna.training_db_schema import VANNA_ACTIVE_TABLES

        dialect = vn.dialect.lower()
        ddls = []

        if dialect == 'databricks':
            for table in VANNA_ACTIVE_TABLES:
                ddl_sql = f"SHOW CREATE TABLE {table}"
                ddl = await vn.run_sql(ddl_sql)
                ddl = ddl.to_string()  # Convert DataFrame to string
                ddls.append(ddl)
        else:
            error_msg = (f"Auto-extraction of DDL is currently only supported for Databricks. "
                         f"Current dialect: {vn.dialect}. "
                         "Please either set auto_train=False or use 'databricks' as the dialect.")
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
    else:
        ddls = VANNA_TRAINING_DDL

    for ddl in ddls:
        await vn.add_ddl(ddl=ddl)

    # Train with documentation
    for doc in VANNA_TRAINING_DOCUMENTATION:
        await vn.add_documentation(documentation=doc)

    # Train with examples
    # Add manual examples
    examples = []
    examples.extend(VANNA_TRAINING_EXAMPLES)

    if auto_train:
        logger.info("Generating training examples with LLM...")
        # Retrieve relevant context in parallel
        retrieval_tasks = [vn.get_related_record(vn.ddl_collection), vn.get_related_record(vn.doc_collection)]

        ddl_list, doc_list = await asyncio.gather(*retrieval_tasks)

        prompt = vn.get_training_sql_prompt(
            ddl_list=ddl_list,
            doc_list=doc_list,
        )

        llm_response = await vn.submit_prompt(prompt)

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
                    log_msg = f"Adding valid LLM-generated Question-SQL:\n{question_sql.get('question', '')}\n{sql}"
                    logger.info(log_msg)
                except Exception as e:
                    logger.debug(f"Dropping invalid LLM-generated SQL: {e}")
        except Exception as e:
            logger.warning(f"Failed to parse LLM response for training examples: {e}")

    # Train with validated examples
    logger.info(f"Training Vanna with {len(examples)} validated examples")
    for example in examples:
        await vn.add_question_sql(question=example["question"], sql=example["sql"])
    df = await vn.get_training_data()
    df.to_csv("vanna_training_data.csv", index=False)
    logger.info("Vanna training complete")
