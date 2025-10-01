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

"""Vanna utilities for SQL generation using NVIDIA NIM services."""

import logging

from langchain_nvidia import NVIDIAEmbeddings
from vanna.base import VannaBase
from vanna.chromadb import ChromaDB_VectorStore

logger = logging.getLogger(__name__)


class NIMCustomLLM(VannaBase):
    """Custom LLM implementation for Vanna using NVIDIA NIM."""

    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)

        if not config:
            raise ValueError("config must be passed")

        # default parameters - can be overrided using config
        self.temperature = 0.7

        if "temperature" in config:
            self.temperature = config["temperature"]

        # If only config is passed
        if "api_key" not in config:
            raise ValueError("config must contain a NIM api_key")

        if "model" not in config:
            raise ValueError("config must contain a NIM model")

        api_key = config["api_key"]
        model = config["model"]

        # Initialize ChatNVIDIA client
        from langchain_nvidia import ChatNVIDIA

        self.client = ChatNVIDIA(
            api_key=api_key,
            model=model,
            temperature=self.temperature,
        )
        self.model = model

    def system_message(self, message: str) -> dict:
        """Create a system message."""
        return {
            "role": "system",
            "content": message + "\n DO NOT PRODUCE MARKDOWN, ONLY RESPOND IN PLAIN TEXT",
        }

    def user_message(self, message: str) -> dict:
        """Create a user message."""
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> dict:
        """Create an assistant message."""
        return {"role": "assistant", "content": message}

    def submit_prompt(self, prompt, **kwargs) -> str:
        """Submit a prompt to the LLM."""
        if prompt is None:
            raise Exception("Prompt is None")

        if len(prompt) == 0:
            raise Exception("Prompt is empty")

        # Count the number of tokens in the message log
        # Use 4 as an approximation for the number of characters per token
        num_tokens = 0
        for message in prompt:
            num_tokens += len(message["content"]) / 4
        logger.debug(f"Using model {self.model} for {num_tokens} tokens (approx)")

        logger.debug(f"Submitting prompt with {len(prompt)} messages")
        logger.debug(f"Prompt content preview: {str(prompt)[:500]}...")

        try:
            response = self.client.invoke(prompt)
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response content type: {type(response.content)}")
            logger.debug(
                f"Response content length: {len(response.content) if response.content else 0}"
            )
            logger.debug(
                f"Response content preview: {response.content[:200] if response.content else 'None'}..."
            )
            return response.content
        except Exception as e:
            logger.error(f"Error in submit_prompt: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise


class NIMVanna(ChromaDB_VectorStore, NIMCustomLLM):
    """Vanna implementation using NVIDIA NIM for LLM and ChromaDB for vector storage."""

    def __init__(self, VectorConfig=None, LLMConfig=None):
        ChromaDB_VectorStore.__init__(self, config=VectorConfig)
        NIMCustomLLM.__init__(self, config=LLMConfig)


class NVIDIAEmbeddingFunction:
    """
    A class that can be used as a replacement for chroma's DefaultEmbeddingFunction.
    It takes in input (text or list of texts) and returns embeddings using NVIDIA's API.

    This class fixes two major interface compatibility issues between ChromaDB and NVIDIA embeddings:

    1. INPUT FORMAT MISMATCH:
       - ChromaDB passes ['query text'] (list) to embed_query()
       - But langchain_nvidia's embed_query() expects 'query text' (string)
       - When list is passed, langchain does [text] internally → [['query text']] → API 500 error
       - FIX: Detect list input and extract string before calling langchain

    2. OUTPUT FORMAT MISMATCH:
       - ChromaDB expects embed_query() to return [[embedding_vector]] (list of embeddings)
       - But langchain returns [embedding_vector] (single embedding vector)
       - This causes: TypeError: 'float' object cannot be converted to 'Sequence'
       - FIX: Wrap single embedding in list: return [embeddings]
    """

    def __init__(self, api_key, model="nvidia/llama-3.2-nv-embedqa-1b-v2"):
        """
        Initialize the embedding function with the API key and model name.

        Parameters:
        - api_key (str): The API key for authentication.
        - model (str): The model name to use for embeddings.
                      Default: nvidia/llama-3.2-nv-embedqa-1b-v2 (tested and working)
        """
        self.api_key = api_key
        self.model = model

        logger.info(f"Initializing NVIDIA embeddings with model: {model}")
        logger.debug(f"API key length: {len(api_key) if api_key else 0}")

        self.embeddings = NVIDIAEmbeddings(
            api_key=api_key, model_name=model, input_type="query", truncate="NONE"
        )
        logger.info("Successfully initialized NVIDIA embeddings")

    def __call__(self, input):
        """
        Call method to make the object callable, as required by chroma's EmbeddingFunction interface.

        NOTE: This method is used by ChromaDB for batch embedding operations.
        The embed_query() method above handles the single query case with the critical fixes.

        Parameters:
        - input (str or list): The input data for which embeddings need to be generated.

        Returns:
        - embedding (list): The embedding vector(s) for the input data.
        """
        logger.debug(f"__call__ method called with input type: {type(input)}")
        logger.debug(f"__call__ input: {input}")

        # Ensure input is a list, as required by ChromaDB
        if isinstance(input, str):
            input_data = [input]
        else:
            input_data = input

        logger.debug(f"Processing {len(input_data)} texts for embedding")

        # Generate embeddings for each text
        embeddings = []
        for i, text in enumerate(input_data):
            logger.debug(f"Embedding text {i+1}/{len(input_data)}: {text[:50]}...")
            embedding = self.embeddings.embed_query(text)
            embeddings.append(embedding)

        logger.debug(f"Generated {len(embeddings)} embeddings")
        # Always return a list of embeddings for ChromaDB
        return embeddings

    def name(self):
        """
        Returns a custom name for the embedding function.

        Returns:
            str: The name of the embedding function.
        """
        return "NVIDIA Embedding Function"

    def embed_query(self, input: str) -> list[list[float]]:
        """
        Generate embeddings for a single query.

        ChromaDB calls this method with ['query text'] (list) but langchain_nvidia expects 'query text' (string).
        We must extract the string from the list to prevent API 500 errors.

        ChromaDB expects this method to return [[embedding_vector]] (list of embeddings)
        but langchain returns [embedding_vector] (single embedding). We wrap it in a list.
        """
        logger.debug(f"Embedding query: {input}")
        logger.debug(f"Input type: {type(input)}")
        logger.debug(f"Using model: {self.model}")

        # Handle ChromaDB's list input format
        # ChromaDB sometimes passes a list instead of a string
        # Extract the string from the list if needed
        if isinstance(input, list):
            if len(input) == 1:
                query_text = input[0]
                logger.debug(f"Extracted string from list: {query_text}")
            else:
                logger.error(f"Unexpected list length: {len(input)}")
                raise ValueError(
                    f"Expected single string or list with one element, got list with {len(input)} elements"
                )
        else:
            query_text = input

        try:
            # Call langchain_nvidia with the extracted string
            embeddings = self.embeddings.embed_query(query_text)
            logger.debug(
                f"Successfully generated embeddings of length: {len(embeddings) if embeddings else 0}"
            )

            # Wrap single embedding in list for ChromaDB compatibility
            # ChromaDB expects a list of embeddings, even for a single query
            return [embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings for query: {e}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Query text: {query_text}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.

        This function expects a list of strings. If it's a list of lists of strings, flatten it to handle cases
        where the input is unexpectedly nested.
        """
        logger.debug(f"Embedding {len(input)} documents...")
        logger.debug(f"Using model: {self.model}")

        try:
            embeddings = self.embeddings.embed_documents(input)
            logger.debug("Successfully generated document embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating document embeddings: {e}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Input documents count: {len(input)}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise


def chunk_documentation(text: str, max_chars: int = 1500) -> list:
    """
    Split long documentation into smaller chunks to avoid token limits.

    Args:
        text: The documentation text to chunk
        max_chars: Maximum characters per chunk (approximate)

    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for paragraph in paragraphs:
        # If adding this paragraph would exceed the limit, save current chunk and start new one
        if len(current_chunk) + len(paragraph) + 2 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # If any chunk is still too long, split it further
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            # Split long chunk into sentences
            sentences = chunk.split(". ")
            temp_chunk = ""
            for sentence in sentences:
                if len(temp_chunk) + len(sentence) + 2 > max_chars and temp_chunk:
                    final_chunks.append(temp_chunk.strip() + ".")
                    temp_chunk = sentence
                else:
                    if temp_chunk:
                        temp_chunk += ". " + sentence
                    else:
                        temp_chunk = sentence
            if temp_chunk.strip():
                final_chunks.append(temp_chunk.strip())
        else:
            final_chunks.append(chunk)

    return final_chunks


def init_vanna(vn, training_data_path: str | None = None):
    """
    Initialize and train a Vanna instance for SQL generation using configurable training data.

    This function configures a Vanna SQL generation agent with training data loaded from a YAML file,
    making it scalable for different SQL data sources with different contexts.

    Args:
        vn: Vanna instance to be trained and configured
        training_data_path: Path to YAML file containing training data. If None, no training is applied.

    Returns:
        None: Modifies the Vanna instance in-place

    Example:
        >>> from vanna.chromadb import ChromaDB_VectorStore
        >>> vn = NIMCustomLLM(config) & ChromaDB_VectorStore()
        >>> vn.connect_to_sqlite("path/to/database.db")
        >>> init_vanna(vn, "path/to/training_data.yaml")
        >>> # Vanna is now ready to generate SQL queries
    """
    import os

    logger.info("=== Starting Vanna initialization ===")

    # Get and train DDL from sqlite_master (if connected to SQLite)
    if hasattr(vn, "run_sql") and vn.run_sql_is_set:
        logger.info("Loading DDL from database...")
        try:
            # Try SQLite-specific query first
            try:
                df_ddl = vn.run_sql(
                    "SELECT type, sql FROM sqlite_master WHERE sql is not null"
                )
            except Exception:
                # For non-SQLite databases, try standard information_schema
                try:
                    df_ddl = vn.run_sql(
                        """
                        SELECT table_name, column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_schema = 'public'
                    """
                    )
                except Exception as e:
                    logger.warning(f"Could not auto-load DDL from database: {e}")
                    df_ddl = None

            if df_ddl is not None and not df_ddl.empty:
                ddl_count = len(df_ddl)
                logger.info(f"Found {ddl_count} DDL statements in database")

                if "sql" in df_ddl.columns:
                    for i, ddl in enumerate(df_ddl["sql"].to_list(), 1):
                        if ddl:
                            logger.debug(f"Training DDL {i}/{ddl_count}: {ddl[:100]}...")
                            vn.train(ddl=ddl)

                logger.info(
                    f"Successfully trained {ddl_count} DDL statements from database"
                )
        except Exception as e:
            logger.error(f"Error loading DDL from database: {e}")
            # Continue with training data from YAML

    # Load and apply training data from YAML file
    if training_data_path:
        logger.info(f"Training data path provided: {training_data_path}")

        if os.path.exists(training_data_path):
            logger.info("Training data file exists, loading YAML...")

            try:
                import yaml

                with open(training_data_path, "r") as f:
                    training_data = yaml.safe_load(f)

                logger.info("Successfully loaded YAML training data")
                logger.info(
                    f"Training data keys: {list(training_data.keys()) if training_data else 'None'}"
                )

                # Train DDL statements
                ddl_statements = training_data.get("ddl", [])
                logger.info(f"Found {len(ddl_statements)} DDL statements")

                ddl_trained = 0
                for i, ddl_statement in enumerate(ddl_statements, 1):
                    if ddl_statement.strip():  # Only train non-empty statements
                        logger.debug(f"Training DDL {i}: {ddl_statement[:100]}...")
                        vn.train(ddl=ddl_statement)
                        ddl_trained += 1
                    else:
                        logger.warning(f"Skipping empty DDL statement at index {i}")

                logger.info(
                    f"Successfully trained {ddl_trained}/{len(ddl_statements)} DDL statements"
                )

                # Train documentation with chunking
                documentation_list = training_data.get("documentation", [])
                logger.info(f"Found {len(documentation_list)} documentation entries")

                doc_chunks = []
                for i, doc_entry in enumerate(documentation_list, 1):
                    if doc_entry.strip():
                        logger.debug(
                            f"Processing documentation entry {i}: {doc_entry[:100]}..."
                        )
                        # Chunk each documentation entry to avoid token limits
                        entry_chunks = chunk_documentation(doc_entry)
                        doc_chunks.extend(entry_chunks)
                    else:
                        logger.warning(f"Skipping empty documentation entry at index {i}")

                logger.info(f"Split documentation into {len(doc_chunks)} total chunks")

                for i, chunk in enumerate(doc_chunks, 1):
                    try:
                        logger.debug(
                            f"Training documentation chunk {i}/{len(doc_chunks)} ({len(chunk)} chars)"
                        )
                        vn.train(documentation=chunk)
                    except Exception as e:
                        logger.error(f"Error training documentation chunk {i}: {e}")
                        # Continue with other chunks

                logger.info(f"Successfully trained {len(doc_chunks)} documentation chunks")

                # Train question-SQL pairs
                question_sql_pairs = training_data.get("sql", [])
                logger.info(f"Found {len(question_sql_pairs)} question-SQL pairs")

                pairs_trained = 0
                for i, pair in enumerate(question_sql_pairs, 1):
                    question = pair.get("question", "")
                    sql = pair.get("sql", "")
                    if question.strip() and sql.strip():  # Only train non-empty pairs
                        logger.debug(
                            f"Training question-SQL pair {i}: Q='{question[:50]}...' SQL='{sql[:50]}...'"
                        )
                        vn.train(question=question, sql=sql)
                        pairs_trained += 1
                    else:
                        if not question.strip():
                            logger.warning(f"Skipping question-SQL pair {i}: empty question")
                        if not sql.strip():
                            logger.warning(f"Skipping question-SQL pair {i}: empty SQL")

                logger.info(
                    f"Successfully trained {pairs_trained}/{len(question_sql_pairs)} question-SQL pairs"
                )

                # Summary
                total_trained = ddl_trained + len(doc_chunks) + pairs_trained
                logger.info("=== Training Summary ===")
                logger.info(f"  DDL statements: {ddl_trained}")
                logger.info(f"  Documentation chunks: {len(doc_chunks)}")
                logger.info(f"  Question-SQL pairs: {pairs_trained}")
                logger.info(f"  Total items trained: {total_trained}")

            except Exception as e:
                import yaml

                if isinstance(e, yaml.YAMLError):
                    logger.error(f"Error parsing YAML file {training_data_path}: {e}")
                else:
                    logger.error(f"Error loading training data from {training_data_path}: {e}")
                raise
        else:
            logger.warning(f"Training data file does not exist: {training_data_path}")
            logger.warning("Proceeding without YAML training data")
    else:
        logger.info("No training data path provided, skipping YAML training")

    logger.info("=== Vanna initialization completed ===")

