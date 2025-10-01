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

"""VannaManager - A simplified manager for Vanna instances."""

import hashlib
import logging
import os
import threading
from typing import Dict

from nat.retriever.sql_retriever.vanna_util import NIMVanna
from nat.retriever.sql_retriever.vanna_util import NVIDIAEmbeddingFunction
from nat.retriever.sql_retriever.vanna_util import init_vanna

logger = logging.getLogger(__name__)


class VannaManager:
    """
    A simplified singleton manager for Vanna instances.

    Key features:
    - Singleton pattern to ensure only one instance per configuration
    - Thread-safe operations
    - Simple instance management
    - Support for multiple database types: SQLite, generic SQL, and PostgreSQL
    """

    _instances: Dict[str, "VannaManager"] = {}
    _lock = threading.Lock()

    def __new__(cls, config_key: str):
        """Ensure singleton pattern per configuration."""
        with cls._lock:
            if config_key not in cls._instances:
                logger.debug(
                    f"VannaManager: Creating new singleton instance for config: {config_key}"
                )
                cls._instances[config_key] = super().__new__(cls)
                cls._instances[config_key]._initialized = False
            else:
                logger.debug(
                    f"VannaManager: Returning existing singleton instance for config: {config_key}"
                )
            return cls._instances[config_key]

    def __init__(
        self,
        config_key: str,
        vanna_llm_config=None,
        vanna_embedder_config=None,
        vector_store_path: str | None = None,
        db_connection_string: str | None = None,
        db_type: str = "sqlite",
        training_data_path: str | None = None,
        nvidia_api_key: str | None = None,
    ):
        """
        Initialize the VannaManager and create Vanna instance immediately if all config is provided.

        Args:
            config_key: Unique key for this configuration
            vanna_llm_config: LLM configuration object
            vanna_embedder_config: Embedder configuration object
            vector_store_path: Path to ChromaDB vector store
            db_connection_string: Database connection string (path for SQLite, connection string for others)
            db_type: Type of database - 'sqlite', 'postgres', or 'sql' (generic SQL with SQLAlchemy)
            training_data_path: Path to YAML training data file
            nvidia_api_key: NVIDIA API key (optional, can use NVIDIA_API_KEY env var)
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.config_key = config_key
        self.lock = threading.Lock()

        # Store configuration
        self.vanna_llm_config = vanna_llm_config
        self.vanna_embedder_config = vanna_embedder_config
        self.vector_store_path = vector_store_path
        self.db_connection_string = db_connection_string
        self.db_type = db_type
        self.training_data_path = training_data_path
        self.nvidia_api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY")

        # Create and initialize Vanna instance immediately if all required config is provided
        self.vanna_instance = None
        if all(
            [
                vanna_llm_config,
                vanna_embedder_config,
                vector_store_path,
                db_connection_string,
            ]
        ):
            logger.debug("VannaManager: Initializing with immediate Vanna instance creation")
            self.vanna_instance = self._create_instance()
        else:
            if any(
                [
                    vanna_llm_config,
                    vanna_embedder_config,
                    vector_store_path,
                    db_connection_string,
                ]
            ):
                logger.debug(
                    "VannaManager: Partial configuration provided, Vanna instance will be created later"
                )
            else:
                logger.debug(
                    "VannaManager: No configuration provided, Vanna instance will be created later"
                )

        self._initialized = True
        logger.debug(f"VannaManager initialized for config: {config_key}")

    def get_instance(
        self,
        vanna_llm_config=None,
        vanna_embedder_config=None,
        vector_store_path: str | None = None,
        db_connection_string: str | None = None,
        db_type: str | None = None,
        training_data_path: str | None = None,
        nvidia_api_key: str | None = None,
    ) -> NIMVanna:
        """
        Get the Vanna instance. If not created during init, create it now with provided parameters.
        """
        with self.lock:
            if self.vanna_instance is None:
                logger.debug("VannaManager: No instance created during init, creating now...")

                # Update configuration with provided parameters
                self.vanna_llm_config = vanna_llm_config or self.vanna_llm_config
                self.vanna_embedder_config = (
                    vanna_embedder_config or self.vanna_embedder_config
                )
                self.vector_store_path = vector_store_path or self.vector_store_path
                self.db_connection_string = (
                    db_connection_string or self.db_connection_string
                )
                self.db_type = db_type or self.db_type
                self.training_data_path = training_data_path or self.training_data_path
                self.nvidia_api_key = nvidia_api_key or self.nvidia_api_key

                if all(
                    [
                        self.vanna_llm_config,
                        self.vanna_embedder_config,
                        self.vector_store_path,
                        self.db_connection_string,
                    ]
                ):
                    self.vanna_instance = self._create_instance()
                else:
                    raise RuntimeError(
                        "VannaManager: Missing required configuration parameters"
                    )
            else:
                logger.debug(
                    f"VannaManager: Returning pre-initialized Vanna instance (ID: {id(self.vanna_instance)})"
                )

                # Show vector store status for pre-initialized instances
                try:
                    if self.vector_store_path and os.path.exists(self.vector_store_path):
                        list_of_folders = [
                            d
                            for d in os.listdir(self.vector_store_path)
                            if os.path.isdir(os.path.join(self.vector_store_path, d))
                        ]
                        logger.debug(
                            f"VannaManager: Vector store contains {len(list_of_folders)} collections/folders"
                        )
                        if list_of_folders:
                            logger.debug(
                                f"VannaManager: Vector store folders: {list_of_folders}"
                            )
                    else:
                        logger.debug("VannaManager: Vector store directory does not exist")
                except Exception as e:
                    logger.warning(f"VannaManager: Could not check vector store status: {e}")

            return self.vanna_instance

    def _create_instance(self) -> NIMVanna:
        """
        Create a new Vanna instance using the stored configuration.
        """
        # Type guards - these should never be None at this point due to earlier checks
        if not all(
            [
                self.vanna_llm_config,
                self.vanna_embedder_config,
                self.vector_store_path,
                self.db_connection_string,
            ]
        ):
            raise RuntimeError(
                "VannaManager: Cannot create instance without required configuration"
            )

        logger.info(f"VannaManager: Creating instance for {self.config_key}")
        logger.debug(f"VannaManager: Vector store path: {self.vector_store_path}")
        logger.debug(f"VannaManager: Database connection: {self.db_connection_string}")
        logger.debug(f"VannaManager: Database type: {self.db_type}")
        logger.debug(f"VannaManager: Training data path: {self.training_data_path}")

        # Create instance
        vn_instance = NIMVanna(
            VectorConfig={
                "client": "persistent",
                "path": self.vector_store_path,
                "embedding_function": NVIDIAEmbeddingFunction(
                    api_key=self.nvidia_api_key,
                    model=self.vanna_embedder_config.model_name,
                ),
            },
            LLMConfig={
                "api_key": self.nvidia_api_key,
                "model": self.vanna_llm_config.model_name,
            },
        )

        # Connect to database based on type
        logger.debug(f"VannaManager: Connecting to {self.db_type} database...")
        if self.db_type == "sqlite":
            vn_instance.connect_to_sqlite(self.db_connection_string)
        elif self.db_type == "postgres" or self.db_type == "postgresql":
            self._connect_to_postgres(vn_instance, self.db_connection_string)
        elif self.db_type == "sql":
            self._connect_to_sql(vn_instance, self.db_connection_string)
        else:
            raise ValueError(
                f"Unsupported database type: {self.db_type}. "
                "Supported types: 'sqlite', 'postgres', 'sql'"
            )

        # Set configuration - allow LLM to see data for database introspection
        vn_instance.allow_llm_to_see_data = True
        logger.debug("VannaManager: Set allow_llm_to_see_data = True")

        # Initialize if needed (check if vector store is empty)
        needs_init = self._needs_initialization()
        if needs_init:
            logger.info(
                "VannaManager: Vector store needs initialization, starting training..."
            )
            try:
                init_vanna(vn_instance, self.training_data_path)
                logger.info("VannaManager: Vector store initialization complete")
            except Exception as e:
                logger.error(f"VannaManager: Error during initialization: {e}")
                raise
        else:
            logger.debug(
                "VannaManager: Vector store already initialized, skipping training"
            )

        logger.info("VannaManager: Instance created successfully")
        return vn_instance

    def _connect_to_postgres(self, vn_instance: NIMVanna, connection_string: str):
        """
        Connect to a PostgreSQL database.

        Args:
            vn_instance: The Vanna instance to connect
            connection_string: PostgreSQL connection string in format:
                postgresql://user:password@host:port/database
        """
        try:
            import psycopg2
            from psycopg2.pool import SimpleConnectionPool

            logger.info("Connecting to PostgreSQL database...")

            # Parse connection string if needed
            if connection_string.startswith("postgresql://"):
                # Use SQLAlchemy-style connection for Vanna
                vn_instance.connect_to_postgres(url=connection_string)
            else:
                # Assume it's a psycopg2 connection string
                vn_instance.connect_to_postgres(url=f"postgresql://{connection_string}")

            logger.info("Successfully connected to PostgreSQL database")
        except ImportError:
            logger.error(
                "psycopg2 is required for PostgreSQL connections. "
                "Install it with: pip install psycopg2-binary"
            )
            raise
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            raise

    def _connect_to_sql(self, vn_instance: NIMVanna, connection_string: str):
        """
        Connect to a generic SQL database using SQLAlchemy.

        Args:
            vn_instance: The Vanna instance to connect
            connection_string: SQLAlchemy-compatible connection string, e.g.:
                - MySQL: mysql+pymysql://user:password@host:port/database
                - PostgreSQL: postgresql://user:password@host:port/database
                - SQL Server: mssql+pyodbc://user:password@host:port/database?driver=ODBC+Driver+17+for+SQL+Server
                - Oracle: oracle+cx_oracle://user:password@host:port/?service_name=service
        """
        try:
            from sqlalchemy import create_engine

            logger.info("Connecting to SQL database via SQLAlchemy...")

            # Create SQLAlchemy engine
            engine = create_engine(connection_string)

            # Connect Vanna to the database using the engine
            vn_instance.connect_to_sqlalchemy(engine)

            logger.info("Successfully connected to SQL database")
        except ImportError:
            logger.error(
                "SQLAlchemy is required for generic SQL connections. "
                "Install it with: pip install sqlalchemy"
            )
            raise
        except Exception as e:
            logger.error(f"Error connecting to SQL database: {e}")
            raise

    def _needs_initialization(self) -> bool:
        """
        Check if the vector store needs initialization by checking if it's empty.
        """
        logger.debug("VannaManager: Checking if vector store needs initialization...")
        logger.debug(f"VannaManager: Vector store path: {self.vector_store_path}")

        # Type guard - vector_store_path should be set at this point
        if self.vector_store_path is None:
            logger.warning("VannaManager: Vector store path is None, assuming initialization needed")
            return True

        try:
            if not os.path.exists(self.vector_store_path):
                logger.debug(
                    "VannaManager: Vector store directory does not exist -> needs initialization"
                )
                return True

            # Check if there are any subdirectories (ChromaDB creates subdirectories when data is stored)
            list_of_folders = [
                d
                for d in os.listdir(self.vector_store_path)
                if os.path.isdir(os.path.join(self.vector_store_path, d))
            ]

            logger.debug(
                f"VannaManager: Found {len(list_of_folders)} folders in vector store"
            )
            if list_of_folders:
                logger.debug(f"VannaManager: Vector store folders: {list_of_folders}")
                logger.debug(
                    "VannaManager: Vector store is populated -> skipping initialization"
                )
                return False
            else:
                logger.debug("VannaManager: Vector store is empty -> needs initialization")
                return True

        except Exception as e:
            logger.warning(f"VannaManager: Could not check vector store status: {e}")
            logger.warning("VannaManager: Defaulting to needs initialization = True")
            return True

    def generate_sql_safe(self, question: str) -> str:
        """
        Generate SQL with error handling.
        """
        with self.lock:
            if self.vanna_instance is None:
                raise RuntimeError("VannaManager: No instance available")

            try:
                logger.debug(f"VannaManager: Generating SQL for question: {question}")

                # Generate SQL with allow_llm_to_see_data=True for database introspection
                sql = self.vanna_instance.generate_sql(
                    question=question, allow_llm_to_see_data=True
                )

                # Validate SQL response
                if not sql or sql.strip() == "":
                    raise ValueError("Empty SQL response")

                return sql

            except Exception as e:
                logger.error(f"VannaManager: Error in SQL generation: {e}")
                raise

    def force_reset(self):
        """
        Force reset the instance (useful for cleanup).
        """
        with self.lock:
            if self.vanna_instance:
                logger.debug(f"VannaManager: Resetting instance for {self.config_key}")
                self.vanna_instance = None

    def get_stats(self) -> Dict:
        """
        Get manager statistics.
        """
        return {
            "config_key": self.config_key,
            "instance_id": id(self.vanna_instance) if self.vanna_instance else None,
            "has_instance": self.vanna_instance is not None,
            "db_type": self.db_type,
        }

    @classmethod
    def create_with_config(
        cls,
        vanna_llm_config,
        vanna_embedder_config,
        vector_store_path: str,
        db_connection_string: str,
        db_type: str = "sqlite",
        training_data_path: str | None = None,
        nvidia_api_key: str | None = None,
    ):
        """
        Class method to create a VannaManager with full configuration.
        Uses create_config_key to ensure singleton behavior based on configuration.

        Args:
            vanna_llm_config: LLM configuration object
            vanna_embedder_config: Embedder configuration object
            vector_store_path: Path to ChromaDB vector store
            db_connection_string: Database connection string
            db_type: Type of database - 'sqlite', 'postgres', or 'sql'
            training_data_path: Path to YAML training data file
            nvidia_api_key: NVIDIA API key (optional)
        """
        config_key = create_config_key(
            vanna_llm_config,
            vanna_embedder_config,
            vector_store_path,
            db_connection_string,
            db_type,
        )

        # Create instance with just config_key (singleton pattern)
        instance = cls(config_key)

        # If this is a new instance that hasn't been configured yet, set the configuration
        if not hasattr(instance, "vanna_llm_config") or instance.vanna_llm_config is None:
            instance.vanna_llm_config = vanna_llm_config
            instance.vanna_embedder_config = vanna_embedder_config
            instance.vector_store_path = vector_store_path
            instance.db_connection_string = db_connection_string
            instance.db_type = db_type
            instance.training_data_path = training_data_path
            instance.nvidia_api_key = nvidia_api_key

            # Create Vanna instance immediately if all config is available
            if instance.vanna_instance is None:
                logger.debug("VannaManager: Creating Vanna instance for existing singleton")
                instance.vanna_instance = instance._create_instance()

        return instance


def create_config_key(
    vanna_llm_config,
    vanna_embedder_config,
    vector_store_path: str,
    db_connection_string: str,
    db_type: str = "sqlite",
) -> str:
    """
    Create a unique configuration key for the VannaManager singleton.

    Args:
        vanna_llm_config: LLM configuration object
        vanna_embedder_config: Embedder configuration object
        vector_store_path: Path to vector store
        db_connection_string: Database connection string
        db_type: Type of database

    Returns:
        str: Unique configuration key
    """
    config_str = f"{vanna_llm_config.model_name}_{vanna_embedder_config.model_name}_{vector_store_path}_{db_connection_string}_{db_type}"
    return hashlib.md5(config_str.encode()).hexdigest()[:12]

