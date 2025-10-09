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
import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic import HttpUrl

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.retry_mixin import RetryMixin

logger = logging.getLogger(__name__)


class NvidiaRAGLibConfig(FunctionBaseConfig, RetryMixin, name="nvidia_rag_lib"):
    """Configuration for NVIDIA RAG Library integration.

    This configuration manages the setup and instantiation of the NVIDIA RAG library,
    providing retrieval-augmented generation capabilities including document ingestion,
    vector search, reranking, and response generation. The configuration handles
    environment setup, model references, deployment modes, and operational parameters.
    """

    # Core RAG configuration
    vdb_endpoint: HttpUrl = Field(default=HttpUrl("http://localhost:19530"), description="Vector database endpoint URL")
    reranker_top_k: int = Field(default=10, description="Number of top results to rerank")
    vdb_top_k: int = Field(default=100, description="Number of top results from vector database")
    collection_names: list[str] | None = Field(default=None, description="List of collection names to use for queries")

    # Document processing
    chunk_size: int = Field(default=512, description="Size of document chunks for processing")
    chunk_overlap: int = Field(default=150, description="Overlap between document chunks")
    generate_summary: bool = Field(default=False, description="Whether to generate document summaries")
    blocking_upload: bool = Field(default=False, description="Whether to use blocking upload for documents")

    # Infrastructure
    vectorstore_gpu_device_id: str | None = Field(default="0", description="GPU device ID for vector store")
    model_directory: str | None = Field(default="~/.cache/model-cache", description="Directory for model cache")

    # Model configuration (NAT component references)
    llm_name: LLMRef | None = Field(default=None, description="Reference to the LLM to use for responses")
    embedder_name: EmbedderRef | None = Field(default=None,
                                              description="Reference to the embedder to use for embeddings")
    ranking_modelname: str | None = Field(default=None, description="Name of the ranking model")

    # Service endpoints
    app_embeddings_serverurl: str | None = Field(default="", description="Embeddings service URL")
    app_llm_serverurl: str | None = Field(default="", description="LLM service URL")
    app_ranking_serverurl: str | None = Field(default=None, description="Ranking service URL")

    # Deployment and operational
    deployment_mode: Literal["on_prem", "hosted", "mixed"] = Field(default="hosted",
                                                                   description="Deployment mode for the RAG system")
    timeout: float | None = Field(default=60.0, description="Timeout for operations in seconds")

    # Setup and management options
    env_library_path: str | None = Field(default=None, description="Path to .env_library file for environment setup")
    use_accuracy_profile: bool = Field(default=False, description="Load accuracy profile settings")
    use_perf_profile: bool = Field(default=False, description="Load performance profile settings")
    verify_prerequisites: bool = Field(default=True, description="Verify prerequisites before initialization")
    health_check_dependencies: bool = Field(default=True, description="Perform health check on dependent services")
    health_check_timeout: float = Field(default=30.0, description="Timeout in seconds for health check operations")


async def _load_env_library(config: NvidiaRAGLibConfig) -> None:
    """Load environment variables from a specified .env_library file.

    This function loads environment variables from an external environment
    library file if specified in the configuration. This allows for
    centralized environment management across multiple RAG deployments.

    Args:
        config: Configuration containing the path to the environment library file
    """
    if not config.env_library_path:
        return

    env_path = Path(config.env_library_path).expanduser()
    if not env_path.exists():
        logger.warning("Environment library file not found: %s", env_path)
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        logger.info("Loaded environment library: %s", env_path)
    except ImportError:
        logger.warning("python-dotenv not available, skipping .env_library loading")


async def _setup_environment_variables(config: NvidiaRAGLibConfig, builder: Builder) -> None:
    """Configure environment variables required by the NVIDIA RAG library.

    This function sets up all necessary environment variables that the NVIDIA RAG
    library expects, including vector database endpoints, model configurations,
    document processing parameters, and infrastructure settings. Model names are
    dynamically extracted from NAT component references when available.

    Args:
        config: Configuration containing RAG parameters and component references
        builder: NAT builder instance for accessing LLM and embedder configurations
    """
    # Core configuration
    if config.vdb_endpoint:
        os.environ["VDB_ENDPOINT"] = str(config.vdb_endpoint)

    os.environ["RERANKER_TOP_K"] = str(config.reranker_top_k)
    os.environ["VDB_TOP_K"] = str(config.vdb_top_k)

    if config.collection_names:
        os.environ["COLLECTION_NAMES"] = ",".join(config.collection_names)

    # Document processing
    os.environ["CHUNK_SIZE"] = str(config.chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(config.chunk_overlap)
    os.environ["GENERATE_SUMMARY"] = str(config.generate_summary).lower()
    os.environ["BLOCKING_UPLOAD"] = str(config.blocking_upload).lower()

    # Infrastructure
    if config.vectorstore_gpu_device_id is not None:
        os.environ["VECTORSTORE_GPU_DEVICE_ID"] = config.vectorstore_gpu_device_id

    if config.model_directory:
        model_dir = Path(config.model_directory).expanduser()
        os.environ["MODEL_DIRECTORY"] = str(model_dir)

    # Model names from NAT component references
    if config.llm_name:
        try:
            llm_config = builder.get_llm_config(config.llm_name)
            model_name = getattr(llm_config, 'model_name', None)
            if model_name:
                os.environ["APP_LLM_MODELNAME"] = str(model_name)
                logger.debug("Set APP_LLM_MODELNAME from LLM reference: %s", model_name)
        except Exception as e:
            logger.warning("Failed to get LLM config for %s: %s", config.llm_name, e)

    if config.embedder_name:
        try:
            embedder_config = builder.get_embedder_config(config.embedder_name)
            model_name = getattr(embedder_config, 'model_name', None)
            if model_name:
                os.environ["APP_EMBEDDINGS_MODELNAME"] = str(model_name)
                logger.debug("Set APP_EMBEDDINGS_MODELNAME from embedder reference: %s", model_name)
        except Exception as e:
            logger.warning("Failed to get embedder config for %s: %s", config.embedder_name, e)

    if config.ranking_modelname:
        os.environ["APP_RANKING_MODELNAME"] = config.ranking_modelname

    # Service URLs
    if config.app_embeddings_serverurl is not None:
        os.environ["APP_EMBEDDINGS_SERVERURL"] = config.app_embeddings_serverurl
    if config.app_llm_serverurl is not None:
        os.environ["APP_LLM_SERVERURL"] = config.app_llm_serverurl
    if config.app_ranking_serverurl is not None:
        os.environ["APP_RANKING_SERVERURL"] = config.app_ranking_serverurl

    # Deployment mode
    os.environ["DEPLOYMENT_MODE"] = config.deployment_mode


async def _load_profiles(config: NvidiaRAGLibConfig) -> None:
    """Load accuracy and performance optimization profiles.

    This function loads predefined environment configurations for accuracy
    or performance optimization. These profiles contain environment variable
    settings that optimize the RAG system for specific use cases.

    Args:
        config: Configuration specifying which profiles to load
    """
    if config.use_accuracy_profile:
        accuracy_profile_path = Path("accuracy_profile.env")
        if accuracy_profile_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(accuracy_profile_path)
                logger.info("Loaded accuracy profile")
            except ImportError:
                logger.warning("python-dotenv not available, skipping accuracy profile")
        else:
            logger.warning("Accuracy profile file not found: %s", accuracy_profile_path)

    if config.use_perf_profile:
        perf_profile_path = Path("perf_profile.env")
        if perf_profile_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(perf_profile_path)
                logger.info("Loaded performance profile")
            except ImportError:
                logger.warning("python-dotenv not available, skipping performance profile")
        else:
            logger.warning("Performance profile file not found: %s", perf_profile_path)


def _verify_prerequisites(config: NvidiaRAGLibConfig) -> None:
    """Verify that all required prerequisites are met before initialization.

    This function checks that necessary directories exist, dependencies are
    available, and the system is properly configured for RAG operations.

    Args:
        config: Configuration containing prerequisite specifications
    """
    # Check model directory
    if config.model_directory:
        model_dir = Path(config.model_directory).expanduser()
        if not model_dir.exists():
            logger.warning("Model directory does not exist: %s", model_dir)


@register_function(config_type=NvidiaRAGLibConfig)
async def nvidia_rag_lib(config: NvidiaRAGLibConfig, builder: Builder):
    """
    Initialize and configure the NVIDIA RAG library client.

    This function orchestrates the complete setup process for the NVIDIA RAG library,
    including environment configuration, profile loading, prerequisite verification,
    and service health checks. It yields a query function that provides access to
    the fully configured RAG system for retrieval-augmented generation operations.

    Args:
        config: Configuration parameters for the NVIDIA RAG library
        builder: NAT builder instance for accessing other components

    Yields:
        NvidiaRAG: Fully configured NVIDIA RAG library client instance

    Raises:
        ImportError: If the nvidia-rag library is not installed
        RuntimeError: If required prerequisites are not met
    """
    logger.info("Starting NVIDIA RAG setup...")

    try:
        # Step 1: Load .env_library if available
        if config.env_library_path:
            logger.debug("Loading environment library: %s", config.env_library_path)
            await _load_env_library(config)

        # Step 2: Set up environment variables
        logger.debug("Setting up environment variables...")
        await _setup_environment_variables(config, builder)

        # Step 3: Load profiles if requested
        if config.use_accuracy_profile or config.use_perf_profile:
            logger.debug("Loading performance profiles...")
            await _load_profiles(config)

        # Step 4: Verify prerequisites
        if config.verify_prerequisites:
            logger.debug("Verifying prerequisites...")
            _verify_prerequisites(config)

        # Step 5: Import and instantiate the NVIDIA RAG library
        logger.debug("Importing NVIDIA RAG library...")
        from nvidia_rag import NvidiaRAG

        rag = NvidiaRAG()

        # Step 6: Health check if requested
        if config.health_check_dependencies:
            logger.debug("Performing health check...")
            try:
                health_status = await asyncio.wait_for(rag.health(check_dependencies=True),
                                                       timeout=config.health_check_timeout)
                logger.info("Health check completed: %s", health_status)
            except TimeoutError:
                logger.warning("Health check timed out after %ss, but continuing", config.health_check_timeout)
            except Exception as e:
                logger.warning("Health check failed, but continuing: %s", e)

        # Yield the RAG instance
        yield rag

    except ImportError as e:
        logger.error("nvidia_rag library not available. Install with: pip install nvidia-rag. Error: %s", e)
        raise ImportError("nvidia-rag is required for this function. "
                          "Follow installation steps: pip install nvidia-rag --force-reinstall") from e
    except Exception as e:
        logger.error("Failed to set up NVIDIA RAG: %s", e)
        raise
    finally:
        pass
