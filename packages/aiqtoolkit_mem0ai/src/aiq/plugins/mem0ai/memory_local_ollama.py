# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_memory
from aiq.data_models.memory import MemoryBaseConfig


class Mem0LocalMemoryClientConfig(MemoryBaseConfig, name="mem0_memory_local_ollama"):
    """
    Mem0 Memory Client Configuration. Setup for use with local Ollama instaces.
    """
    # Defaults are set to work with Ollama and Milvus local instances with a local Mem0 instance
    # change them according to your local setup or override them in your workflow config file
    vec_store_provider: str = "milvus"  # Change to "qdrant" if you prefer that
    vec_store_collection_name: str = "DefaultAIQCollectionNew"
    vec_store_url: str = "http://localhost:19530"  # Default Local Milvus URL, change if needed
    vec_store_embedding_model_dims: int = 1024  # Updated to match the actual embedding dimensions
    llm_provider: str = "ollama"
    llm_model: str = "aliafshar/gemma3-it-qat-tools:27b"  # Change to your preferred model
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2000
    llm_base_url: str = "http://localhost:11434"  # Default Ollama URL, change if needed
    embedder_provider: str = "ollama"
    embedder_model: str = "snowflake-arctic-embed2:latest"
    embedder_base_url: str = "http://localhost:11434"  # Default Ollama URL, change if needed


@register_memory(config_type=Mem0LocalMemoryClientConfig)
async def mem0_memory_client(config: Mem0LocalMemoryClientConfig, builder: Builder):
    # UPDATED: Import AsyncMemory for v2 API compatibility
    from mem0 import AsyncMemory

    from aiq.plugins.mem0ai.mem0_editor import Mem0Editor

    # UPDATED: Create configuration dictionary for AsyncMemory
    # This includes all the necessary configuration for the local embedder, LLM, and vector store
    config_dict = {
        "vector_store": {
            "provider": config.vec_store_provider,
            "config": {
                "collection_name": config.vec_store_collection_name,
                "url": config.vec_store_url,
                "embedding_model_dims": config.vec_store_embedding_model_dims,
            },
        },
        "llm": {
            "provider": config.llm_provider,
            "config": {
                "model": config.llm_model,
                "temperature": config.llm_temperature,
                "max_tokens": config.llm_max_tokens,
                "ollama_base_url": config.llm_base_url,
            },
        },
        "embedder": {
            "provider": config.embedder_provider,
            "config": {
                "model": config.embedder_model,
                "ollama_base_url": config.embedder_base_url,
            },
        },
    }

    # UPDATED: Initialize AsyncMemory with the configuration
    # This is compatible with the v2 API and the updated mem0_editor.py
    # Use from_config to create an AsyncMemory instance from a dictionary
    mem0_client = await AsyncMemory.from_config(config_dict)

    memory_editor = Mem0Editor(mem0_client=mem0_client)

    yield memory_editor
