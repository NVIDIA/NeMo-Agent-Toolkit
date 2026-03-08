# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_memory
from nat.data_models.memory import MemoryBaseConfig


class SelfMemoryProviderConfig(MemoryBaseConfig, name="selfmemory"):
    vector_store_provider: str = "qdrant"
    vector_store_config: dict = {}
    embedding_provider: str = "openai"
    embedding_config: dict = {}
    llm_provider: str | None = None
    llm_config: dict = {}
    encryption_key: str | None = None


@register_memory(config_type=SelfMemoryProviderConfig)
async def selfmemory_provider(config: SelfMemoryProviderConfig, builder: Builder):
    import os

    from selfmemory import SelfMemory

    from nat.plugins.selfmemory.selfmemory_editor import SelfMemoryEditor

    config_dict = {
        "vector_store": {
            "provider": config.vector_store_provider,
            "config": config.vector_store_config,
        },
        "embedding": {
            "provider": config.embedding_provider,
            "config": config.embedding_config,
        },
    }

    if config.llm_provider:
        config_dict["llm"] = {
            "provider": config.llm_provider,
            "config": config.llm_config,
        }

    encryption_key = config.encryption_key or os.environ.get("MASTER_ENCRYPTION_KEY")

    if encryption_key:
        os.environ["MASTER_ENCRYPTION_KEY"] = encryption_key

    memory = SelfMemory(config=config_dict)

    try:
        yield SelfMemoryEditor(memory)
    finally:
        memory.close()
