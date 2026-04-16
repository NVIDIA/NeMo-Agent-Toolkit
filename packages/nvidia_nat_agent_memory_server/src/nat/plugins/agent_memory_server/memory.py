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

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_memory
from nat.data_models.memory import MemoryBaseConfig

from .agent_memory_editor import AgentMemoryServerEditor


class AgentMemoryServerMemoryConfig(MemoryBaseConfig, name="agent_memory_server"):
    """Config for Redis Agent Memory Server as a NAT memory type."""

    base_url: str = Field(
        default="http://localhost:8000",
        description="Agent Memory Server base URL (e.g. http://localhost:8000).",
    )
    default_namespace: str = Field(
        default="nat",
        description="Default namespace for the memory client.",
    )


@register_memory(config_type=AgentMemoryServerMemoryConfig)
async def agent_memory_server_memory_client(config: AgentMemoryServerMemoryConfig, builder: Builder):
    from agent_memory_client import create_memory_client

    client = await create_memory_client(
        base_url=config.base_url,
        default_namespace=config.default_namespace,
    )
    try:
        editor = AgentMemoryServerEditor(client)
        yield editor
    finally:
        await client.close()
