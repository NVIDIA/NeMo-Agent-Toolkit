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

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_memory
from aiq.data_models.memory import MemoryBaseConfig


class Mem0MemoryClientConfig(MemoryBaseConfig, name="mem0_memory"):
    host: str | None = None
    organization: str | None = None
    project: str | None = None
    org_id: str | None = None
    project_id: str | None = None


@register_memory(config_type=Mem0MemoryClientConfig)
async def mem0_memory_client(config: Mem0MemoryClientConfig, builder: Builder):
    import os

    from mem0 import AsyncMemoryClient

    from aiq.plugins.mem0ai.mem0_editor import Mem0Editor

    mem0_api_key = os.environ.get("MEM0_API_KEY")

    if mem0_api_key is None:
        raise RuntimeError("Mem0 API key is not set. Please specify it in the environment variable 'MEM0_API_KEY'.")

    mem0_client = AsyncMemoryClient(api_key=mem0_api_key,
                                    host=config.host,
                                    org_id=config.org_id,
                                    project_id=config.project_id)

    memory_editor = Mem0Editor(mem0_client=mem0_client)

    yield memory_editor
