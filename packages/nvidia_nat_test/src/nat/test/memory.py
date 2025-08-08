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

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_memory
from nat.data_models.memory import MemoryBaseConfig
from nat.memory.interfaces import MemoryEditor
from nat.memory.models import MemoryItem


class DummyMemoryConfig(MemoryBaseConfig, name="test_dummy"):
    pass


@register_memory(config_type=DummyMemoryConfig)
async def echo_function(config: DummyMemoryConfig, builder: Builder):

    class DummyMemoryEditor(MemoryEditor):

        async def add_items(self, items: list[MemoryItem]) -> None:
            pass

        async def search(self, query: str, top_k: int = 5, **kwargs) -> list[MemoryItem]:
            return []

        async def remove_items(self, **kwargs) -> None:
            pass

    yield DummyMemoryEditor()
