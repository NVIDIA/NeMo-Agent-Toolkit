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

from contextlib import asynccontextmanager

from nat.builder.workflow_builder import WorkflowBuilder
from nat.object_store.in_memory_object_store import InMemoryObjectStoreConfig
from nat.test.object_store_tests import ObjectStoreTests


class TestInMemoryObjectStore(ObjectStoreTests):

    @asynccontextmanager
    async def _get_store(self):
        async with WorkflowBuilder() as builder:
            await builder.add_object_store("object_store_name", InMemoryObjectStoreConfig())

            yield await builder.get_object_store_client("object_store_name")
