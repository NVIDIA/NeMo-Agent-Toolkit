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

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_serializer


class FunctionDependencies(BaseModel):
    """
    A class to represent the dependencies of a function.
    """
    functions: set[str] = Field(default_factory=set)
    llms: set[str] = Field(default_factory=set)
    embedders: set[str] = Field(default_factory=set)
    memory_clients: set[str] = Field(default_factory=set)
    object_stores: set[str] = Field(default_factory=set)
    retrievers: set[str] = Field(default_factory=set)

    @field_serializer("functions", when_used="json")
    def serialize_functions(self, v: set[str]) -> list[str]:
        return list(v)

    @field_serializer("llms", when_used="json")
    def serialize_llms(self, v: set[str]) -> list[str]:
        return list(v)

    @field_serializer("embedders", when_used="json")
    def serialize_embedders(self, v: set[str]) -> list[str]:
        return list(v)

    @field_serializer("memory_clients", when_used="json")
    def serialize_memory_clients(self, v: set[str]) -> list[str]:
        return list(v)

    @field_serializer("object_stores", when_used="json")
    def serialize_object_stores(self, v: set[str]) -> list[str]:
        return list(v)

    @field_serializer("retrievers", when_used="json")
    def serialize_retrievers(self, v: set[str]) -> list[str]:
        return list(v)

    def add_function(self, function: str):
        self.functions.add(function)  # pylint: disable=no-member

    def add_llm(self, llm: str):
        self.llms.add(llm)  # pylint: disable=no-member

    def add_embedder(self, embedder: str):
        self.embedders.add(embedder)  # pylint: disable=no-member

    def add_memory_client(self, memory_client: str):
        self.memory_clients.add(memory_client)  # pylint: disable=no-member

    def add_object_store(self, object_store: str):
        self.object_stores.add(object_store)  # pylint: disable=no-member

    def add_retriever(self, retriever: str):
        self.retrievers.add(retriever)  # pylint: disable=no-member
