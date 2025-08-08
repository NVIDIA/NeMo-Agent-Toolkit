# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest

from _utils.configs import EmbedderProviderTestConfig
from _utils.configs import FunctionTestConfig
from _utils.configs import LLMProviderTestConfig
from _utils.configs import MemoryTestConfig
from _utils.configs import ObjectStoreTestConfig
from _utils.configs import RegistryHandlerTestConfig
from aiq.builder.builder import Builder
from aiq.builder.embedder import EmbedderProviderInfo
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function import Function
from aiq.builder.llm import LLMProviderInfo
from aiq.cli.register_workflow import register_embedder_client
from aiq.cli.register_workflow import register_embedder_provider
from aiq.cli.register_workflow import register_function
from aiq.cli.register_workflow import register_llm_client
from aiq.cli.register_workflow import register_llm_provider
from aiq.cli.register_workflow import register_memory
from aiq.cli.register_workflow import register_object_store
from aiq.cli.register_workflow import register_registry_handler
from aiq.cli.register_workflow import register_tool_wrapper
from aiq.cli.type_registry import TypeRegistry
from aiq.memory.interfaces import MemoryEditor
from aiq.memory.models import MemoryItem
from aiq.registry_handlers.registry_handler_base import AbstractRegistryHandler
from aiq.registry_handlers.schemas.package import PackageNameVersionList
from aiq.registry_handlers.schemas.publish import Artifact
from aiq.registry_handlers.schemas.publish import PublishResponse
from aiq.registry_handlers.schemas.pull import PullRequestPackages
from aiq.registry_handlers.schemas.pull import PullResponse
from aiq.registry_handlers.schemas.remove import RemoveResponse
from aiq.registry_handlers.schemas.search import SearchQuery
from aiq.registry_handlers.schemas.search import SearchResponse


def test_add_registration_changed_hook(registry: TypeRegistry):

    called = False

    def hook():
        nonlocal called
        called = True

    registry.add_registration_changed_hook(hook)

    @register_function(config_type=FunctionTestConfig)
    async def build_fn(config: FunctionTestConfig, builder: Builder):

        async def _arun():
            pass

        yield _arun

    assert called


def test_register_function(registry: TypeRegistry):
    with pytest.raises(KeyError):
        registry.get_function(FunctionTestConfig)

    @register_function(config_type=FunctionTestConfig)
    async def build_fn(config: FunctionTestConfig, builder: Builder):

        async def _arun():
            pass

        yield _arun

    func_info = registry.get_function(FunctionTestConfig)
    assert func_info.full_type == FunctionTestConfig.static_full_type()
    assert func_info.local_name == FunctionTestConfig.static_type()
    assert func_info.config_type is FunctionTestConfig
    assert func_info.build_fn is build_fn


def test_register_llm_provider(registry: TypeRegistry):
    with pytest.raises(KeyError):
        registry.get_llm_provider(LLMProviderTestConfig)

    @register_llm_provider(config_type=LLMProviderTestConfig)
    async def build_fn(config: LLMProviderTestConfig, builder: Builder):
        yield LLMProviderInfo(config=config, description="test llm")

    llm_info = registry.get_llm_provider(LLMProviderTestConfig)
    assert llm_info.full_type == LLMProviderTestConfig.static_full_type()
    assert llm_info.local_name == LLMProviderTestConfig.static_type()
    assert llm_info.config_type is LLMProviderTestConfig
    assert llm_info.build_fn is build_fn


def test_register_llm_client(registry: TypeRegistry):
    with pytest.raises(KeyError):
        registry.get_llm_client(LLMProviderTestConfig, LLMFrameworkEnum.LANGCHAIN)

    @register_llm_client(config_type=LLMProviderTestConfig, wrapper_type="test_framework")
    async def build_fn(config: LLMProviderTestConfig, builder: Builder):
        yield

    llm_client_info = registry.get_llm_client(LLMProviderTestConfig, "test_framework")

    assert llm_client_info.full_type == LLMProviderTestConfig.static_full_type()
    assert llm_client_info.local_name == LLMProviderTestConfig.static_type()
    assert llm_client_info.config_type is LLMProviderTestConfig
    assert llm_client_info.llm_framework == "test_framework"
    assert llm_client_info.build_fn is build_fn


def test_register_embedder_provider(registry: TypeRegistry):
    with pytest.raises(KeyError):
        registry.get_embedder_provider(EmbedderProviderTestConfig)

    @register_embedder_provider(config_type=EmbedderProviderTestConfig)
    async def build_fn(config: EmbedderProviderTestConfig, builder: Builder):
        yield EmbedderProviderInfo(config=config, description="test llm")

    embedder_provider_info = registry.get_embedder_provider(EmbedderProviderTestConfig)
    assert embedder_provider_info.full_type == EmbedderProviderTestConfig.static_full_type()
    assert embedder_provider_info.local_name == EmbedderProviderTestConfig.static_type()
    assert embedder_provider_info.config_type is EmbedderProviderTestConfig
    assert embedder_provider_info.build_fn is build_fn


def test_register_embedder_client(registry: TypeRegistry):
    with pytest.raises(KeyError):
        registry.get_embedder_client(EmbedderProviderTestConfig, LLMFrameworkEnum.LANGCHAIN)

    @register_embedder_client(config_type=EmbedderProviderTestConfig, wrapper_type="test_framework")
    async def build_fn(config: EmbedderProviderTestConfig, builder: Builder):
        yield

    embedder_client_info = registry.get_embedder_client(EmbedderProviderTestConfig, "test_framework")
    assert embedder_client_info.full_type == EmbedderProviderTestConfig.static_full_type()
    assert embedder_client_info.local_name == EmbedderProviderTestConfig.static_type()
    assert embedder_client_info.config_type is EmbedderProviderTestConfig
    assert embedder_client_info.llm_framework == "test_framework"
    assert embedder_client_info.build_fn is build_fn


def test_register_memory_client(registry: TypeRegistry):

    with pytest.raises(KeyError):
        registry.get_memory(MemoryTestConfig)

    @register_memory(config_type=MemoryTestConfig)
    async def build_fn(config: MemoryTestConfig, builder: Builder):

        class TestMemory(MemoryEditor):

            async def add_items(self, items: list[MemoryItem]) -> None:
                raise NotImplementedError

            async def search(self, query: str, top_k: int = 5, **kwargs) -> list[MemoryItem]:
                raise NotImplementedError

            async def remove_items(self, **kwargs) -> None:
                raise NotImplementedError

        yield TestMemory()

    memory_client_info = registry.get_memory(MemoryTestConfig)
    assert memory_client_info.full_type == MemoryTestConfig.static_full_type()
    assert memory_client_info.local_name == MemoryTestConfig.static_type()
    assert memory_client_info.config_type is MemoryTestConfig
    assert memory_client_info.build_fn is build_fn


def test_register_object_store(registry: TypeRegistry):

    with pytest.raises(KeyError):
        registry.get_object_store(ObjectStoreTestConfig)

    @register_object_store(config_type=ObjectStoreTestConfig)
    async def build_fn(config: ObjectStoreTestConfig, builder: Builder):
        yield

    object_store_info = registry.get_object_store(ObjectStoreTestConfig)
    assert object_store_info.full_type == ObjectStoreTestConfig.static_full_type()
    assert object_store_info.local_name == ObjectStoreTestConfig.static_type()
    assert object_store_info.config_type is ObjectStoreTestConfig
    assert object_store_info.build_fn is build_fn


def test_register_tool_wrapper(registry: TypeRegistry):
    with pytest.raises(KeyError):
        registry.get_tool_wrapper("test_framework")

    @register_tool_wrapper(wrapper_type="test_framework")
    def build_fn(name: str, fn: Function, builder: Builder):
        pass

    tool_wrapper_info = registry.get_tool_wrapper("test_framework")
    assert tool_wrapper_info.llm_framework == "test_framework"
    assert tool_wrapper_info.build_fn is build_fn


def test_register_registry_handler(registry: TypeRegistry):
    with pytest.raises(KeyError):
        registry.get_registry_handler("test_handler")

    @register_registry_handler(config_type=RegistryHandlerTestConfig)
    def build_fn(config: RegistryHandlerTestConfig):

        class TestRegistryHandler(AbstractRegistryHandler):

            @asynccontextmanager
            async def publish(self, artifact: Artifact) -> AsyncGenerator[PublishResponse]:
                raise NotImplementedError

            @asynccontextmanager
            async def pull(self, packages: PullRequestPackages) -> AsyncGenerator[PullResponse]:
                raise NotImplementedError

            @asynccontextmanager
            async def search(self, query: SearchQuery) -> AsyncGenerator[SearchResponse]:
                raise NotImplementedError

            @asynccontextmanager
            async def remove(self, packages: PackageNameVersionList) -> AsyncGenerator[RemoveResponse]:
                raise NotImplementedError

        yield TestRegistryHandler()

    registry_handler_info = registry.get_registry_handler(RegistryHandlerTestConfig)
    assert registry_handler_info.full_type == RegistryHandlerTestConfig.static_full_type()
    assert registry_handler_info.local_name == RegistryHandlerTestConfig.static_type()
    assert registry_handler_info.config_type is RegistryHandlerTestConfig
    assert registry_handler_info.build_fn is build_fn
