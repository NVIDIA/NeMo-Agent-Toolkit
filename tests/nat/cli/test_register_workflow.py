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
from _utils.configs import TrainerAdapterTestConfig
from _utils.configs import TrainerTestConfig
from _utils.configs import TrajectoryBuilderTestConfig
from nat.builder.builder import Builder
from nat.builder.embedder import EmbedderProviderInfo
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_embedder_client
from nat.cli.register_workflow import register_embedder_provider
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_llm_client
from nat.cli.register_workflow import register_llm_provider
from nat.cli.register_workflow import register_memory
from nat.cli.register_workflow import register_object_store
from nat.cli.register_workflow import register_registry_handler
from nat.cli.register_workflow import register_tool_wrapper
from nat.cli.register_workflow import register_trainer
from nat.cli.register_workflow import register_trainer_adapter
from nat.cli.register_workflow import register_trajectory_builder
from nat.cli.type_registry import TypeRegistry
from nat.memory.interfaces import MemoryEditor
from nat.memory.models import MemoryItem
from nat.registry_handlers.registry_handler_base import AbstractRegistryHandler
from nat.registry_handlers.schemas.package import PackageNameVersionList
from nat.registry_handlers.schemas.publish import Artifact
from nat.registry_handlers.schemas.publish import PublishResponse
from nat.registry_handlers.schemas.pull import PullRequestPackages
from nat.registry_handlers.schemas.pull import PullResponse
from nat.registry_handlers.schemas.remove import RemoveResponse
from nat.registry_handlers.schemas.search import SearchQuery
from nat.registry_handlers.schemas.search import SearchResponse


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


def test_register_trainer(registry: TypeRegistry):
    """Test registration of trainer components"""
    with pytest.raises(KeyError):
        registry.get_trainer(TrainerTestConfig)

    @register_trainer(config_type=TrainerTestConfig)
    async def build_fn(config: TrainerTestConfig, builder: Builder):
        # For test purposes, just yield a mock object
        from unittest.mock import MagicMock
        mock_trainer = MagicMock()
        yield mock_trainer

    trainer_info = registry.get_trainer(TrainerTestConfig)
    assert trainer_info.full_type == TrainerTestConfig.static_full_type()
    assert trainer_info.local_name == TrainerTestConfig.static_type()
    assert trainer_info.config_type is TrainerTestConfig
    assert trainer_info.build_fn is build_fn


def test_register_trainer_adapter(registry: TypeRegistry):
    """Test registration of trainer adapter components"""
    with pytest.raises(KeyError):
        registry.get_trainer_adapter(TrainerAdapterTestConfig)

    @register_trainer_adapter(config_type=TrainerAdapterTestConfig)
    async def build_fn(config: TrainerAdapterTestConfig, builder: Builder):
        # For test purposes, just yield a mock object
        from unittest.mock import MagicMock
        mock_adapter = MagicMock()
        yield mock_adapter

    trainer_adapter_info = registry.get_trainer_adapter(TrainerAdapterTestConfig)
    assert trainer_adapter_info.full_type == TrainerAdapterTestConfig.static_full_type()
    assert trainer_adapter_info.local_name == TrainerAdapterTestConfig.static_type()
    assert trainer_adapter_info.config_type is TrainerAdapterTestConfig
    assert trainer_adapter_info.build_fn is build_fn


def test_register_trajectory_builder(registry: TypeRegistry):
    """Test registration of trajectory builder components"""
    with pytest.raises(KeyError):
        registry.get_trajectory_builder(TrajectoryBuilderTestConfig)

    @register_trajectory_builder(config_type=TrajectoryBuilderTestConfig)
    async def build_fn(config: TrajectoryBuilderTestConfig, builder: Builder):
        # For test purposes, just yield a mock object
        from unittest.mock import MagicMock
        mock_builder = MagicMock()
        yield mock_builder

    trajectory_builder_info = registry.get_trajectory_builder(TrajectoryBuilderTestConfig)
    assert trajectory_builder_info.full_type == TrajectoryBuilderTestConfig.static_full_type()
    assert trajectory_builder_info.local_name == TrajectoryBuilderTestConfig.static_type()
    assert trajectory_builder_info.config_type is TrajectoryBuilderTestConfig
    assert trajectory_builder_info.build_fn is build_fn
