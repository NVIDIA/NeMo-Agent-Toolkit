# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pydantic import BaseModel

from _utils.configs import EmbedderProviderTestConfig
from _utils.configs import FunctionTestConfig
from _utils.configs import LLMProviderTestConfig
from _utils.configs import MemoryTestConfig
from _utils.configs import ObjectStoreTestConfig
from _utils.configs import PerUserFunctionTestConfig
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
from nat.cli.register_workflow import register_per_user_function
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


def test_register_per_user_function_with_single_output(registry: TypeRegistry):
    """Test per-user function registration with single output schema."""

    class PerUserInputSchema(BaseModel):
        message: str

    class PerUserOutputSchema(BaseModel):
        result: str

    with pytest.raises(KeyError):
        registry.get_function(PerUserFunctionTestConfig)

    @register_per_user_function(config_type=PerUserFunctionTestConfig,
                                input_type=PerUserInputSchema,
                                single_output_type=PerUserOutputSchema)
    async def build_fn(config: PerUserFunctionTestConfig, builder: Builder):

        async def _impl(inp: PerUserInputSchema) -> PerUserOutputSchema:
            return PerUserOutputSchema(result=inp.message)

        yield _impl

    func_info = registry.get_function(PerUserFunctionTestConfig)
    assert func_info.full_type == PerUserFunctionTestConfig.static_full_type()
    assert func_info.local_name == PerUserFunctionTestConfig.static_type()
    assert func_info.config_type is PerUserFunctionTestConfig
    assert func_info.build_fn is build_fn

    assert func_info.is_per_user is True
    assert func_info.per_user_function_input_schema is PerUserInputSchema
    assert func_info.per_user_function_single_output_schema is PerUserOutputSchema
    assert func_info.per_user_function_streaming_output_schema is None


def test_register_per_user_function_with_streaming(registry: TypeRegistry):
    """Test per-user function registration with streaming output schema."""

    class StreamInputSchema(BaseModel):
        text: str

    class StreamOutputSchema(BaseModel):
        chunk: str

    class PerUserStreamFunctionConfig(FunctionTestConfig, name="test_per_user_stream"):
        pass

    # Register with streaming output schema
    @register_per_user_function(config_type=PerUserStreamFunctionConfig,
                                input_type=StreamInputSchema,
                                streaming_output_type=StreamOutputSchema)
    async def build_fn(config: PerUserStreamFunctionConfig, builder: Builder):

        async def _impl(inp: StreamInputSchema):
            yield StreamOutputSchema(chunk=inp.text)

        yield _impl

    # Verify registration
    func_info = registry.get_function(PerUserStreamFunctionConfig)
    assert func_info.is_per_user is True
    assert func_info.per_user_function_input_schema is StreamInputSchema
    assert func_info.per_user_function_single_output_schema is None
    assert func_info.per_user_function_streaming_output_schema is StreamOutputSchema


def test_register_per_user_function_with_both_outputs(registry: TypeRegistry):
    """Test per-user function registration with both single and streaming output schemas."""

    class DualInputSchema(BaseModel):
        value: int

    class DualSingleOutputSchema(BaseModel):
        total: int

    class DualStreamOutputSchema(BaseModel):
        partial: int

    class PerUserDualFunctionConfig(FunctionTestConfig, name="test_per_user_dual"):
        pass

    # Register with both output schemas
    @register_per_user_function(config_type=PerUserDualFunctionConfig,
                                input_type=DualInputSchema,
                                single_output_type=DualSingleOutputSchema,
                                streaming_output_type=DualStreamOutputSchema)
    async def build_fn(config: PerUserDualFunctionConfig, builder: Builder):

        async def _impl(inp: DualInputSchema) -> DualSingleOutputSchema:
            return DualSingleOutputSchema(total=inp.value)

        yield _impl

    # Verify registration
    func_info = registry.get_function(PerUserDualFunctionConfig)
    assert func_info.is_per_user is True
    assert func_info.per_user_function_input_schema is DualInputSchema
    assert func_info.per_user_function_single_output_schema is DualSingleOutputSchema
    assert func_info.per_user_function_streaming_output_schema is DualStreamOutputSchema


def test_register_per_user_function_missing_output_schema(registry: TypeRegistry):
    """Test that registration fails when no output schema is provided."""

    class MissingOutputInputSchema(BaseModel):
        data: str

    class MissingOutputFunctionConfig(FunctionTestConfig, name="test_missing_output"):
        pass

    # Should fail validation - no output schema provided
    with pytest.raises(
            ValueError,
            match="per_user_function_single_output_schema or per_user_function_streaming_output_schema must be provided"
    ):

        @register_per_user_function(config_type=MissingOutputFunctionConfig, input_type=MissingOutputInputSchema)
        async def build_fn(config: MissingOutputFunctionConfig, builder: Builder):

            async def _impl(inp: MissingOutputInputSchema):
                pass

            yield _impl


def test_register_per_user_function_missing_input_schema(registry: TypeRegistry):
    """Test that registration fails when no input schema is provided."""

    class MissingInputOutputSchema(BaseModel):
        result: str

    class MissingInputFunctionConfig(FunctionTestConfig, name="test_missing_input"):
        pass

    # Should fail validation - no input schema provided
    with pytest.raises(ValueError, match="input_type must be provided to register a per-user function"):

        @register_per_user_function(
            config_type=MissingInputFunctionConfig,
            input_type=None,  # type: ignore
            single_output_type=MissingInputOutputSchema)
        async def build_fn(config: MissingInputFunctionConfig, builder: Builder):

            async def _impl():
                return MissingInputOutputSchema(result="test")

            yield _impl


def test_register_per_user_function_vs_regular_function(registry: TypeRegistry):
    """Test that per-user functions are distinguished from regular functions."""

    # Register a regular function
    class RegularFunctionConfig(FunctionTestConfig, name="test_regular"):
        pass

    @register_function(config_type=RegularFunctionConfig)
    async def regular_build_fn(config: RegularFunctionConfig, builder: Builder):

        async def _impl():
            pass

        yield _impl

    # Register a per-user function
    class PerUserCompareInputSchema(BaseModel):
        text: str

    class PerUserCompareOutputSchema(BaseModel):
        result: str

    class PerUserCompareFunctionConfig(FunctionTestConfig, name="test_per_user_compare"):
        pass

    @register_per_user_function(config_type=PerUserCompareFunctionConfig,
                                input_type=PerUserCompareInputSchema,
                                single_output_type=PerUserCompareOutputSchema)
    async def per_user_build_fn(config: PerUserCompareFunctionConfig, builder: Builder):

        async def _impl(inp: PerUserCompareInputSchema) -> PerUserCompareOutputSchema:
            return PerUserCompareOutputSchema(result=inp.text)

        yield _impl

    # Verify regular function is not per-user
    regular_func_info = registry.get_function(RegularFunctionConfig)
    assert regular_func_info.is_per_user is False
    assert regular_func_info.per_user_function_input_schema is None
    assert regular_func_info.per_user_function_single_output_schema is None
    assert regular_func_info.per_user_function_streaming_output_schema is None

    # Verify per-user function is marked correctly
    per_user_func_info = registry.get_function(PerUserCompareFunctionConfig)
    assert per_user_func_info.is_per_user is True
    assert per_user_func_info.per_user_function_input_schema is PerUserCompareInputSchema
    assert per_user_func_info.per_user_function_single_output_schema is PerUserCompareOutputSchema


def test_register_per_user_function_with_framework_wrappers(registry: TypeRegistry):
    """Test per-user function registration with framework wrappers."""

    class WrapperInputSchema(BaseModel):
        query: str

    class WrapperOutputSchema(BaseModel):
        answer: str

    class PerUserWrapperFunctionConfig(FunctionTestConfig, name="test_per_user_wrapper"):
        pass

    # Register with framework wrappers
    @register_per_user_function(config_type=PerUserWrapperFunctionConfig,
                                input_type=WrapperInputSchema,
                                single_output_type=WrapperOutputSchema,
                                framework_wrappers=["langchain", "llama_index"])
    async def build_fn(config: PerUserWrapperFunctionConfig, builder: Builder):

        async def _impl(inp: WrapperInputSchema) -> WrapperOutputSchema:
            return WrapperOutputSchema(answer=inp.query)

        yield _impl

    # Verify framework wrappers are registered
    func_info = registry.get_function(PerUserWrapperFunctionConfig)
    assert func_info.is_per_user is True
    assert func_info.framework_wrappers == ["langchain", "llama_index"]


# ==================== Simple Type Conversion Tests ====================


def test_register_per_user_function_with_simple_input_type(registry: TypeRegistry):
    """Test that simple input types (str, int) are converted to Pydantic models."""

    class SimpleInputOutputSchema(BaseModel):
        result: str

    class SimpleInputFunctionConfig(FunctionTestConfig, name="test_simple_input"):
        pass

    @register_per_user_function(config_type=SimpleInputFunctionConfig,
                                input_type=str,
                                single_output_type=SimpleInputOutputSchema)
    async def build_fn(config: SimpleInputFunctionConfig, builder: Builder):

        async def _impl(inp: str) -> SimpleInputOutputSchema:
            return SimpleInputOutputSchema(result=inp)

        yield _impl

    func_info = registry.get_function(SimpleInputFunctionConfig)
    assert func_info.is_per_user is True

    # The input schema should be a Pydantic model (not str directly)
    input_schema = func_info.per_user_function_input_schema
    assert input_schema is not None
    assert issubclass(input_schema, BaseModel)

    # The converted model should have a 'value' field of type str
    assert 'value' in input_schema.model_fields
    assert input_schema.model_fields['value'].annotation is str

    # Output schema should remain as-is (already a Pydantic model)
    assert func_info.per_user_function_single_output_schema is SimpleInputOutputSchema


def test_register_per_user_function_with_simple_output_type(registry: TypeRegistry):
    """Test that simple output types (str, int) are converted to Pydantic models."""

    class SimpleOutputInputSchema(BaseModel):
        query: str

    class SimpleOutputFunctionConfig(FunctionTestConfig, name="test_simple_output"):
        pass

    @register_per_user_function(config_type=SimpleOutputFunctionConfig,
                                input_type=SimpleOutputInputSchema,
                                single_output_type=str)
    async def build_fn(config: SimpleOutputFunctionConfig, builder: Builder):

        async def _impl(inp: SimpleOutputInputSchema) -> str:
            return inp.query

        yield _impl

    func_info = registry.get_function(SimpleOutputFunctionConfig)
    assert func_info.is_per_user is True

    # Input schema should remain as-is
    assert func_info.per_user_function_input_schema is SimpleOutputInputSchema

    # The output schema should be a Pydantic model (not str directly)
    output_schema = func_info.per_user_function_single_output_schema
    assert output_schema is not None
    assert issubclass(output_schema, BaseModel)

    # The converted model should have a 'value' field of type str
    assert 'value' in output_schema.model_fields
    assert output_schema.model_fields['value'].annotation is str


def test_register_per_user_function_with_all_simple_types(registry: TypeRegistry):
    """Test that all simple types (input and outputs) are converted to Pydantic models."""

    class AllSimpleFunctionConfig(FunctionTestConfig, name="test_all_simple"):
        pass

    @register_per_user_function(config_type=AllSimpleFunctionConfig,
                                input_type=str,
                                single_output_type=int,
                                streaming_output_type=float)
    async def build_fn(config: AllSimpleFunctionConfig, builder: Builder):

        async def _impl(inp: str) -> int:
            return len(inp)

        yield _impl

    func_info = registry.get_function(AllSimpleFunctionConfig)
    assert func_info.is_per_user is True

    # Verify input schema conversion
    input_schema = func_info.per_user_function_input_schema
    assert issubclass(input_schema, BaseModel)
    assert input_schema.model_fields['value'].annotation is str

    # Verify single output schema conversion
    single_output_schema = func_info.per_user_function_single_output_schema
    assert issubclass(single_output_schema, BaseModel)
    assert single_output_schema.model_fields['value'].annotation is int

    # Verify streaming output schema conversion
    streaming_output_schema = func_info.per_user_function_streaming_output_schema
    assert issubclass(streaming_output_schema, BaseModel)
    assert streaming_output_schema.model_fields['value'].annotation is float


def test_register_per_user_function_pydantic_model_unchanged(registry: TypeRegistry):
    """Test that Pydantic models are passed through unchanged."""

    class UnchangedInputSchema(BaseModel):
        message: str
        count: int

    class UnchangedOutputSchema(BaseModel):
        result: str

    class UnchangedFunctionConfig(FunctionTestConfig, name="test_unchanged"):
        pass

    @register_per_user_function(config_type=UnchangedFunctionConfig,
                                input_type=UnchangedInputSchema,
                                single_output_type=UnchangedOutputSchema)
    async def build_fn(config: UnchangedFunctionConfig, builder: Builder):

        async def _impl(inp: UnchangedInputSchema) -> UnchangedOutputSchema:
            return UnchangedOutputSchema(result=inp.message)

        yield _impl

    func_info = registry.get_function(UnchangedFunctionConfig)

    # Pydantic models should be passed through unchanged
    assert func_info.per_user_function_input_schema is UnchangedInputSchema
    assert func_info.per_user_function_single_output_schema is UnchangedOutputSchema


def test_register_per_user_function_with_complex_simple_type(registry: TypeRegistry):
    """Test conversion of more complex simple types like list[str]."""

    class ComplexSimpleFunctionConfig(FunctionTestConfig, name="test_complex_simple"):
        pass

    @register_per_user_function(config_type=ComplexSimpleFunctionConfig,
                                input_type=list[str],
                                single_output_type=dict[str, int])
    async def build_fn(config: ComplexSimpleFunctionConfig, builder: Builder):

        async def _impl(inp: list[str]) -> dict[str, int]:
            return {s: len(s) for s in inp}

        yield _impl

    func_info = registry.get_function(ComplexSimpleFunctionConfig)
    assert func_info.is_per_user is True

    # Verify input schema is a Pydantic model
    input_schema = func_info.per_user_function_input_schema
    assert issubclass(input_schema, BaseModel)
    assert 'value' in input_schema.model_fields

    # Verify output schema is a Pydantic model
    output_schema = func_info.per_user_function_single_output_schema
    assert issubclass(output_schema, BaseModel)
    assert 'value' in output_schema.model_fields
