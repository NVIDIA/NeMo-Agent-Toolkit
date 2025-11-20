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
"""Shared fixtures and test configurations for builder tests."""

import pytest
from pydantic import Field

from nat.authentication.interfaces import AuthProviderBase
from nat.builder.builder import Builder
from nat.builder.embedder import EmbedderProviderInfo
from nat.builder.function import Function
from nat.builder.function import FunctionGroup
from nat.builder.function_info import FunctionInfo
from nat.builder.llm import LLMProviderInfo
from nat.builder.retriever import RetrieverProviderInfo
from nat.cli.register_workflow import register_auth_provider
from nat.cli.register_workflow import register_embedder_provider
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_function_group
from nat.cli.register_workflow import register_llm_provider
from nat.cli.register_workflow import register_memory
from nat.cli.register_workflow import register_object_store
from nat.cli.register_workflow import register_retriever_provider
from nat.cli.register_workflow import register_telemetry_exporter
from nat.cli.register_workflow import register_ttc_strategy
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.authentication import AuthResult
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.memory import MemoryBaseConfig
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.data_models.retriever import RetrieverBaseConfig
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.memory.interfaces import MemoryEditor
from nat.memory.models import MemoryItem
from nat.object_store.in_memory_object_store import InMemoryObjectStore
from nat.observability.exporter.base_exporter import BaseExporter


class FunctionReturningFunctionConfig(FunctionBaseConfig, name="fn_return_fn"):
    pass


class FunctionReturningInfoConfig(FunctionBaseConfig, name="fn_return_info"):
    pass


class FunctionReturningDerivedConfig(FunctionBaseConfig, name="fn_return_derived"):
    pass


class TAuthConfig(AuthProviderBaseConfig, name="test_auth"):
    pass


class TLLMProviderConfig(LLMBaseConfig, name="test_llm"):
    raise_error: bool = False


class TEmbedderProviderConfig(EmbedderBaseConfig, name="test_embedder_provider"):
    raise_error: bool = False


class TMemoryConfig(MemoryBaseConfig, name="test_memory"):
    raise_error: bool = False


class TRetrieverProviderConfig(RetrieverBaseConfig, name="test_retriever"):
    raise_error: bool = False


class TTelemetryExporterConfig(TelemetryExporterBaseConfig, name="test_telemetry_exporter"):
    raise_error: bool = False


class TObjectStoreConfig(ObjectStoreBaseConfig, name="test_object_store"):
    raise_error: bool = False


class TTTCStrategyConfig(TTCStrategyBaseConfig, name="test_ttc_strategy"):
    raise_error: bool = False


class FailingFunctionConfig(FunctionBaseConfig, name="failing_function"):
    pass


# Function Group Test Configurations
class IncludesFunctionGroupConfig(FunctionGroupBaseConfig, name="test_includes_function_group"):
    """Test configuration for function groups."""
    include: list[str] = Field(default_factory=lambda: ["add", "multiply"])
    raise_error: bool = False


class ExcludesFunctionGroupConfig(FunctionGroupBaseConfig, name="test_excludes_function_group"):
    """Test configuration for function groups."""
    exclude: list[str] = Field(default_factory=lambda: ["add", "multiply"])
    raise_error: bool = False


class DefaultFunctionGroup(FunctionGroupBaseConfig, name="default_function_group"):
    """Test configuration with no included functions."""
    exclude: list[str] = Field(default_factory=lambda: ["internal_function"])  # Exclude the only function
    raise_error: bool = False


class AllIncludesFunctionGroupConfig(FunctionGroupBaseConfig, name="all_includes_function_group"):
    """Test configuration that includes all functions."""
    include: list[str] = Field(default_factory=lambda: ["add", "multiply", "subtract"])
    raise_error: bool = False


class AllExcludesFunctionGroupConfig(FunctionGroupBaseConfig, name="all_excludes_function_group"):
    """Test configuration that includes all functions."""
    exclude: list[str] = Field(default_factory=lambda: ["add", "multiply", "subtract"])
    raise_error: bool = False


class FailingFunctionGroupConfig(FunctionGroupBaseConfig, name="failing_function_group"):
    """Test configuration for function group that fails during initialization."""
    raise_error: bool = True


@pytest.fixture(scope="module", autouse=True)
async def _register():

    @register_auth_provider(config_type=TAuthConfig)
    async def register_test_auth(config: TAuthConfig, b: Builder):
        """Register test authentication provider."""

        class TestAuthProvider(AuthProviderBase[TAuthConfig]):
            """Mock auth provider for testing."""

            async def authenticate(self, user_id: str | None = None, **kwargs) -> AuthResult:
                """Mock authentication that always succeeds."""
                return AuthResult()

        yield TestAuthProvider(config)

    @register_function(config_type=TSharedFunctionConfig)
    async def register_shared_function(config: TSharedFunctionConfig, b: Builder):
        """Register shared function."""

        async def _shared_fn(some_input: str) -> str:
            return f"This is a shared function: {some_input}"

        yield _shared_fn

    @register_function(config_type=TPerUserFunctionConfig)
    async def register_per_user_function(config: TPerUserFunctionConfig, b: Builder):
        """Register per-user function."""

        async def _per_user_fn(some_input: str) -> str:
            return f"This is a per-user function: {some_input}"

        yield _per_user_fn

    @register_function(config_type=FunctionReturningFunctionConfig)
    async def register1(config: FunctionReturningFunctionConfig, b: Builder):

        async def _inner(some_input: str) -> str:
            return some_input + "!"

        yield _inner

    @register_function(config_type=FunctionReturningInfoConfig)
    async def register2(config: FunctionReturningInfoConfig, b: Builder):

        async def _inner(some_input: str) -> str:
            return some_input + "!"

        def _convert(int_input: int) -> str:
            return str(int_input)

        yield FunctionInfo.from_fn(_inner, converters=[_convert])

    @register_function(config_type=FunctionReturningDerivedConfig)
    async def register3(config: FunctionReturningDerivedConfig, b: Builder):

        class DerivedFunction(Function[str, str, str]):

            def __init__(self, config: FunctionReturningDerivedConfig):
                super().__init__(config=config, description="Test function")

            def some_method(self, val):
                return "some_method" + val

            async def _ainvoke(self, value: str) -> str:
                return value + "!"

            async def _astream(self, value: str):
                yield value + "!"

        yield DerivedFunction(config)

    @register_function(config_type=FailingFunctionConfig)
    async def register_failing_function(config: FailingFunctionConfig, b: Builder):
        # This function always raises an exception during initialization
        raise ValueError("Function initialization failed")
        yield  # This line will never be reached, but needed for the AsyncGenerator type

    @register_llm_provider(config_type=TLLMProviderConfig)
    async def register4(config: TLLMProviderConfig, b: Builder):

        if (config.raise_error):
            raise ValueError("Error")

        yield LLMProviderInfo(config=config, description="A test client.")

    @register_embedder_provider(config_type=TEmbedderProviderConfig)
    async def register5(config: TEmbedderProviderConfig, b: Builder):

        if (config.raise_error):
            raise ValueError("Error")

        yield EmbedderProviderInfo(config=config, description="A test client.")

    @register_memory(config_type=TMemoryConfig)
    async def register6(config: TMemoryConfig, b: Builder):

        if (config.raise_error):
            raise ValueError("Error")

        class TestMemoryEditor(MemoryEditor):

            async def add_items(self, items: list[MemoryItem]) -> None:
                raise NotImplementedError

            async def search(self, query: str, top_k: int = 5, **kwargs) -> list[MemoryItem]:
                raise NotImplementedError

            async def remove_items(self, **kwargs) -> None:
                raise NotImplementedError

        yield TestMemoryEditor()

    # Register mock provider
    @register_retriever_provider(config_type=TRetrieverProviderConfig)
    async def register7(config: TRetrieverProviderConfig, _builder: Builder):

        if (config.raise_error):
            raise ValueError("Error")

        yield RetrieverProviderInfo(config=config, description="Mock retriever to test the registration process")

    @register_object_store(config_type=TObjectStoreConfig)
    async def register8(config: TObjectStoreConfig, _builder: Builder):
        if (config.raise_error):
            raise ValueError("Error")

        yield InMemoryObjectStore()

    # Register mock telemetry exporter
    @register_telemetry_exporter(config_type=TTelemetryExporterConfig)
    async def register9(config: TTelemetryExporterConfig, _builder: Builder):

        if (config.raise_error):
            raise ValueError("Error")

        class TestTelemetryExporter(BaseExporter):

            def export(self, event: IntermediateStep):
                pass

        yield TestTelemetryExporter()

    @register_ttc_strategy(config_type=TTTCStrategyConfig)
    async def register_ttc(config: TTTCStrategyConfig, _builder: Builder):

        if config.raise_error:
            raise ValueError("Error")

        class DummyTTCStrategy(StrategyBase):
            """Very small pass-through strategy used only for testing."""

            async def ainvoke(self,
                              items: list,
                              original_prompt: str | None = None,
                              agent_context: str | None = None,
                              **kwargs) -> list:
                # Do nothing, just return what we got
                return items

            async def build_components(self, builder: Builder) -> None:
                pass

            def supported_pipeline_types(self) -> list[PipelineTypeEnum]:
                return [PipelineTypeEnum.AGENT_EXECUTION]

            def stage_type(self) -> StageTypeEnum:
                return StageTypeEnum.SCORING

        yield DummyTTCStrategy(config)

    # Function Group registrations
    @register_function_group(config_type=IncludesFunctionGroupConfig)
    async def register_test_includes_function_group(config: IncludesFunctionGroupConfig, _builder: Builder):
        """Register a test function group with basic arithmetic operations."""

        if config.raise_error:
            raise ValueError("Function group initialization failed")

        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        async def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        async def subtract(a: int, b: int) -> int:
            """Subtract two numbers."""
            return a - b

        group = FunctionGroup(config=config)

        group.add_function("add", add, description="Add two numbers")
        group.add_function("multiply", multiply, description="Multiply two numbers")
        group.add_function("subtract", subtract, description="Subtract two numbers")

        yield group

    @register_function_group(config_type=ExcludesFunctionGroupConfig)
    async def register_test_excludes_function_group(config: ExcludesFunctionGroupConfig, _builder: Builder):
        """Register a test function group with basic arithmetic operations."""

        if config.raise_error:
            raise ValueError("Function group initialization failed")

        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        async def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        async def subtract(a: int, b: int) -> int:
            """Subtract two numbers."""
            return a - b

        group = FunctionGroup(config=config)

        group.add_function("add", add, description="Add two numbers")
        group.add_function("multiply", multiply, description="Multiply two numbers")
        group.add_function("subtract", subtract, description="Subtract two numbers")

        yield group

    @register_function_group(config_type=DefaultFunctionGroup)
    async def register_empty_includes_group(config: DefaultFunctionGroup, _builder: Builder):
        """Register a function group with no included functions."""

        if config.raise_error:
            raise ValueError("Function group initialization failed")

        async def internal_function(x: int) -> int:
            """Internal function that is not included."""
            return x * 2

        group = FunctionGroup(config=config)

        group.add_function("internal_function", internal_function, description="Internal function")

        yield group

    @register_function_group(config_type=AllIncludesFunctionGroupConfig)
    async def register_all_includes_group(config: AllIncludesFunctionGroupConfig, _builder: Builder):
        """Register a function group that includes all functions."""

        if config.raise_error:
            raise ValueError("Function group initialization failed")

        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        async def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        async def subtract(a: int, b: int) -> int:
            """Subtract two numbers."""
            return a - b

        group = FunctionGroup(config=config)

        group.add_function("add", add, description="Add two numbers")
        group.add_function("multiply", multiply, description="Multiply two numbers")
        group.add_function("subtract", subtract, description="Subtract two numbers")

        yield group

    @register_function_group(config_type=AllExcludesFunctionGroupConfig)
    async def register_all_excludes_group(config: AllExcludesFunctionGroupConfig, _builder: Builder):
        """Register a function group that excludes all functions."""

        if config.raise_error:
            raise ValueError("Function group initialization failed")

        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        async def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        async def subtract(a: int, b: int) -> int:
            """Subtract two numbers."""
            return a - b

        group = FunctionGroup(config=config)

        group.add_function("add", add, description="Add two numbers")
        group.add_function("multiply", multiply, description="Multiply two numbers")
        group.add_function("subtract", subtract, description="Subtract two numbers")

        yield group

    @register_function_group(config_type=FailingFunctionGroupConfig)
    async def register_failing_function_group(config: FailingFunctionGroupConfig, _builder: Builder):
        """Register a function group that always fails during initialization."""

        # This function group always raises an exception during initialization
        raise ValueError("Function group initialization failed")
        yield  # This line will never be reached, but needed for the AsyncGenerator type
