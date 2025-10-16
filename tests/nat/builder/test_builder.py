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

import logging
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.embedder import EmbedderProviderInfo
from nat.builder.function import Function
from nat.builder.function import FunctionGroup
from nat.builder.function_info import FunctionInfo
from nat.builder.llm import LLMProviderInfo
from nat.builder.retriever import RetrieverProviderInfo
from nat.builder.workflow import Workflow
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_embedder_client
from nat.cli.register_workflow import register_embedder_provider
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_function_group
from nat.cli.register_workflow import register_llm_client
from nat.cli.register_workflow import register_llm_provider
from nat.cli.register_workflow import register_memory
from nat.cli.register_workflow import register_object_store
from nat.cli.register_workflow import register_retriever_client
from nat.cli.register_workflow import register_retriever_provider
from nat.cli.register_workflow import register_telemetry_exporter
from nat.cli.register_workflow import register_tool_wrapper
from nat.cli.register_workflow import register_ttc_strategy
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
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
from nat.retriever.interface import Retriever
from nat.retriever.models import Document
from nat.retriever.models import RetrieverOutput


class FunctionReturningFunctionConfig(FunctionBaseConfig, name="fn_return_fn"):
    pass


class FunctionReturningInfoConfig(FunctionBaseConfig, name="fn_return_info"):
    pass


class FunctionReturningDerivedConfig(FunctionBaseConfig, name="fn_return_derived"):
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

            async def ainvoke(self, items=None, **kwargs):
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


async def test_build():

    async with WorkflowBuilder() as builder:

        # Test building without anything set
        with pytest.raises(ValueError):
            workflow = await builder.build()

        # Add a workflows
        await builder.set_workflow(FunctionReturningFunctionConfig())

        # Test building with a workflow set
        workflow = await builder.build()

        assert isinstance(workflow, Workflow)


async def test_add_function():

    class FunctionReturningBadConfig(FunctionBaseConfig, name="fn_return_bad"):
        pass

    @register_function(config_type=FunctionReturningBadConfig)  # type: ignore
    async def register2(config: FunctionReturningBadConfig, b: Builder):

        yield {}

    async with WorkflowBuilder() as builder:

        fn = await builder.add_function("ret_function", FunctionReturningFunctionConfig())
        assert isinstance(fn, Function)

        fn = await builder.add_function("ret_info", FunctionReturningInfoConfig())
        assert isinstance(fn, Function)

        fn = await builder.add_function("ret_derived", FunctionReturningDerivedConfig())
        assert isinstance(fn, Function)

        with pytest.raises(ValueError):
            await builder.add_function("ret_bad", FunctionReturningBadConfig())

        # Try and add a function with the same name
        with pytest.raises(ValueError):
            await builder.add_function("ret_function", FunctionReturningFunctionConfig())


async def test_get_function():

    async with WorkflowBuilder() as builder:

        fn = await builder.add_function("ret_function", FunctionReturningFunctionConfig())
        assert await builder.get_function("ret_function") == fn

        with pytest.raises(ValueError):
            await builder.get_function("ret_function_not_exist")


async def test_get_function_config():

    async with WorkflowBuilder() as builder:

        config = FunctionReturningFunctionConfig()

        fn = await builder.add_function("ret_function", config)
        assert builder.get_function_config("ret_function") == fn.config
        assert builder.get_function_config("ret_function") is config

        with pytest.raises(ValueError):
            builder.get_function_config("ret_function_not_exist")


async def test_set_workflow():

    class FunctionReturningBadConfig(FunctionBaseConfig, name="fn_return_bad"):
        pass

    @register_function(config_type=FunctionReturningBadConfig)  # type: ignore
    async def register2(config: FunctionReturningBadConfig, b: Builder):

        yield {}

    async with WorkflowBuilder() as builder:

        fn = await builder.set_workflow(FunctionReturningFunctionConfig())
        assert isinstance(fn, Function)

        with pytest.warns(UserWarning, match=r"^Overwriting existing workflow$"):
            fn = await builder.set_workflow(FunctionReturningInfoConfig())

        assert isinstance(fn, Function)

        with pytest.warns(UserWarning, match=r"^Overwriting existing workflow$"):
            fn = await builder.set_workflow(FunctionReturningDerivedConfig())

        assert isinstance(fn, Function)

        with pytest.raises(ValueError):
            with pytest.warns(UserWarning, match=r"^Overwriting existing workflow$"):
                await builder.set_workflow(FunctionReturningBadConfig())

        # Try and add a function with the same name
        with pytest.warns(UserWarning, match=r"^Overwriting existing workflow$"):
            await builder.set_workflow(FunctionReturningFunctionConfig())


async def test_get_workflow():

    async with WorkflowBuilder() as builder:

        with pytest.raises(ValueError):
            builder.get_workflow()

        fn = await builder.set_workflow(FunctionReturningFunctionConfig())
        assert builder.get_workflow() == fn


async def test_get_workflow_config():

    async with WorkflowBuilder() as builder:

        with pytest.raises(ValueError):
            builder.get_workflow_config()

        config = FunctionReturningFunctionConfig()

        fn = await builder.set_workflow(config)
        assert builder.get_workflow_config() == fn.config
        assert builder.get_workflow_config() is config


async def test_get_tool():

    @register_tool_wrapper(wrapper_type="test_framework")
    def tool_wrapper(name: str, fn: Function, builder: Builder):

        class TestFrameworkTool(BaseModel):

            model_config = ConfigDict(arbitrary_types_allowed=True)

            name: str
            fn: Function
            builder: Builder

        return TestFrameworkTool(name=name, fn=fn, builder=builder)

    async with WorkflowBuilder() as builder:

        with pytest.raises(ValueError):
            await builder.get_tool("ret_function", "test_framework")

        fn = await builder.add_function("ret_function", FunctionReturningFunctionConfig())

        tool = await builder.get_tool("ret_function", "test_framework")

        assert tool.name == "ret_function"
        assert tool.fn == fn


async def test_add_llm():

    async with WorkflowBuilder() as builder:

        await builder.add_llm("llm_name", TLLMProviderConfig())

        with pytest.raises(ValueError):
            await builder.add_llm("llm_name2", TLLMProviderConfig(raise_error=True))

        # Try and add a llm with the same name
        with pytest.raises(ValueError):
            await builder.add_llm("llm_name", TLLMProviderConfig())


async def test_get_llm():

    @register_llm_client(config_type=TLLMProviderConfig, wrapper_type="test_framework")
    async def register(config: TLLMProviderConfig, b: Builder):

        class TestFrameworkLLM(BaseModel):

            model_config = ConfigDict(arbitrary_types_allowed=True)

            config: TLLMProviderConfig
            builder: Builder

        yield TestFrameworkLLM(config=config, builder=b)

    async with WorkflowBuilder() as builder:

        config = TLLMProviderConfig()

        await builder.add_llm("llm_name", config)

        llm = await builder.get_llm("llm_name", wrapper_type="test_framework")

        assert llm.config == builder.get_llm_config("llm_name")

        with pytest.raises(ValueError):
            await builder.get_llm("llm_name_not_exist", wrapper_type="test_framework")


async def test_get_llm_config():

    async with WorkflowBuilder() as builder:

        config = TLLMProviderConfig()

        await builder.add_llm("llm_name", config)

        assert builder.get_llm_config("llm_name") == config

        with pytest.raises(ValueError):
            builder.get_llm_config("llm_name_not_exist")


async def test_add_embedder():

    async with WorkflowBuilder() as builder:

        await builder.add_embedder("embedder_name", TEmbedderProviderConfig())

        with pytest.raises(ValueError):
            await builder.add_embedder("embedder_name2", TEmbedderProviderConfig(raise_error=True))

        # Try and add the same name
        with pytest.raises(ValueError):
            await builder.add_embedder("embedder_name", TEmbedderProviderConfig())


async def test_get_embedder():

    @register_embedder_client(config_type=TEmbedderProviderConfig, wrapper_type="test_framework")
    async def register(config: TEmbedderProviderConfig, b: Builder):

        class TestFrameworkEmbedder(BaseModel):

            model_config = ConfigDict(arbitrary_types_allowed=True)

            config: TEmbedderProviderConfig
            builder: Builder

        yield TestFrameworkEmbedder(config=config, builder=b)

    async with WorkflowBuilder() as builder:

        config = TEmbedderProviderConfig()

        await builder.add_embedder("embedder_name", config)

        embedder = await builder.get_embedder("embedder_name", wrapper_type="test_framework")

        assert embedder.config == builder.get_embedder_config("embedder_name")

        with pytest.raises(ValueError):
            await builder.get_embedder("embedder_name_not_exist", wrapper_type="test_framework")


async def test_get_embedder_config():

    async with WorkflowBuilder() as builder:

        config = TEmbedderProviderConfig()

        await builder.add_embedder("embedder_name", config)

        assert builder.get_embedder_config("embedder_name") == config

        with pytest.raises(ValueError):
            builder.get_embedder_config("embedder_name_not_exist")


async def test_add_memory():

    async with WorkflowBuilder() as builder:

        await builder.add_memory_client("memory_name", TMemoryConfig())

        with pytest.raises(ValueError):
            await builder.add_memory_client("memory_name2", TMemoryConfig(raise_error=True))

        # Try and add the same name
        with pytest.raises(ValueError):
            await builder.add_memory_client("memory_name", TMemoryConfig())


async def test_get_memory():

    async with WorkflowBuilder() as builder:

        config = TMemoryConfig()

        memory = await builder.add_memory_client("memory_name", config)

        assert memory == await builder.get_memory_client("memory_name")

        with pytest.raises(ValueError):
            await builder.get_memory_client("memory_name_not_exist")


async def test_get_memory_config():

    async with WorkflowBuilder() as builder:

        config = TMemoryConfig()

        await builder.add_memory_client("memory_name", config)

        assert builder.get_memory_client_config("memory_name") == config

        with pytest.raises(ValueError):
            builder.get_memory_client_config("memory_name_not_exist")


async def test_add_retriever():

    async with WorkflowBuilder() as builder:
        await builder.add_retriever("retriever_name", TRetrieverProviderConfig())

        with pytest.raises(ValueError):
            await builder.add_retriever("retriever_name2", TRetrieverProviderConfig(raise_error=True))

        with pytest.raises(ValueError):
            await builder.add_retriever("retriever_name", TRetrieverProviderConfig())


async def test_add_object_store():

    async with WorkflowBuilder() as builder:
        await builder.add_object_store("object_store_name", TObjectStoreConfig())

        with pytest.raises(ValueError):
            await builder.add_object_store("object_store_name2", TObjectStoreConfig(raise_error=True))

        with pytest.raises(ValueError):
            await builder.add_object_store("object_store_name", TObjectStoreConfig())


async def test_get_object_store():

    async with WorkflowBuilder() as builder:

        object_store = await builder.add_object_store("object_store_name", TObjectStoreConfig())

        assert object_store == await builder.get_object_store_client("object_store_name")

        with pytest.raises(ValueError):
            await builder.get_object_store_client("object_store_name_not_exist")


async def test_get_object_store_config():

    async with WorkflowBuilder() as builder:

        config = TObjectStoreConfig()

        await builder.add_object_store("object_store_name", config)

        assert builder.get_object_store_config("object_store_name") == config

        with pytest.raises(ValueError):
            builder.get_object_store_config("object_store_name_not_exist")


async def test_get_retriever():

    @register_retriever_client(config_type=TRetrieverProviderConfig, wrapper_type="test_framework")
    async def register(config: TRetrieverProviderConfig, b: Builder):

        class TestFrameworkRetriever(BaseModel):

            model_config = ConfigDict(arbitrary_types_allowed=True)

            config: TRetrieverProviderConfig
            builder: Builder

        yield TestFrameworkRetriever(config=config, builder=b)

    @register_retriever_client(config_type=TRetrieverProviderConfig, wrapper_type=None)
    async def register_no_framework(config: TRetrieverProviderConfig, _builder: Builder):

        class TestRetriever(Retriever):

            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            async def search(self, query: str, **kwargs):
                return RetrieverOutput(results=[Document(page_content="page content", metadata={})])

        yield TestRetriever(**config.model_dump())

    async with WorkflowBuilder() as builder:

        config = TRetrieverProviderConfig()

        await builder.add_retriever("retriever_name", config)

        retriever = await builder.get_retriever("retriever_name", wrapper_type="test_framework")

        assert retriever.config == await builder.get_retriever_config("retriever_name")

        with pytest.raises(ValueError):
            await builder.get_retriever("retriever_name_not_exist", wrapper_type="test_framework")

        retriever = await builder.get_retriever("retriever_name", wrapper_type=None)

        assert isinstance(retriever, Retriever)


async def test_get_retriever_config():

    async with WorkflowBuilder() as builder:

        config = TRetrieverProviderConfig()

        await builder.add_retriever("retriever_name", config)

        assert await builder.get_retriever_config("retriever_name") == config

        with pytest.raises(ValueError):
            await builder.get_retriever_config("retriever_name_not_exist")


async def test_add_ttc_strategy():

    async with WorkflowBuilder() as builder:
        # Normal case
        await builder.add_ttc_strategy("ttc_strategy", TTTCStrategyConfig())

        # Provider raises
        with pytest.raises(ValueError):
            await builder.add_ttc_strategy("ttc_strategy_err", TTTCStrategyConfig(raise_error=True))

        # Duplicate name
        with pytest.raises(ValueError):
            await builder.add_ttc_strategy("ttc_strategy", TTTCStrategyConfig())


async def test_get_ttc_strategy_and_config():

    async with WorkflowBuilder() as builder:
        cfg = TTTCStrategyConfig()
        await builder.add_ttc_strategy("ttc_strategy", cfg)

        strat = await builder.get_ttc_strategy(
            "ttc_strategy",
            pipeline_type=PipelineTypeEnum.AGENT_EXECUTION,
            stage_type=StageTypeEnum.SCORING,
        )

        with pytest.raises(ValueError):
            await builder.get_ttc_strategy(
                "ttc_strategy",
                pipeline_type=PipelineTypeEnum.PLANNING,  # Wrong pipeline type
                stage_type=StageTypeEnum.SCORING,
            )

        assert strat.config == await builder.get_ttc_strategy_config(
            "ttc_strategy",
            pipeline_type=PipelineTypeEnum.AGENT_EXECUTION,
            stage_type=StageTypeEnum.SCORING,
        )

        # Non-existent name
        with pytest.raises(ValueError):
            await builder.get_ttc_strategy(
                "does_not_exist",
                pipeline_type=PipelineTypeEnum.AGENT_EXECUTION,
                stage_type=StageTypeEnum.SCORING,
            )


async def test_built_config():

    general_config = GeneralConfig()
    function_config = FunctionReturningFunctionConfig()
    workflow_config = FunctionReturningFunctionConfig()
    llm_config = TLLMProviderConfig()
    embedder_config = TEmbedderProviderConfig()
    memory_config = TMemoryConfig()
    retriever_config = TRetrieverProviderConfig()
    object_store_config = TObjectStoreConfig()
    ttc_config = TTTCStrategyConfig()

    async with WorkflowBuilder(general_config=general_config) as builder:

        await builder.add_function("function1", function_config)

        await builder.set_workflow(workflow_config)

        await builder.add_llm("llm1", llm_config)

        await builder.add_embedder("embedder1", embedder_config)

        await builder.add_memory_client("memory1", memory_config)

        await builder.add_retriever("retriever1", retriever_config)

        await builder.add_object_store("object_store1", object_store_config)

        await builder.add_ttc_strategy("ttc_strategy", ttc_config)

        workflow = await builder.build()

        workflow_config = workflow.config

        assert workflow_config.general == general_config
        assert workflow_config.functions == {"function1": function_config}
        assert workflow_config.workflow == workflow_config.workflow
        assert workflow_config.llms == {"llm1": llm_config}
        assert workflow_config.embedders == {"embedder1": embedder_config}
        assert workflow_config.memory == {"memory1": memory_config}
        assert workflow_config.retrievers == {"retriever1": retriever_config}
        assert workflow_config.object_stores == {"object_store1": object_store_config}
        assert workflow_config.ttc_strategies == {"ttc_strategy": ttc_config}


# Function Group Tests


async def test_add_function_group():
    """Test adding function groups to a workflow builder."""

    async with WorkflowBuilder() as builder:
        includes_group = await builder.add_function_group("includes_group", IncludesFunctionGroupConfig())
        assert isinstance(includes_group, FunctionGroup)

        excludes_group = await builder.add_function_group("excludes_group", ExcludesFunctionGroupConfig())
        assert isinstance(excludes_group, FunctionGroup)

        # Test adding a function group with no included functions
        empty_group = await builder.add_function_group("empty_group", DefaultFunctionGroup())
        assert isinstance(empty_group, FunctionGroup)

        # Test adding a function group that includes all functions
        all_includes_group = await builder.add_function_group("all_includes_group", AllIncludesFunctionGroupConfig())
        assert isinstance(all_includes_group, FunctionGroup)

        all_excludes_group = await builder.add_function_group("all_excludes_group", AllExcludesFunctionGroupConfig())
        assert isinstance(all_excludes_group, FunctionGroup)

        # Test error when adding function group with existing name
        with pytest.raises(ValueError):
            await builder.add_function_group("includes_group", IncludesFunctionGroupConfig())

        # Test error when adding function group that fails during initialization
        with pytest.raises(ValueError):
            await builder.add_function_group("failing_group", FailingFunctionGroupConfig())


async def test_get_function_group():
    """Test getting function groups from a workflow builder."""

    async with WorkflowBuilder() as builder:
        # Add a function group
        added_group = await builder.add_function_group("math_group", IncludesFunctionGroupConfig())

        # Test getting existing function group
        retrieved_group = await builder.get_function_group("math_group")
        assert retrieved_group == added_group

        # Test error when getting non-existent function group
        with pytest.raises(ValueError):
            await builder.get_function_group("non_existent_group")


async def test_get_function_group_config():
    """Test getting function group configurations."""

    async with WorkflowBuilder() as builder:
        # Add a function group
        config = IncludesFunctionGroupConfig()
        await builder.add_function_group("includes_group", config)

        # Test getting existing function group config
        retrieved_config = builder.get_function_group_config("includes_group")
        assert retrieved_config == config
        assert retrieved_config is config

        # Test error when getting non-existent function group config
        with pytest.raises(ValueError):
            builder.get_function_group_config("non_existent_group")


async def test_function_group_included_functions():
    """Test that included functions from function groups are accessible."""

    async with WorkflowBuilder() as builder:
        # Add function group with some included functions
        await builder.add_function_group("includes_group", IncludesFunctionGroupConfig())

        # Test that included functions are accessible as regular functions
        add_fn = await builder.get_function("includes_group.add")
        multiply_fn = await builder.get_function("includes_group.multiply")

        assert add_fn is not None
        assert multiply_fn is not None

        # Test that non-included functions are not accessible
        with pytest.raises(ValueError):
            await builder.get_function("includes_group.subtract")


async def test_function_group_excluded_functions():
    """Test that excluded functions from function groups are not accessible."""

    async with WorkflowBuilder() as builder:
        # Add function group with some excluded functions
        await builder.add_function_group("excludes_group", ExcludesFunctionGroupConfig())

        # Test that NO functions are accessible globally since the group uses exclude (not include)
        # The function group doesn't expose any functions to the global registry when using exclude only
        with pytest.raises(ValueError):
            await builder.get_function("excludes_group.add")
        with pytest.raises(ValueError):
            await builder.get_function("excludes_group.multiply")
        with pytest.raises(ValueError):
            await builder.get_function("excludes_group.subtract")

        # But the functions should be accessible through the function group itself
        group = await builder.get_function_group("excludes_group")
        accessible_functions = await group.get_accessible_functions()

        # Should have only subtract (add and multiply are excluded)
        assert len(accessible_functions) == 1
        assert "excludes_group.subtract" in accessible_functions


async def test_function_group_empty_includes_and_excludes():
    """Test function group with no included functions."""

    async with WorkflowBuilder() as builder:
        # Add function group with no included functions
        await builder.add_function_group("empty_group", DefaultFunctionGroup())

        # Verify no functions were added to global registry
        included_functions = [k for k in builder._functions.keys() if k.startswith("empty_group.")]
        assert len(included_functions) == 0

        # But the group itself should exist
        group = await builder.get_function_group("empty_group")
        assert isinstance(group, FunctionGroup)

        assert len(await group.get_accessible_functions()) == 0  # No functions accessible (empty include list)
        assert len(await group.get_all_functions()) == 1  # One function in the group (internal_function)
        assert len(await group.get_included_functions()) == 0  # No functions in include list


async def test_function_group_all_includes():
    """Test function group that includes all functions."""

    async with WorkflowBuilder() as builder:
        # Add function group that includes all functions
        await builder.add_function_group("all_includes_group", AllIncludesFunctionGroupConfig())

        # All functions should be accessible
        add_fn = await builder.get_function("all_includes_group.add")
        multiply_fn = await builder.get_function("all_includes_group.multiply")
        subtract_fn = await builder.get_function("all_includes_group.subtract")

        assert add_fn is not None
        assert multiply_fn is not None
        assert subtract_fn is not None

        group = await builder.get_function_group("all_includes_group")

        assert len(await group.get_accessible_functions()) == 3
        assert len(await group.get_all_functions()) == 3
        assert len(await group.get_included_functions()) == 3


async def test_function_group_all_excludes():
    """Test function group that excludes all functions."""

    async with WorkflowBuilder() as builder:
        # Add function group that excludes all functions
        await builder.add_function_group("all_excludes_group", AllExcludesFunctionGroupConfig())

        # No functions should be accessible globally (function group uses exclude only)
        with pytest.raises(ValueError):
            await builder.get_function("all_excludes_group.add")
        with pytest.raises(ValueError):
            await builder.get_function("all_excludes_group.multiply")
        with pytest.raises(ValueError):
            await builder.get_function("all_excludes_group.subtract")

        group = await builder.get_function_group("all_excludes_group")

        assert len(await group.get_accessible_functions()) == 0
        assert len(await group.get_all_functions()) == 3
        assert len(await group.get_included_functions()) == 0


async def test_function_group_name_conflicts():
    """Test function group name conflict handling."""

    async with WorkflowBuilder() as builder:
        # Add a function first
        await builder.add_function("math_group", FunctionReturningFunctionConfig())

        # Try to add function group with same name - should fail
        with pytest.raises(ValueError):
            await builder.add_function_group("math_group", IncludesFunctionGroupConfig())


async def test_function_group_dependencies_tracking():
    """Test that function group dependencies are properly tracked."""

    async with WorkflowBuilder() as builder:
        await builder.add_function_group("math_group", IncludesFunctionGroupConfig())

        # Check that dependencies are tracked
        assert "math_group" in builder.function_group_dependencies
        from nat.data_models.function_dependencies import FunctionDependencies
        dependencies = builder.function_group_dependencies["math_group"]
        assert isinstance(dependencies, FunctionDependencies)


async def test_function_group_integration_with_workflow():
    """Test building a workflow that includes function groups."""

    async with WorkflowBuilder() as builder:
        # Add function groups
        await builder.add_function_group("math_group", IncludesFunctionGroupConfig())
        await builder.add_function_group("empty_group", DefaultFunctionGroup())

        # Add regular functions
        await builder.add_function("regular_fn", FunctionReturningFunctionConfig())

        # Set workflow
        await builder.set_workflow(FunctionReturningFunctionConfig())

        # Test that function groups were added correctly
        assert "math_group" in builder._function_groups
        assert "empty_group" in builder._function_groups

        # Test that included functions are accessible
        assert "math_group.add" in builder._functions
        assert "math_group.multiply" in builder._functions

        # Test that non-included functions are not accessible
        assert "math_group.subtract" not in builder._functions

        # Test that no functions were included from empty group
        empty_group_functions = [k for k in builder._functions.keys() if k.startswith("empty_group.")]
        assert len(empty_group_functions) == 0

        # Test that regular functions still work
        assert "regular_fn" in builder._functions


async def test_function_group_config_validation():
    """Test function group configuration validation."""

    # Test that function group configs are stored correctly in the builder
    async with WorkflowBuilder() as builder:
        config = IncludesFunctionGroupConfig()
        await builder.add_function_group("math_group", config)

        # Test getting function group config
        retrieved_config = builder.get_function_group_config("math_group")
        assert retrieved_config == config
        assert retrieved_config is config

        # Test that function group is stored correctly
        function_group = await builder.get_function_group("math_group")
        assert isinstance(function_group, FunctionGroup)


async def test_function_group_add_function_validation():
    """Test function group add_function validation errors."""

    config = IncludesFunctionGroupConfig()
    group = FunctionGroup(config=config)

    # Test empty function name
    with pytest.raises(ValueError, match="Function name cannot be empty"):

        async def dummy_func(x: int) -> int:
            return x

        group.add_function("", dummy_func)

    # Test function name with whitespace
    with pytest.raises(ValueError,
                       match="Function name can only contain letters, numbers, underscores, periods, and hyphens"):

        async def dummy_func2(x: int) -> int:
            return x

        group.add_function("invalid name", dummy_func2)

    # Test duplicate function names
    async def test_func(x: int) -> int:
        return x

    group.add_function("test_func", test_func)
    with pytest.raises(ValueError):
        group.add_function("test_func", test_func)  # Should fail - duplicate name


async def test_function_group_get_excluded_functions():
    """Test getting excluded functions from function groups."""

    async with WorkflowBuilder() as builder:
        # Test group with exclude configuration
        await builder.add_function_group("excludes_group", ExcludesFunctionGroupConfig())
        group = await builder.get_function_group("excludes_group")

        excluded_functions = await group.get_excluded_functions()
        assert len(excluded_functions) == 2  # add and multiply are excluded
        assert "excludes_group.add" in excluded_functions
        assert "excludes_group.multiply" in excluded_functions
        assert "excludes_group.subtract" not in excluded_functions

        # Test group with no exclude configuration
        await builder.add_function_group("includes_group", IncludesFunctionGroupConfig())
        includes_group = await builder.get_function_group("includes_group")

        excluded_from_includes = await includes_group.get_excluded_functions()
        assert len(excluded_from_includes) == 0  # No exclude list defined


async def test_function_group_invalid_include_configuration():
    """Test function group with invalid include configuration."""

    class InvalidIncludeConfig(FunctionGroupBaseConfig, name="invalid_include_group"):
        include: list[str] = Field(default_factory=lambda: ["non_existent_function"])
        raise_error: bool = False

    @register_function_group(config_type=InvalidIncludeConfig)
    async def register_invalid_group(config: InvalidIncludeConfig, _builder: Builder):
        group = FunctionGroup(config=config)

        async def real_function(x: int) -> int:
            return x

        group.add_function("real_function", real_function, description="A real function")
        yield group

    async with WorkflowBuilder() as builder:
        # Should raise error during add_function_group when validation happens
        with pytest.raises(ValueError, match=r"Unknown included functions"):
            await builder.add_function_group("invalid_group", InvalidIncludeConfig())


async def test_function_group_invalid_exclude_configuration():
    """Test function group with invalid exclude configuration."""

    class InvalidExcludeConfig(FunctionGroupBaseConfig, name="invalid_exclude_group"):
        exclude: list[str] = Field(default_factory=lambda: ["non_existent_function"])
        raise_error: bool = False

    @register_function_group(config_type=InvalidExcludeConfig)
    async def register_invalid_exclude_group(config: InvalidExcludeConfig, _builder: Builder):
        group = FunctionGroup(config=config)

        async def real_function(x: int) -> int:
            return x

        group.add_function("real_function", real_function, description="A real function")
        yield group

    async with WorkflowBuilder() as builder:
        await builder.add_function_group("invalid_exclude_group", InvalidExcludeConfig())
        group = await builder.get_function_group("invalid_exclude_group")

        # Should raise error when trying to get excluded functions
        with pytest.raises(ValueError, match=r"Unknown excluded functions"):
            await group.get_excluded_functions()

        # Should also raise error when trying to get accessible functions
        with pytest.raises(ValueError, match=r"Unknown excluded functions"):
            await group.get_accessible_functions()


async def test_function_group_get_config():
    """Test getting function group configuration."""

    config = IncludesFunctionGroupConfig()
    group = FunctionGroup(config=config)

    retrieved_config = group.get_config()
    assert retrieved_config == config
    assert retrieved_config is config


async def test_function_group_function_execution():
    """Test executing functions within function groups."""

    async with WorkflowBuilder() as builder:
        await builder.add_function_group("math_group", IncludesFunctionGroupConfig())

        # Get and execute functions from the group
        add_fn = await builder.get_function("math_group.add")
        result = await add_fn.ainvoke({"a": 5, "b": 3})
        assert result == 8

        multiply_fn = await builder.get_function("math_group.multiply")
        result = await multiply_fn.ainvoke({"a": 4, "b": 6})
        assert result == 24


async def test_function_group_custom_instance_name():
    """Test function group with custom instance name."""

    # Create a config that includes the "add" function
    class CustomInstanceConfig(FunctionGroupBaseConfig, name="custom_instance_group"):
        include: list[str] = Field(default_factory=lambda: ["add"])
        raise_error: bool = False

    config = CustomInstanceConfig()
    group = FunctionGroup(config=config, instance_name="custom_math_group")

    async def add_func(a: int, b: int) -> int:
        return a + b

    group.add_function("add", add_func, description="Add two numbers")

    # Function should be returned with instance name prefix
    all_functions = await group.get_all_functions()
    assert "custom_math_group.add" in all_functions

    # When getting included functions, should use custom instance name prefix
    included = await group.get_included_functions()
    assert "custom_math_group.add" in included


async def test_add_telemetry_exporter():

    workflow_config = FunctionReturningFunctionConfig()
    telemetry_exporter_config = TTelemetryExporterConfig()

    async with WorkflowBuilder() as builder:

        await builder.set_workflow(workflow_config)

        await builder.add_telemetry_exporter("exporter1", telemetry_exporter_config)

        with pytest.raises(ValueError):
            await builder.add_telemetry_exporter("exporter2", TTelemetryExporterConfig(raise_error=True))

        with pytest.raises(ValueError):
            await builder.add_telemetry_exporter("exporter1", TTelemetryExporterConfig())

        workflow = await builder.build()

        exporter1_instance = workflow.telemetry_exporters.get("exporter1", None)

        assert exporter1_instance is not None
        assert issubclass(type(exporter1_instance), BaseExporter)


# Error Logging Tests


@pytest.fixture
def caplog_fixture(caplog):
    """Configure caplog to capture ERROR level logs."""
    caplog.set_level(logging.ERROR)
    return caplog


@pytest.fixture
def mock_component_data():
    """Create mock component data for testing."""
    # Create a mock failing component
    failing_component = MagicMock()
    failing_component.name = "test_component"
    failing_component.component_group.value = "llms"

    return failing_component


def test_log_build_failure_helper_method(caplog_fixture, mock_component_data):
    """Test the _log_build_failure helper method directly."""
    builder = WorkflowBuilder()

    completed_components = [("comp1", "llms"), ("comp2", "embedders")]
    remaining_components = [("comp3", "functions"), ("comp4", "memory")]
    original_error = ValueError("Test error message")

    # Call the helper method
    builder._log_build_failure_component(mock_component_data,
                                         completed_components,
                                         remaining_components,
                                         original_error)

    # Verify error logging content
    log_text = caplog_fixture.text
    assert "Failed to initialize component test_component (llms)" in log_text
    assert "Successfully built components:" in log_text
    assert "- comp1 (llms)" in log_text
    assert "- comp2 (embedders)" in log_text
    assert "Remaining components to build:" in log_text
    assert "- comp3 (functions)" in log_text
    assert "- comp4 (memory)" in log_text
    assert "Original error:" in log_text
    assert "Test error message" in log_text


def test_log_build_failure_workflow_helper_method(caplog_fixture):
    """Test the _log_build_failure_workflow helper method directly."""
    builder = WorkflowBuilder()

    completed_components = [("comp1", "llms"), ("comp2", "embedders")]
    remaining_components = [("comp3", "functions")]
    original_error = ValueError("Workflow build failed")

    # Call the helper method
    builder._log_build_failure_workflow(completed_components, remaining_components, original_error)

    # Verify error logging content
    log_text = caplog_fixture.text
    assert "Failed to initialize component <workflow> (workflow)" in log_text
    assert "Successfully built components:" in log_text
    assert "- comp1 (llms)" in log_text
    assert "- comp2 (embedders)" in log_text
    assert "Remaining components to build:" in log_text
    assert "- comp3 (functions)" in log_text
    assert "Original error:" in log_text


def test_log_build_failure_no_completed_components(caplog_fixture, mock_component_data):
    """Test error logging when no components have been successfully built."""
    builder = WorkflowBuilder()

    completed_components = []
    remaining_components = [("comp1", "embedders"), ("comp2", "functions")]
    original_error = ValueError("First component failed")

    builder._log_build_failure_component(mock_component_data,
                                         completed_components,
                                         remaining_components,
                                         original_error)

    log_text = caplog_fixture.text
    assert "Failed to initialize component test_component (llms)" in log_text
    assert "No components were successfully built before this failure" in log_text
    assert "Remaining components to build:" in log_text
    assert "- comp1 (embedders)" in log_text
    assert "- comp2 (functions)" in log_text
    assert "Original error:" in log_text


def test_log_build_failure_no_remaining_components(caplog_fixture, mock_component_data):
    """Test error logging when no components remain to be built."""
    builder = WorkflowBuilder()

    completed_components = [("comp1", "llms"), ("comp2", "embedders")]
    remaining_components = []
    original_error = ValueError("Last component failed")

    builder._log_build_failure_component(mock_component_data,
                                         completed_components,
                                         remaining_components,
                                         original_error)

    log_text = caplog_fixture.text
    assert "Failed to initialize component test_component (llms)" in log_text
    assert "Successfully built components:" in log_text
    assert "- comp1 (llms)" in log_text
    assert "- comp2 (embedders)" in log_text
    assert "No remaining components to build" in log_text
    assert "Original error:" in log_text


# Evaluator Error Logging Tests


def test_log_evaluator_build_failure_helper_method(caplog_fixture):
    """Test the _log_evaluator_build_failure helper method directly."""
    from nat.builder.eval_builder import WorkflowEvalBuilder

    builder = WorkflowEvalBuilder()

    completed_evaluators = ["eval1", "eval2"]
    remaining_evaluators = ["eval3", "eval4"]
    original_error = ValueError("Evaluator build failed")

    # Call the helper method
    builder._log_build_failure_evaluator("failing_evaluator",
                                         completed_evaluators,
                                         remaining_evaluators,
                                         original_error)

    # Verify error logging content
    log_text = caplog_fixture.text
    assert "Failed to initialize component failing_evaluator (evaluator)" in log_text
    assert "Successfully built components:" in log_text
    assert "- eval1 (evaluator)" in log_text
    assert "- eval2 (evaluator)" in log_text
    assert "Remaining components to build:" in log_text
    assert "- eval3 (evaluator)" in log_text
    assert "- eval4 (evaluator)" in log_text
    assert "Original error:" in log_text


def test_log_evaluator_build_failure_no_completed(caplog_fixture):
    """Test evaluator error logging when no evaluators have been successfully built."""
    from nat.builder.eval_builder import WorkflowEvalBuilder

    builder = WorkflowEvalBuilder()

    completed_evaluators = []
    remaining_evaluators = ["eval1", "eval2"]
    original_error = ValueError("First evaluator failed")

    builder._log_build_failure_evaluator("failing_evaluator",
                                         completed_evaluators,
                                         remaining_evaluators,
                                         original_error)

    log_text = caplog_fixture.text
    assert "Failed to initialize component failing_evaluator (evaluator)" in log_text
    assert "No components were successfully built before this failure" in log_text
    assert "Remaining components to build:" in log_text
    assert "- eval1 (evaluator)" in log_text
    assert "- eval2 (evaluator)" in log_text
    assert "Original error:" in log_text


def test_log_evaluator_build_failure_no_remaining(caplog_fixture):
    """Test evaluator error logging when no evaluators remain to be built."""
    from nat.builder.eval_builder import WorkflowEvalBuilder

    builder = WorkflowEvalBuilder()

    completed_evaluators = ["eval1", "eval2"]
    remaining_evaluators = []
    original_error = ValueError("Last evaluator failed")

    builder._log_build_failure_evaluator("failing_evaluator",
                                         completed_evaluators,
                                         remaining_evaluators,
                                         original_error)

    log_text = caplog_fixture.text
    assert "Failed to initialize component failing_evaluator (evaluator)" in log_text
    assert "Successfully built components:" in log_text
    assert "- eval1 (evaluator)" in log_text
    assert "- eval2 (evaluator)" in log_text
    assert "No remaining components to build" in log_text
    assert "Original error:" in log_text


async def test_integration_error_logging_with_failing_function(caplog_fixture):
    """Integration test: Verify error logging when building a workflow with a function that fails during initialization.

    This test creates a real failing function (not mocked) and attempts to build a workflow,
    then verifies that the error logging messages are correct.
    """
    # Create a config with one successful function and one failing function
    config_dict = {
        "functions": {
            "working_function": FunctionReturningFunctionConfig(),
            "failing_function": FailingFunctionConfig(),
            "another_working_function": FunctionReturningInfoConfig()
        },
        "workflow": FunctionReturningFunctionConfig()
    }

    config = Config.model_validate(config_dict)

    async with WorkflowBuilder() as builder:
        with pytest.raises(ValueError, match="Function initialization failed"):
            await builder.populate_builder(config)

    # Verify the error logging output
    log_text = caplog_fixture.text

    # Should have the main error message with component name and type
    assert "Failed to initialize component failing_function (functions)" in log_text

    # Should list successfully built components before the failure
    assert "Successfully built components:" in log_text
    assert "- working_function (functions)" in log_text

    # Should list remaining components that still need to be built
    assert "Remaining components to build:" in log_text
    assert "- another_working_function (functions)" in log_text
    assert "- <workflow> (workflow)" in log_text

    # Should include the original error
    assert "Original error:" in log_text
    assert "Function initialization failed" in log_text

    # Verify the error was propagated (not just logged)
    assert "ValueError: Function initialization failed" in log_text


async def test_integration_error_logging_with_workflow_failure(caplog_fixture):
    """Integration test: Verify error logging when workflow setup fails.

    This test attempts to build with a failing workflow and verifies the error messages.
    """
    # Create a config with successful functions but failing workflow
    config_dict = {
        "functions": {
            "working_function1": FunctionReturningFunctionConfig(), "working_function2": FunctionReturningInfoConfig()
        },
        "workflow":
            FailingFunctionConfig()  # This will fail during workflow setup
    }

    config = Config.model_validate(config_dict)

    async with WorkflowBuilder() as builder:
        with pytest.raises(ValueError, match="Function initialization failed"):
            await builder.populate_builder(config)

    # Verify the error logging output
    log_text = caplog_fixture.text

    # Should have the main error message for workflow failure
    assert "Failed to initialize component <workflow> (workflow)" in log_text

    # Should list all successfully built components (functions should have succeeded)
    assert "Successfully built components:" in log_text
    assert "- working_function1 (functions)" in log_text
    assert "- working_function2 (functions)" in log_text

    # Should show no remaining components to build (since workflow is the last step)
    assert "No remaining components to build" in log_text

    # Should include the original error
    assert "Original error:" in log_text
    assert "Function initialization failed" in log_text
