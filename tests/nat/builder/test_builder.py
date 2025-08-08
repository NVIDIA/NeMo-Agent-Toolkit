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
from openai import BaseModel
from pydantic import ConfigDict

from nat.builder.builder import Builder
from nat.builder.embedder import EmbedderProviderInfo
from nat.builder.function import Function
from nat.builder.function_info import FunctionInfo
from nat.builder.llm import LLMProviderInfo
from nat.builder.retriever import RetrieverProviderInfo
from nat.builder.workflow import Workflow
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_embedder_client
from nat.cli.register_workflow import register_embedder_provider
from nat.cli.register_workflow import register_function
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


class TestTTCStrategyConfig(TTCStrategyBaseConfig, name="test_ttc_strategy"):
    raise_error: bool = False


class FailingFunctionConfig(FunctionBaseConfig, name="failing_function"):
    pass


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

        class DerivedFunction(Function[str, str, None]):

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
    async def register7(config: TRetrieverProviderConfig, builder: Builder):

        if (config.raise_error):
            raise ValueError("Error")

        yield RetrieverProviderInfo(config=config, description="Mock retriever to test the registration process")

    @register_object_store(config_type=TObjectStoreConfig)
    async def register8(config: TObjectStoreConfig, builder: Builder):
        if (config.raise_error):
            raise ValueError("Error")

        yield InMemoryObjectStore()

    # Register mock telemetry exporter
    @register_telemetry_exporter(config_type=TTelemetryExporterConfig)
    async def register9(config: TTelemetryExporterConfig, builder: Builder):

        if (config.raise_error):
            raise ValueError("Error")

        class TestTelemetryExporter(BaseExporter):

            def export(self, event: IntermediateStep):
                pass

        yield TestTelemetryExporter()

    @register_ttc_strategy(config_type=TestTTCStrategyConfig)
    async def register_ttc(config: TestTTCStrategyConfig, builder: Builder):

        if config.raise_error:
            raise ValueError("Error")

        class DummyTTCStrategy(StrategyBase):
            """Very small pass-through strategy used only for testing."""

            async def ainvoke(self, items=None, **kwargs):
                # Do nothing, just return what we got
                return items

            async def build_components(self, builder: Builder) -> None:
                pass

            def supported_pipeline_types(self) -> [PipelineTypeEnum]:
                return [PipelineTypeEnum.AGENT_EXECUTION]

            def stage_type(self) -> StageTypeEnum:
                return StageTypeEnum.SCORING

        yield DummyTTCStrategy(config)


async def test_build():

    async with WorkflowBuilder() as builder:

        # Test building without anything set
        with pytest.raises(ValueError):
            workflow = builder.build()

        # Add a workflows
        await builder.set_workflow(FunctionReturningFunctionConfig())

        # Test building with a workflow set
        workflow = builder.build()

        assert isinstance(workflow, Workflow)


async def test_add_function():

    class FunctionReturningBadConfig(FunctionBaseConfig, name="fn_return_bad"):
        pass

    @register_function(config_type=FunctionReturningBadConfig)
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
        assert builder.get_function("ret_function") == fn

        with pytest.raises(ValueError):
            builder.get_function("ret_function_not_exist")


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

    @register_function(config_type=FunctionReturningBadConfig)
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
            builder.get_tool("ret_function", "test_framework")

        fn = await builder.add_function("ret_function", FunctionReturningFunctionConfig())

        tool = builder.get_tool("ret_function", "test_framework")

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

        assert memory == builder.get_memory_client("memory_name")

        with pytest.raises(ValueError):
            builder.get_memory_client("memory_name_not_exist")


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


async def get_retriever():

    @register_retriever_client(config_type=TRetrieverProviderConfig, wrapper_type="test_framework")
    async def register(config: TRetrieverProviderConfig, b: Builder):

        class TestFrameworkRetriever(BaseModel):

            model_config = ConfigDict(arbitrary_types_allowed=True)

            config: TRetrieverProviderConfig
            builder: Builder

        yield TestFrameworkRetriever(config=config, builder=b)

    @register_retriever_client(config_type=TRetrieverProviderConfig, wrapper_type=None)
    async def register_no_framework(config: TRetrieverProviderConfig, builder: Builder):

        class TestRetriever(Retriever):

            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            async def search(self, query: str, **kwargs):
                return RetrieverOutput(results=[Document(page_content="page content", metadata={})])

            async def add_items(self, items):
                return await super().add_items(items)

            async def remove_items(self, **kwargs):
                return await super().remove_items(**kwargs)

        yield TestRetriever(**config.model_dump())

    async with WorkflowBuilder() as builder:

        config = TRetrieverProviderConfig()

        await builder.add_retriever("retriever_name", config)

        retriever = await builder.get_retriever("retriever_name", wrapper_type="test_framework")

        assert retriever.config == builder.get_retriever_config("retriever_name")

        with pytest.raises(ValueError):
            await builder.get_retriever("retriever_name_not_exist", wrapper_type="test_framework")

        retriever = await builder.get_retriever("retriever_name", wrapper_type=None)

        assert isinstance(retriever, Retriever)


async def get_retriever_config():

    async with WorkflowBuilder() as builder:

        config = TRetrieverProviderConfig()

        await builder.add_retriever("retriever_name", config)

        assert builder.get_retriever_config("retriever_name") == config

        with pytest.raises(ValueError):
            builder.get_retriever_config("retriever_name_not_exist")


async def test_add_ttc_strategy():

    async with WorkflowBuilder() as builder:
        # Normal case
        await builder.add_ttc_strategy("ttc_strategy", TestTTCStrategyConfig())

        # Provider raises
        with pytest.raises(ValueError):
            await builder.add_ttc_strategy("ttc_strategy_err", TestTTCStrategyConfig(raise_error=True))

        # Duplicate name
        with pytest.raises(ValueError):
            await builder.add_ttc_strategy("ttc_strategy", TestTTCStrategyConfig())


async def test_get_ttc_strategy_and_config():

    async with WorkflowBuilder() as builder:
        cfg = TestTTCStrategyConfig()
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

    general_config = GeneralConfig(cache_dir="Something else")
    function_config = FunctionReturningFunctionConfig()
    workflow_config = FunctionReturningFunctionConfig()
    llm_config = TLLMProviderConfig()
    embedder_config = TEmbedderProviderConfig()
    memory_config = TMemoryConfig()
    retriever_config = TRetrieverProviderConfig()
    object_store_config = TObjectStoreConfig()
    ttc_config = TestTTCStrategyConfig()

    async with WorkflowBuilder(general_config=general_config) as builder:

        await builder.add_function("function1", function_config)

        await builder.set_workflow(workflow_config)

        await builder.add_llm("llm1", llm_config)

        await builder.add_embedder("embedder1", embedder_config)

        await builder.add_memory_client("memory1", memory_config)

        await builder.add_retriever("retriever1", retriever_config)

        await builder.add_object_store("object_store1", object_store_config)

        await builder.add_ttc_strategy("ttc_strategy", ttc_config)

        workflow = builder.build()

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

        workflow = builder.build()

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
