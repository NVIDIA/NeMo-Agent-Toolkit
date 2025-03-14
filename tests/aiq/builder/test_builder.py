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

import pytest
from openai import BaseModel
from pydantic import ConfigDict

from aiq.builder.builder import Builder
from aiq.builder.embedder import EmbedderProviderInfo
from aiq.builder.function import Function
from aiq.builder.function_info import FunctionInfo
from aiq.builder.llm import LLMProviderInfo
from aiq.builder.retriever import RetrieverProviderInfo
from aiq.builder.workflow import Workflow
from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.cli.register_workflow import register_embedder_client
from aiq.cli.register_workflow import register_embedder_provider
from aiq.cli.register_workflow import register_function
from aiq.cli.register_workflow import register_llm_client
from aiq.cli.register_workflow import register_llm_provider
from aiq.cli.register_workflow import register_memory
from aiq.cli.register_workflow import register_retriever_client
from aiq.cli.register_workflow import register_retriever_provider
from aiq.cli.register_workflow import register_tool_wrapper
from aiq.data_models.config import GeneralConfig
from aiq.data_models.embedder import EmbedderBaseConfig
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.llm import LLMBaseConfig
from aiq.data_models.memory import MemoryBaseConfig
from aiq.data_models.retriever import RetrieverBaseConfig
from aiq.memory.interfaces import MemoryEditor
from aiq.memory.models import MemoryItem
from aiq.retriever.interface import AIQRetriever
from aiq.retriever.models import AIQDocument
from aiq.retriever.models import RetrieverOutput


class FunctionReturningFunctionConfig(FunctionBaseConfig, name="fn_return_fn"):
    pass


class FunctionReturningInfoConfig(FunctionBaseConfig, name="fn_return_info"):
    pass


class FunctionReturningDerivedConfig(FunctionBaseConfig, name="fn_return_derived"):
    pass


class TestLLMProviderConfig(LLMBaseConfig, name="test_llm"):
    raise_error: bool = False


class TestEmbedderProviderConfig(EmbedderBaseConfig, name="test_embedder_provider"):
    raise_error: bool = False


class TestMemoryConfig(MemoryBaseConfig, name="test_memory"):
    raise_error: bool = False


class TestRetrieverProviderConfig(RetrieverBaseConfig, name="test_retriever"):
    raise_error: bool = False


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

    @register_llm_provider(config_type=TestLLMProviderConfig)
    async def register4(config: TestLLMProviderConfig, b: Builder):

        if (config.raise_error):
            raise ValueError("Error")

        yield LLMProviderInfo(config=config, description="A test client.")

    @register_embedder_provider(config_type=TestEmbedderProviderConfig)
    async def registe5(config: TestEmbedderProviderConfig, b: Builder):

        if (config.raise_error):
            raise ValueError("Error")

        yield EmbedderProviderInfo(config=config, description="A test client.")

    @register_memory(config_type=TestMemoryConfig)
    async def register6(config: TestMemoryConfig, b: Builder):

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
    @register_retriever_provider(config_type=TestRetrieverProviderConfig)
    async def register7(config: TestRetrieverProviderConfig, builder: Builder):

        if (config.raise_error):
            raise ValueError("Error")

        yield RetrieverProviderInfo(config=config, description="Mock retriever to test the registration process")


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

        fn = await builder.set_workflow(FunctionReturningInfoConfig())
        assert isinstance(fn, Function)

        fn = await builder.set_workflow(FunctionReturningDerivedConfig())
        assert isinstance(fn, Function)

        with pytest.raises(ValueError):
            await builder.set_workflow(FunctionReturningBadConfig())

        # Try and add a function with the same name
        with pytest.warns():
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

        await builder.add_llm("llm_name", TestLLMProviderConfig())

        with pytest.raises(ValueError):
            await builder.add_llm("llm_name2", TestLLMProviderConfig(raise_error=True))

        # Try and add a llm with the same name
        with pytest.raises(ValueError):
            await builder.add_llm("llm_name", TestLLMProviderConfig())


async def test_get_llm():

    @register_llm_client(config_type=TestLLMProviderConfig, wrapper_type="test_framework")
    async def register(config: TestLLMProviderConfig, b: Builder):

        class TestFrameworkLLM(BaseModel):

            model_config = ConfigDict(arbitrary_types_allowed=True)

            config: TestLLMProviderConfig
            builder: Builder

        yield TestFrameworkLLM(config=config, builder=b)

    async with WorkflowBuilder() as builder:

        config = TestLLMProviderConfig()

        await builder.add_llm("llm_name", config)

        llm = await builder.get_llm("llm_name", wrapper_type="test_framework")

        assert llm.config == builder.get_llm_config("llm_name")

        with pytest.raises(ValueError):
            await builder.get_llm("llm_name_not_exist", wrapper_type="test_framework")


async def test_get_llm_config():

    async with WorkflowBuilder() as builder:

        config = TestLLMProviderConfig()

        await builder.add_llm("llm_name", config)

        assert builder.get_llm_config("llm_name") == config

        with pytest.raises(ValueError):
            builder.get_llm_config("llm_name_not_exist")


async def test_add_embedder():

    async with WorkflowBuilder() as builder:

        await builder.add_embedder("embedder_name", TestEmbedderProviderConfig())

        with pytest.raises(ValueError):
            await builder.add_embedder("embedder_name2", TestEmbedderProviderConfig(raise_error=True))

        # Try and add the same name
        with pytest.raises(ValueError):
            await builder.add_embedder("embedder_name", TestEmbedderProviderConfig())


async def test_get_embedder():

    @register_embedder_client(config_type=TestEmbedderProviderConfig, wrapper_type="test_framework")
    async def register(config: TestEmbedderProviderConfig, b: Builder):

        class TestFrameworkEmbedder(BaseModel):

            model_config = ConfigDict(arbitrary_types_allowed=True)

            config: TestEmbedderProviderConfig
            builder: Builder

        yield TestFrameworkEmbedder(config=config, builder=b)

    async with WorkflowBuilder() as builder:

        config = TestEmbedderProviderConfig()

        await builder.add_embedder("embedder_name", config)

        embedder = await builder.get_embedder("embedder_name", wrapper_type="test_framework")

        assert embedder.config == builder.get_embedder_config("embedder_name")

        with pytest.raises(ValueError):
            await builder.get_embedder("embedder_name_not_exist", wrapper_type="test_framework")


async def test_get_embedder_config():

    async with WorkflowBuilder() as builder:

        config = TestEmbedderProviderConfig()

        await builder.add_embedder("embedder_name", config)

        assert builder.get_embedder_config("embedder_name") == config

        with pytest.raises(ValueError):
            builder.get_embedder_config("embedder_name_not_exist")


async def test_add_memory():

    async with WorkflowBuilder() as builder:

        await builder.add_memory_client("memory_name", TestMemoryConfig())

        with pytest.raises(ValueError):
            await builder.add_memory_client("memory_name2", TestMemoryConfig(raise_error=True))

        # Try and add the same name
        with pytest.raises(ValueError):
            await builder.add_memory_client("memory_name", TestMemoryConfig())


async def test_get_memory():

    async with WorkflowBuilder() as builder:

        config = TestMemoryConfig()

        memory = await builder.add_memory_client("memory_name", config)

        assert memory == builder.get_memory_client("memory_name")

        with pytest.raises(ValueError):
            builder.get_memory_client("memory_name_not_exist")


async def test_get_memory_config():

    async with WorkflowBuilder() as builder:

        config = TestMemoryConfig()

        await builder.add_memory_client("memory_name", config)

        assert builder.get_memory_client_config("memory_name") == config

        with pytest.raises(ValueError):
            builder.get_memory_client_config("memory_name_not_exist")


async def test_add_retriever():

    async with WorkflowBuilder() as builder:
        await builder.add_retriever("retriever_name", TestRetrieverProviderConfig())

        with pytest.raises(ValueError):
            await builder.add_retriever("retriever_name2", TestRetrieverProviderConfig(raise_error=True))

        with pytest.raises(ValueError):
            await builder.add_retriever("retriever_name", TestRetrieverProviderConfig())


async def get_retriever():

    @register_retriever_client(config_type=TestRetrieverProviderConfig, wrapper_type="test_framework")
    async def register(config: TestRetrieverProviderConfig, b: Builder):

        class TestFrameworkRetriever(BaseModel):

            model_config = ConfigDict(arbitrary_types_allowed=True)

            config: TestRetrieverProviderConfig
            builder: Builder

        yield TestFrameworkRetriever(config=config, builder=b)

    @register_retriever_client(config_type=TestRetrieverProviderConfig, wrapper_type=None)
    async def register_no_framework(config: TestRetrieverProviderConfig, builder: Builder):

        class TestRetriever(AIQRetriever):

            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            async def search(self, query: str, **kwargs):
                return RetrieverOutput(results=[AIQDocument(page_content="page content", metadata={})])

            async def add_items(self, items):
                return await super().add_items(items)

            async def remove_items(self, **kwargs):
                return await super().remove_items(**kwargs)

        yield TestRetriever(**config.model_dump())

    async with WorkflowBuilder() as builder:

        config = TestRetrieverProviderConfig()

        await builder.add_retriever("retriever_name", config)

        retriever = await builder.get_retriever("retriever_name", wrapper_type="test_framework")

        assert retriever.config == builder.get_retriever_config("retriever_name")

        with pytest.raises(ValueError):
            await builder.get_retriever("retriever_name_not_exist", wrapper_type="test_framework")

        retriever = await builder.get_retriever("retriever_name", wrapper_type=None)

        assert isinstance(retriever, AIQRetriever)


async def get_retriever_config():

    async with WorkflowBuilder() as builder:

        config = TestRetrieverProviderConfig()

        await builder.add_retriever("retriever_name", config)

        assert builder.get_retriever_config("retriever_name") == config

        with pytest.raises(ValueError):
            builder.get_retriever_config("retriever_name_not_exist")


async def test_built_config():

    general_config = GeneralConfig(cache_dir="Something else")
    function_config = FunctionReturningFunctionConfig()
    workflow_config = FunctionReturningFunctionConfig()
    llm_config = TestLLMProviderConfig()
    embedder_config = TestEmbedderProviderConfig()
    memory_config = TestMemoryConfig()
    retriever_config = TestRetrieverProviderConfig()

    async with WorkflowBuilder(general_config=general_config) as builder:

        await builder.add_function("function1", function_config)

        await builder.set_workflow(workflow_config)

        await builder.add_llm("llm1", llm_config)

        await builder.add_embedder("embedder1", embedder_config)

        await builder.add_memory_client("memory1", memory_config)

        await builder.add_retriever("retriever1", retriever_config)

        workflow = builder.build()

        workflow_config = workflow.config

        assert workflow_config.general == general_config
        assert workflow_config.functions == {"function1": function_config}
        assert workflow_config.workflow == workflow_config.workflow
        assert workflow_config.llms == {"llm1": llm_config}
        assert workflow_config.embedders == {"embedder1": embedder_config}
        assert workflow_config.memory == {"memory1": memory_config}
        assert workflow_config.retrievers == {"retriever1": retriever_config}
