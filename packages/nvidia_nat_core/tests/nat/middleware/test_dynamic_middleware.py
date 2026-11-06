# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for DynamicFunctionMiddleware."""

from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

from nat.authentication.interfaces import AuthProviderBase
from nat.builder.builder import Builder
from nat.builder.function import Function
from nat.builder.function import FunctionInfo
from nat.builder.function import LambdaFunction
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.authentication import AuthResult
from nat.data_models.component import ComponentGroup
from nat.data_models.function import FunctionBaseConfig
from nat.memory.interfaces import MemoryEditor
from nat.middleware.dynamic.dynamic_function_middleware import DynamicFunctionMiddleware
from nat.middleware.dynamic.dynamic_middleware_config import DynamicMiddlewareConfig
from nat.middleware.middleware import InvocationContext
from nat.middleware.utils.workflow_inventory import DiscoveredComponent
from nat.middleware.utils.workflow_inventory import DiscoveredFunction
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem
from nat.retriever.interface import Retriever
from nat.retriever.models import RetrieverOutput

# ==================== Fixtures ====================


@pytest.fixture
def mock_function():
    """Create a mock NAT Function instance."""
    func = Mock(spec=Function)
    func.instance_name = "test_function"
    func.config = FunctionBaseConfig()
    func.middleware = []
    func.configure_middleware = Mock()
    func.instance = func
    return func


@pytest.fixture
def llm_client():
    """Mock LLM client."""

    class MockLLM:

        def invoke(self, messages, **kwargs):
            return "response"

        async def ainvoke(self, messages, **kwargs):
            return "response"

        def stream(self, messages, **kwargs):
            yield "chunk"

        async def astream(self, messages, **kwargs):
            yield "chunk"

    return MockLLM()


@pytest.fixture
def embedder_client():
    """Mock Embedder client."""

    class MockEmbedder:

        def embed_query(self, text):
            return [0.1, 0.2, 0.3]

        async def aembed_query(self, text):
            return [0.1, 0.2, 0.3]

    return MockEmbedder()


@pytest.fixture
def retriever_client():
    """Mock Retriever client."""

    class MockRetriever(Retriever):

        async def search(self, query, **kwargs):
            return RetrieverOutput(results=[])

    return MockRetriever()


@pytest.fixture
def memory_client():
    """Mock Memory client."""

    class MockMemory(MemoryEditor):

        async def search(self, query, top_k=5, **kwargs):
            return []

        async def add_items(self, items):
            pass

        async def remove_items(self, **kwargs):
            pass

    return MockMemory()


@pytest.fixture
def object_store_client():
    """Mock ObjectStore client."""

    class MockObjectStore(ObjectStore):

        def __init__(self):
            self._store = {}

        async def put_object(self, key, item):
            self._store[key] = item

        async def get_object(self, key):
            return self._store.get(key, ObjectStoreItem(data=b""))

        async def delete_object(self, key):
            self._store.pop(key, None)

        async def upsert_object(self, key, item):
            self._store[key] = item

    return MockObjectStore()


@pytest.fixture
def auth_provider_client():
    """Mock AuthProvider client."""

    class MockAuthProvider(AuthProviderBase[AuthProviderBaseConfig]):

        def __init__(self):
            super().__init__(config=AuthProviderBaseConfig())

        async def authenticate(self, user_id=None, **kwargs):
            return AuthResult()

    return MockAuthProvider()


# ==================== Helper Functions ====================


def create_function_context(name: str = "test_function",
                            config: dict | None = None,
                            description: str = "Test function"):
    """Helper to create FunctionMiddlewareContext (static metadata only)."""
    from nat.middleware.middleware import FunctionMiddlewareContext
    return FunctionMiddlewareContext(
        name=name,
        config=config or {},
        description=description,
        input_schema=None,
        single_output_schema=type(None),
        stream_output_schema=type(None),
    )


# ==================== Middleware Invoke/Stream Tests ====================


async def test_middleware_invoke_calls_next_with_no_policies(mock_builder):
    """Test that invoke delegates to call_next when no policies are configured."""
    config = DynamicMiddlewareConfig(register_workflow_functions=False)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    test_input = {"value": "test"}
    expected_output = {"result": "success"}

    async def mock_call_next(*args, **kwargs):
        assert args[0] == test_input
        return expected_output

    context = create_function_context()
    result = await middleware.function_middleware_invoke(test_input, call_next=mock_call_next, context=context)
    assert result == expected_output


async def test_middleware_stream_calls_next_with_no_policies(mock_builder):
    """Test that stream delegates to call_next when no policies are configured."""
    config = DynamicMiddlewareConfig(register_workflow_functions=False)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    async def mock_call_next(*args, **kwargs):
        yield "chunk1"
        yield "chunk2"
        yield "chunk3"

    context = create_function_context()

    chunks = []
    async for chunk in middleware.function_middleware_stream({}, call_next=mock_call_next, context=context):
        chunks.append(chunk)

    assert chunks == ["chunk1", "chunk2", "chunk3"]


# ==================== Component Discovery Tests ====================


async def test_discover_llm(mock_builder, llm_client):
    """Test LLM discovery and registration."""
    config = DynamicMiddlewareConfig(register_llms=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_llm = AsyncMock(return_value=llm_client)
    middleware._get_callable_functions = Mock(return_value={"invoke", "ainvoke", "stream", "astream"})

    result = await middleware._discover_and_register_llm("test_llm", "langchain")

    assert result == llm_client
    assert len(middleware._workflow_inventory.llms) == 1
    assert middleware._workflow_inventory.llms[0].name == "test_llm"


async def test_discover_embedder(mock_builder, embedder_client):
    """Test Embedder discovery and registration."""
    config = DynamicMiddlewareConfig(register_embedders=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_embedder = AsyncMock(return_value=embedder_client)
    middleware._get_callable_functions = Mock(return_value={"embed_query", "aembed_query"})

    result = await middleware._discover_and_register_embedder("test_embedder", "langchain")

    assert result == embedder_client
    assert len(middleware._workflow_inventory.embedders) == 1


async def test_discover_retriever(mock_builder, retriever_client):
    """Test Retriever discovery and registration."""
    config = DynamicMiddlewareConfig(register_retrievers=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_retriever = AsyncMock(return_value=retriever_client)
    middleware._get_callable_functions = Mock(return_value={"search"})

    result = await middleware._discover_and_register_retriever("test_retriever")

    assert result == retriever_client
    assert len(middleware._workflow_inventory.retrievers) == 1


async def test_discover_memory(mock_builder, memory_client):
    """Test Memory discovery and registration."""
    config = DynamicMiddlewareConfig(register_memory=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_memory = AsyncMock(return_value=memory_client)
    middleware._get_callable_functions = Mock(return_value={"search", "add_items", "remove_items"})

    result = await middleware._discover_and_register_memory("test_memory")

    assert result == memory_client
    assert len(middleware._workflow_inventory.memory) == 1


async def test_discover_object_store(mock_builder, object_store_client):
    """Test ObjectStore discovery and registration."""
    config = DynamicMiddlewareConfig(register_object_stores=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_object_store = AsyncMock(return_value=object_store_client)
    middleware._get_callable_functions = Mock(return_value={"put_object", "get_object", "delete_object"})

    result = await middleware._discover_and_register_object_store("test_store")

    assert result == object_store_client
    assert len(middleware._workflow_inventory.object_stores) == 1


async def test_discover_auth_provider(mock_builder, auth_provider_client):
    """Test AuthProvider discovery and registration."""
    config = DynamicMiddlewareConfig(register_auth_providers=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_auth_provider = AsyncMock(return_value=auth_provider_client)
    middleware._get_callable_functions = Mock(return_value={"authenticate"})

    result = await middleware._discover_and_register_auth_provider("test_auth")

    assert result == auth_provider_client
    assert len(middleware._workflow_inventory.auth_providers) == 1


async def test_discover_skips_if_not_configured(mock_builder, llm_client):
    """Test component is not registered if not configured."""
    config = DynamicMiddlewareConfig(register_llms=False)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_llm = AsyncMock(return_value=llm_client)

    result = await middleware._discover_and_register_llm("test_llm", "langchain")

    assert result == llm_client
    assert len(middleware._workflow_inventory.llms) == 0


async def test_discover_skips_duplicates(mock_builder, llm_client):
    """Test that duplicate components are not registered twice."""
    config = DynamicMiddlewareConfig(register_llms=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_llm = AsyncMock(return_value=llm_client)
    middleware._get_callable_functions = Mock(return_value={"invoke"})

    await middleware._discover_and_register_llm("test_llm", "langchain")
    await middleware._discover_and_register_llm("test_llm", "langchain")

    assert len(middleware._workflow_inventory.llms) == 1


# ==================== Workflow Function Tests ====================


def test_discover_functions_from_builder(mock_builder, mock_function):
    """Test workflow function discovery from builder."""
    mock_builder._functions = {"func1": mock_function, "func2": mock_function}

    config = DynamicMiddlewareConfig(register_workflow_functions=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert len(middleware._workflow_inventory.workflow_functions) == 2


def test_discover_functions_skip_if_not_configured(mock_builder, mock_function):
    """Test that functions are not discovered if not configured."""
    mock_builder._functions = {"func1": mock_function}

    config = DynamicMiddlewareConfig(register_workflow_functions=False)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert len(middleware._workflow_inventory.workflow_functions) == 0


def test_discover_functions_skip_duplicates(mock_builder, mock_function):
    """Test that duplicate functions are not discovered twice."""
    mock_builder._functions = {"func1": mock_function}

    config = DynamicMiddlewareConfig(register_workflow_functions=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._discover_functions()

    assert len(middleware._workflow_inventory.workflow_functions) == 1


# ==================== _should_intercept_function Tests ====================


def test_should_intercept_function_returns_false_if_already_registered(mock_builder, mock_function):
    """_should_intercept_function returns False when function is already in inventory."""
    config = DynamicMiddlewareConfig(register_workflow_functions=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    already = DiscoveredFunction(name="func1", config=FunctionBaseConfig(), instance=mock_function)
    middleware._workflow_inventory.workflow_functions.append(already)

    assert middleware._should_intercept_function("func1") is False


def test_should_intercept_function_returns_true_when_register_all_enabled(mock_builder):
    """_should_intercept_function returns True for any unseen function when register_workflow_functions is True."""
    config = DynamicMiddlewareConfig(register_workflow_functions=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=Mock(_functions={}))

    assert middleware._should_intercept_function("any_func") is True


@pytest.mark.parametrize("func_name,expected", [
    ("listed_func", True),
    ("unlisted_func", False),
])
def test_should_intercept_function_respects_explicit_list(mock_builder, func_name, expected):
    """_should_intercept_function returns True only for names in workflow_functions list."""
    config = DynamicMiddlewareConfig(register_workflow_functions=False, workflow_functions=["listed_func"])
    middleware = DynamicFunctionMiddleware(config=config, builder=Mock(_functions={}))

    assert middleware._should_intercept_function(func_name) is expected


def test_should_intercept_function_returns_false_when_both_disabled(mock_builder):
    """_should_intercept_function returns False when register_all is off and list is None."""
    config = DynamicMiddlewareConfig(register_workflow_functions=False, workflow_functions=None)
    middleware = DynamicFunctionMiddleware(config=config, builder=Mock(_functions={}))

    assert middleware._should_intercept_function("any_func") is False


# ==================== workflow_functions List Discovery Tests ====================


def test_discover_functions_targeted_list_registers_only_listed(mock_builder, mock_function):
    """_discover_functions only registers functions explicitly named in workflow_functions."""
    mock_builder._functions = {
        "func_a": Mock(config=FunctionBaseConfig(), instance=mock_function),
        "func_b": Mock(config=FunctionBaseConfig(), instance=mock_function),
        "func_c": Mock(config=FunctionBaseConfig(), instance=mock_function),
    }

    config = DynamicMiddlewareConfig(register_workflow_functions=False, workflow_functions=["func_a", "func_c"])
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    registered_names = {f.name for f in middleware._workflow_inventory.workflow_functions}
    assert registered_names == {"func_a", "func_c"}


def test_discover_functions_targeted_list_skips_functions_not_in_builder(mock_builder, mock_function):
    """_discover_functions silently skips names in workflow_functions that are not in the builder."""
    mock_builder._functions = {
        "func_a": Mock(config=FunctionBaseConfig(), instance=mock_function),
    }

    config = DynamicMiddlewareConfig(register_workflow_functions=False, workflow_functions=["func_a", "nonexistent"])
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    registered_names = {f.name for f in middleware._workflow_inventory.workflow_functions}
    assert registered_names == {"func_a"}


def test_discover_functions_register_all_is_not_limited_by_workflow_functions_list(mock_builder, mock_function):
    """register_workflow_functions=True registers all functions regardless of workflow_functions list."""
    mock_builder._functions = {
        "func_a": Mock(config=FunctionBaseConfig(), instance=mock_function),
        "func_b": Mock(config=FunctionBaseConfig(), instance=mock_function),
        "func_c": Mock(config=FunctionBaseConfig(), instance=mock_function),
    }

    config = DynamicMiddlewareConfig(register_workflow_functions=True, workflow_functions=["func_a"])
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    registered_names = {f.name for f in middleware._workflow_inventory.workflow_functions}
    assert registered_names == {"func_a", "func_b", "func_c"}


def test_discover_functions_skips_when_both_disabled(mock_builder, mock_function):
    """_discover_functions does nothing when register_workflow_functions is False and workflow_functions is None."""
    mock_builder._functions = {"func_a": Mock(config=FunctionBaseConfig(), instance=mock_function)}

    config = DynamicMiddlewareConfig(register_workflow_functions=False, workflow_functions=None)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert len(middleware._workflow_inventory.workflow_functions) == 0


# ==================== _discover_and_register_function Runtime Tests ====================


async def test_discover_and_register_function_skips_when_both_disabled(mock_builder, mock_function):
    """_discover_and_register_function returns function without registering when both flags are off."""
    config = DynamicMiddlewareConfig(register_workflow_functions=False, workflow_functions=None)
    middleware = DynamicFunctionMiddleware(config=config, builder=Mock(_functions={}))
    middleware._builder_get_function = AsyncMock(return_value=mock_function)

    result = await middleware._discover_and_register_function("any_func")

    assert result is mock_function
    assert len(middleware._workflow_inventory.workflow_functions) == 0


async def test_discover_and_register_function_registers_when_in_explicit_list(mock_builder, mock_function):
    """_discover_and_register_function registers a function that appears in workflow_functions list."""
    config = DynamicMiddlewareConfig(register_workflow_functions=False, workflow_functions=["target_func"])

    builder_mock = Mock(_functions={})
    builder_mock.get_function_config = Mock(return_value=FunctionBaseConfig())

    middleware = DynamicFunctionMiddleware(config=config, builder=builder_mock)
    middleware._builder_get_function = AsyncMock(return_value=mock_function)

    result = await middleware._discover_and_register_function("target_func")

    assert result is mock_function
    assert len(middleware._workflow_inventory.workflow_functions) == 1
    assert middleware._workflow_inventory.workflow_functions[0].name == "target_func"


async def test_discover_and_register_function_skips_when_not_in_explicit_list(mock_builder, mock_function):
    """_discover_and_register_function skips a function not in the workflow_functions list."""
    config = DynamicMiddlewareConfig(register_workflow_functions=False, workflow_functions=["other_func"])

    middleware = DynamicFunctionMiddleware(config=config, builder=Mock(_functions={}))
    middleware._builder_get_function = AsyncMock(return_value=mock_function)

    result = await middleware._discover_and_register_function("unlisted_func")

    assert result is mock_function
    assert len(middleware._workflow_inventory.workflow_functions) == 0


async def test_discover_and_register_function_registers_all_when_register_all_enabled(mock_builder, mock_function):
    """_discover_and_register_function registers any function when register_workflow_functions=True."""
    config = DynamicMiddlewareConfig(register_workflow_functions=True)

    builder_mock = Mock(_functions={})
    builder_mock.get_function_config = Mock(return_value=FunctionBaseConfig())

    middleware = DynamicFunctionMiddleware(config=config, builder=builder_mock)
    middleware._builder_get_function = AsyncMock(return_value=mock_function)

    await middleware._discover_and_register_function("func_x")
    await middleware._discover_and_register_function("func_y")

    registered_names = {f.name for f in middleware._workflow_inventory.workflow_functions}
    assert registered_names == {"func_x", "func_y"}


async def test_discover_and_register_function_skips_duplicate_in_list(mock_builder, mock_function):
    """_discover_and_register_function does not register the same function twice via the explicit list."""
    config = DynamicMiddlewareConfig(register_workflow_functions=False, workflow_functions=["target_func"])

    builder_mock = Mock(_functions={})
    builder_mock.get_function_config = Mock(return_value=FunctionBaseConfig())

    middleware = DynamicFunctionMiddleware(config=config, builder=builder_mock)
    middleware._builder_get_function = AsyncMock(return_value=mock_function)

    await middleware._discover_and_register_function("target_func")
    await middleware._discover_and_register_function("target_func")

    assert len(middleware._workflow_inventory.workflow_functions) == 1


# ==================== _patch_add_function_group Tests ====================


async def test_patch_add_function_group_registers_workflow_functions_after_group_builds(mock_builder, mock_function):
    """workflow_functions entries inside a function group are registered when the group is built."""
    config = DynamicMiddlewareConfig(register_workflow_functions=False, workflow_functions=["group__func_a"])

    builder_mock = Mock(_functions={})
    builder_mock.get_function_config = Mock(return_value=FunctionBaseConfig())

    async def fake_add_function_group(name, cfg):
        builder_mock._functions["group__func_a"] = Mock(config=FunctionBaseConfig(), instance=mock_function)
        builder_mock._functions["group__func_b"] = Mock(config=FunctionBaseConfig(), instance=mock_function)

    builder_mock.add_function_group = fake_add_function_group

    middleware = DynamicFunctionMiddleware(config=config, builder=builder_mock)

    assert len(middleware._workflow_inventory.workflow_functions) == 0

    await builder_mock.add_function_group("group", None)

    assert len(middleware._workflow_inventory.workflow_functions) == 1
    assert middleware._workflow_inventory.workflow_functions[0].name == "group__func_a"


async def test_patch_add_function_group_skips_patch_when_no_workflow_functions_configured(mock_function):
    """_patch_add_function_group does not replace add_function_group when neither flag is set."""
    config = DynamicMiddlewareConfig(register_workflow_functions=False, workflow_functions=None)

    original_sentinel = object()
    builder_mock = Mock(_functions={})
    builder_mock.add_function_group = original_sentinel

    DynamicFunctionMiddleware(config=config, builder=builder_mock)

    assert builder_mock.add_function_group is original_sentinel


# ==================== Registration Tests ====================


class _CountingDynamicMiddleware(DynamicFunctionMiddleware):
    """Records how many times its `pre_invoke` hook actually executes."""

    def __init__(self, config: DynamicMiddlewareConfig, builder: Builder):
        super().__init__(config=config, builder=builder)
        self.pre_invoke_calls: list[bool] = []

    async def pre_invoke(self, context: InvocationContext) -> None:
        self.pre_invoke_calls.append(True)


async def test_register_function_does_not_duplicate_middleware_already_in_chain():
    """Regression test for #2093: when a function is explicitly configured with a dynamic
    middleware instance and the dynamic registration later appends the same instance, the
    chain must contain the instance only once and it must execute once per invocation.
    """
    config = DynamicMiddlewareConfig(register_workflow_functions=True)
    middleware = _CountingDynamicMiddleware(config=config, builder=Mock(_functions={}))

    async def work(value: int) -> int:
        return value

    info = FunctionInfo.from_fn(work, description="example function")
    func = LambdaFunction.from_info(config=FunctionBaseConfig(), info=info, instance_name="example_function")

    # Explicit per-function middleware configuration (as WorkflowBuilder does when building functions)
    func.configure_middleware([middleware])

    discovered = DiscoveredFunction(name="example_function", config=FunctionBaseConfig(), instance=func)
    middleware._register_function(discovered)

    assert func.middleware == (middleware, ), "middleware instance must not appear twice in the chain"

    await func._middlewared_single(1)
    assert len(middleware.pre_invoke_calls) == 1, "middleware must execute exactly once per invocation"


def test_register_function_prevents_duplicates(mock_function):
    """Test that duplicate function registration is prevented."""
    config = DynamicMiddlewareConfig()
    middleware = DynamicFunctionMiddleware(config=config, builder=Mock(_functions={}))

    discovered = DiscoveredFunction(name="test_function", config=FunctionBaseConfig(), instance=mock_function)

    middleware._register_function(discovered)
    call_count_1 = discovered.instance.configure_middleware.call_count

    middleware._register_function(discovered)
    call_count_2 = discovered.instance.configure_middleware.call_count

    assert call_count_1 == call_count_2


def test_register_component_function_prevents_duplicates(llm_client):
    """Test that duplicate component function registration is prevented."""
    config = DynamicMiddlewareConfig()
    middleware = DynamicFunctionMiddleware(config=config, builder=Mock(_functions={}))

    discovered = DiscoveredComponent(name="gpt4",
                                     component_type=ComponentGroup.LLMS,
                                     instance=llm_client,
                                     callable_functions={"invoke"})

    # Register once
    middleware._register_component_function(discovered, "invoke")
    first_registered = middleware._registered_callables.get("gpt4.invoke")

    # Attempt to register again
    middleware._register_component_function(discovered, "invoke")

    assert "gpt4.invoke" in middleware._registered_callables
    # Should still have only one entry with the same object
    assert middleware._registered_callables["gpt4.invoke"] is first_registered


# ==================== Unregister Tests ====================


def test_unregister_workflow_function(mock_function):
    """Test unregistering a workflow function removes it from middleware interception."""
    config = DynamicMiddlewareConfig()
    middleware = DynamicFunctionMiddleware(config=config, builder=Mock(_functions={}))

    discovered = DiscoveredFunction(name="test_function", config=FunctionBaseConfig(), instance=mock_function)

    # Register the function
    middleware._register_function(discovered)
    assert "test_function" in middleware._registered_callables

    # Get the registered object
    registered = middleware._registered_callables["test_function"]

    # Unregister it
    middleware.unregister(registered)

    # Verify it's removed
    assert "test_function" not in middleware._registered_callables


def test_unregister_component_method(llm_client):
    """Test unregistering a component method restores the original callable."""
    config = DynamicMiddlewareConfig()
    middleware = DynamicFunctionMiddleware(config=config, builder=Mock(_functions={}))

    discovered = DiscoveredComponent(name="gpt4",
                                     component_type=ComponentGroup.LLMS,
                                     instance=llm_client,
                                     callable_functions={"invoke"})

    # Register the component function
    middleware._register_component_function(discovered, "invoke")
    assert "gpt4.invoke" in middleware._registered_callables

    # Get the registered object - it contains the original callable
    registered = middleware._registered_callables["gpt4.invoke"]
    original_callable = registered.original_callable

    # Unregister it
    middleware.unregister(registered)

    # Verify it's removed from tracking
    assert "gpt4.invoke" not in middleware._registered_callables

    # Verify original method is restored (compare by checking it's the stored original)
    assert llm_client.invoke is original_callable


def test_unregister_raises_error_if_not_registered(mock_function):
    """Test that unregistering a non-registered callable raises ValueError."""
    from nat.middleware.utils.workflow_inventory import RegisteredFunction

    config = DynamicMiddlewareConfig()
    middleware = DynamicFunctionMiddleware(config=config, builder=Mock(_functions={}))

    # Create a registered function object that's not actually registered
    fake_registered = RegisteredFunction(key="nonexistent", function_instance=mock_function)

    with pytest.raises(ValueError, match="'nonexistent' is not registered"):
        middleware.unregister(fake_registered)


def test_unregister_component_method_raises_error_if_not_registered():
    """Test that unregistering a non-registered component method raises ValueError."""
    from nat.middleware.utils.workflow_inventory import RegisteredComponentMethod

    config = DynamicMiddlewareConfig()
    middleware = DynamicFunctionMiddleware(config=config, builder=Mock(_functions={}))

    # Create a registered component method object that's not actually registered
    fake_registered = RegisteredComponentMethod(key="fake__method",
                                                component_instance=Mock(),
                                                function_name="method",
                                                original_callable=lambda: None)

    with pytest.raises(ValueError, match=r"'fake__method' is not registered"):
        middleware.unregister(fake_registered)


# ==================== WorkflowBuilder Integration Tests ====================


async def test_dynamic_middleware_patches_workflow_builder():
    """DynamicFunctionMiddleware patches persist on the builder used by the workflow."""
    from nat.builder.builder import Builder
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.cli.register_workflow import register_middleware

    class _PatchTestConfig(DynamicMiddlewareConfig, name="_patch_regression_test"):
        pass

    @register_middleware(config_type=_PatchTestConfig)
    async def _patch_test_middleware(config: _PatchTestConfig, builder: Builder):
        yield DynamicFunctionMiddleware(config=config, builder=builder)

    config = _PatchTestConfig(register_llms=True)

    async with WorkflowBuilder() as builder:
        middleware = await builder.add_middleware("patch_test", config)

        assert isinstance(middleware, DynamicFunctionMiddleware)
        assert builder.get_llm == middleware._discover_and_register_llm


@pytest.mark.parametrize("group,getter,method,args",
                         [
                             ("llms", "get_llm", "ainvoke", ("shared", "test")),
                             ("embedders", "get_embedder", "aembed_query", ("shared", "test")),
                             ("retrievers", "get_retriever", "search", ("shared", "test")),
                             ("memory", "get_memory_client", "search", ("shared", )),
                             ("object_stores", "get_object_store_client", "get_object", ("shared", )),
                             ("auth_providers", "get_auth_provider", "authenticate", ("shared", )),
                         ])
@pytest.mark.parametrize("target", ["auto", "explicit", "unselected"])
async def test_component_middleware_tracks_client_identity(group, getter, method, args, target):
    """Every selected client is intercepted once, even if clients compare equal."""

    async def operation(self, value):
        return value

    # Framework clients may be unhashable or define value-based equality.
    client_type = type("Client", (), {method: operation, "__eq__": lambda self, other: True, "__hash__": None})
    first, second = client_type(), client_type()
    builder = Mock(spec=Builder, _functions={})
    setattr(builder, getter, AsyncMock(side_effect=[first, second, first, second]))
    config = ({
        f"register_{group}": True
    } if target == "auto" else {
        group: ["shared" if target == "explicit" else "other"]
    })
    middleware = _CountingDynamicMiddleware(DynamicMiddlewareConfig(**config), builder)

    for expected in (first, second, first, second):
        client = await getattr(builder, getter)(*args)
        assert client is expected
        assert await getattr(client, method)("value") == "value"

    expected_count = 0 if target == "unselected" else 4
    assert len(middleware.pre_invoke_calls) == expected_count
    assert len(getattr(middleware._workflow_inventory, group)) == (0 if target == "unselected" else 2)

    # Every instance must remain independently addressable and removable.
    keys = middleware.get_registered_keys()
    assert len(keys) == (0 if target == "unselected" else 2)
    for key in keys:
        middleware.unregister(middleware.get_registered(key))
    assert middleware.get_registered_keys() == []
    for client in (first, second):
        assert await getattr(client, method)("value") == "value"
    assert len(middleware.pre_invoke_calls) == expected_count


async def test_component_stream_middleware_applies_to_distinct_clients(llm_client):
    """Streaming methods on every client run middleware exactly once."""
    first, second = llm_client, type(llm_client)()
    builder = Mock(spec=Builder, _functions={})
    builder.get_llm = AsyncMock(side_effect=[first, second, first, second])
    middleware = _CountingDynamicMiddleware(DynamicMiddlewareConfig(register_llms=True), builder)

    for expected in (first, second, first, second):
        client = await builder.get_llm("shared", "test")
        assert client is expected
        assert [chunk async for chunk in client.astream("hi")] == ["chunk"]
    assert len(middleware.pre_invoke_calls) == 4


@pytest.fixture(scope="module")
def shared_llm_configs():
    """Register a provider that creates a fresh client on each real builder call."""
    import asyncio

    from nat.builder.llm import LLMProviderInfo
    from nat.cli.register_workflow import register_llm_client
    from nat.cli.register_workflow import register_llm_provider
    from nat.cli.register_workflow import register_middleware
    from nat.data_models.llm import LLMBaseConfig

    class SharedLLMConfig(LLMBaseConfig, name="_middleware_shared_client_test"):
        pass

    @register_llm_provider(config_type=SharedLLMConfig)
    async def provider(config, builder):
        yield LLMProviderInfo(config=config, description="Test shared LLM")

    class SlowClient:

        async def ainvoke(self, messages, **kwargs):
            await asyncio.sleep(1)
            return "model output"

    @register_llm_client(config_type=SharedLLMConfig, wrapper_type="_middleware_shared_client_test")
    async def client(config, builder):
        yield SlowClient()

    class DenyConfig(DynamicMiddlewareConfig, name="_middleware_deny_client_test"):
        pass

    class DenyMiddleware(DynamicFunctionMiddleware):

        async def function_middleware_invoke(self, *args, call_next, context, **kwargs):
            raise PermissionError("blocked by policy")

    @register_middleware(config_type=DenyConfig)
    async def deny(config, builder):
        yield DenyMiddleware(config=config, builder=builder)

    class ObserveConfig(DynamicMiddlewareConfig, name="_middleware_observe_client_test"):
        pass

    @register_middleware(config_type=ObserveConfig)
    async def observe(config, builder):
        yield _CountingDynamicMiddleware(config=config, builder=builder)

    return SharedLLMConfig, DenyConfig, ObserveConfig


@pytest.mark.parametrize("target", ["auto", "explicit"])
@pytest.mark.parametrize("policy", ["deny", "timeout"])
async def test_workflow_builder_middleware_protects_every_llm_client(shared_llm_configs, target, policy):
    """A tool and agent sharing one LLM must both receive the configured policy."""
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.middleware.timeout.timeout_middleware_config import TimeoutMiddlewareConfig

    llm_config, deny_config, _ = shared_llm_configs
    flags = {"register_llms": True} if target == "auto" else {"llms": ["shared"]}
    config = deny_config(**flags) if policy == "deny" else TimeoutMiddlewareConfig(timeout=0.01, **flags)
    expected_error = PermissionError if policy == "deny" else TimeoutError

    async with WorkflowBuilder() as builder:
        await builder.add_llm("shared", llm_config())
        await builder.add_middleware("policy", config)
        clients = [await builder.get_llm("shared", "_middleware_shared_client_test") for _ in range(3)]
        assert len({id(client) for client in clients}) == 3
        for client in clients:
            with pytest.raises(expected_error):
                await client.ainvoke("hi")


async def test_component_registration_keys_remain_stable_after_unregister():
    """Same-named components have independent registrations across groups and clients."""

    class Client:

        async def search(self, query):
            return query

    middleware = _CountingDynamicMiddleware(DynamicMiddlewareConfig(), Mock(spec=Builder, _functions={}))
    components = [
        DiscoveredComponent(name="shared", component_type=group, instance=Client(), callable_functions={"search"})
        for group in (ComponentGroup.RETRIEVERS, ComponentGroup.RETRIEVERS, ComponentGroup.MEMORY)
    ]
    for component in components:
        middleware._register_component_function(component, "search")

    keys = middleware.get_registered_keys()
    assert len(keys) == 3
    assert keys[0] == "shared.search"
    middleware.unregister(middleware.get_registered(keys[0]))

    # Removing the first registration must not let remaining instances be wrapped again.
    for component in components[1:]:
        middleware._register_component_function(component, "search")
        assert await component.instance.search("query") == "query"
    assert middleware.get_registered_keys() == keys[1:]
    assert len(middleware.pre_invoke_calls) == 2
    assert await components[0].instance.search("query") == "query"
    assert len(middleware.pre_invoke_calls) == 2

    for key in keys[1:]:
        middleware.unregister(middleware.get_registered(key))
    for component in components:
        assert await component.instance.search("query") == "query"
    assert len(middleware.pre_invoke_calls) == 2


async def test_stale_registration_cannot_unregister_a_new_client(llm_client):
    """A reused legacy key must not make an old registration handle valid again."""
    first, second = llm_client, type(llm_client)()
    builder = Mock(spec=Builder, _functions={})
    builder.get_llm = AsyncMock(side_effect=[first, second])
    middleware = _CountingDynamicMiddleware(DynamicMiddlewareConfig(register_llms=True), builder)

    await builder.get_llm("shared", "test")
    stale = middleware.get_registered("shared.ainvoke")
    middleware.unregister(stale)
    await builder.get_llm("shared", "test")
    current = middleware.get_registered("shared.ainvoke")
    assert current.component_instance is second

    with pytest.raises(ValueError, match="'shared.ainvoke' is not registered"):
        middleware.unregister(stale)
    assert middleware.get_registered("shared.ainvoke") is current
    assert await second.ainvoke("hi") == "response"
    assert len(middleware.pre_invoke_calls) == 1

    middleware.unregister(current)
    assert await second.ainvoke("hi") == "response"
    assert len(middleware.pre_invoke_calls) == 1


@pytest.mark.parametrize("group,getter,method,args",
                         [
                             ("llms", "get_llm", "ainvoke", ("shared", "test")),
                             ("embedders", "get_embedder", "aembed_query", ("shared", "test")),
                             ("retrievers", "get_retriever", "search", ("shared", "test")),
                             ("memory", "get_memory_client", "search", ("shared", )),
                             ("object_stores", "get_object_store_client", "get_object", ("shared", )),
                             ("auth_providers", "get_auth_provider", "authenticate", ("shared", )),
                         ])
@pytest.mark.parametrize("target", ["auto", "explicit"])
@pytest.mark.parametrize("removal_order", [(0, 1), (1, 0)])
async def test_component_middleware_composes_and_unregisters_independently(group,
                                                                           getter,
                                                                           method,
                                                                           args,
                                                                           target,
                                                                           removal_order):
    """Every policy runs in order on every client, including after partial removal."""
    events = []

    class RecordingMiddleware(DynamicFunctionMiddleware):

        def __init__(self, config, builder, label):
            self.label = label
            super().__init__(config, builder)

        async def pre_invoke(self, context):
            events.append(f"pre:{self.label}")

        async def post_invoke(self, context):
            events.append(f"post:{self.label}")

    async def operation(self, value):
        events.append("target")
        return value

    client_type = type("Client", (), {method: operation})
    clients = [client_type(), client_type()]
    builder = Mock(spec=Builder, _functions={})
    setattr(builder, getter, AsyncMock(side_effect=clients * 2))
    config = {f"register_{group}": True} if target == "auto" else {group: ["shared"]}
    middleware = [RecordingMiddleware(DynamicMiddlewareConfig(**config), builder, label) for label in ("A", "B")]

    for expected in clients * 2:
        client = await getattr(builder, getter)(*args)
        assert client is expected
        events.clear()
        assert await getattr(client, method)("value") == "value"
        assert events == ["pre:A", "pre:B", "target", "post:B", "post:A"]

    first_originals = [
        middleware[0].get_registered(key).original_callable for key in middleware[0].get_registered_keys()
    ]
    for index in removal_order:
        for key in middleware[index].get_registered_keys():
            middleware[index].unregister(middleware[index].get_registered(key))
        remaining = [mw.label for mw in middleware if mw.get_registered_keys()]
        for client in clients:
            events.clear()
            assert await getattr(client, method)("value") == "value"
            assert events == ([f"pre:{label}" for label in remaining] + ["target"] +
                              [f"post:{label}" for label in reversed(remaining)])
    for client, original in zip(clients, first_originals):
        assert getattr(client, method) is original


@pytest.mark.parametrize("removal_order", [(0, 1), (1, 0)])
async def test_streaming_component_middleware_composes_and_closes(removal_order):
    """Streaming preserves both policies, early-close cleanup, and independent removal."""
    events = []

    class Client:

        async def astream(self, value):
            try:
                yield value
                yield "second"
            finally:
                events.append("closed")

    class RecordingMiddleware(DynamicFunctionMiddleware):

        def __init__(self, builder, label):
            self.label = label
            super().__init__(DynamicMiddlewareConfig(register_llms=True), builder)

        async def pre_invoke(self, context):
            events.append(self.label)

    client = Client()
    builder = Mock(spec=Builder, _functions={})
    builder.get_llm = AsyncMock(return_value=client)
    middleware = [RecordingMiddleware(builder, label) for label in ("A", "B")]
    await builder.get_llm("shared", "test")
    original = middleware[0].get_registered("shared.astream").original_callable

    stream = client.astream("first")
    assert await anext(stream) == "first"
    assert events == ["A", "B"]
    await stream.aclose()
    assert events == ["A", "B", "closed"]

    for index in removal_order:
        middleware[index].unregister(middleware[index].get_registered("shared.astream"))
        events.clear()
        assert [chunk async for chunk in client.astream("first")] == ["first", "second"]
        remaining = [mw.label for mw in middleware if mw.get_registered_keys()]
        assert events == remaining + ["closed"]
    assert client.astream is original


@pytest.mark.parametrize("disabled_index", [0, 1])
async def test_disabled_component_middleware_does_not_remove_active_policy(llm_client, disabled_index):
    builder = Mock(spec=Builder, _functions={})
    builder.get_llm = AsyncMock(return_value=llm_client)
    middleware = [
        _CountingDynamicMiddleware(DynamicMiddlewareConfig(register_llms=True, enabled=index != disabled_index),
                                   builder) for index in range(2)
    ]
    await builder.get_llm("shared", "test")
    disabled, active = middleware[disabled_index], middleware[1 - disabled_index]
    await llm_client.ainvoke("hi")
    assert len(disabled.pre_invoke_calls) == 0
    assert len(active.pre_invoke_calls) == 1
    disabled.unregister(disabled.get_registered("shared.ainvoke"))
    await llm_client.ainvoke("hi")
    assert len(active.pre_invoke_calls) == 2


@pytest.mark.parametrize("first_is_final", [False, True])
async def test_component_middleware_rejects_invalid_final_order(llm_client, first_is_final):
    """Invalid terminal-policy configuration must fail rather than skip a policy."""
    builder = Mock(spec=Builder, _functions={})
    builder.get_llm = AsyncMock(return_value=llm_client)
    config = DynamicMiddlewareConfig(register_llms=True)
    DynamicFunctionMiddleware(config, builder, is_final=first_is_final)
    DynamicFunctionMiddleware(config, builder, is_final=True)
    if first_is_final:
        expected_error = "Only one final middleware"
    else:
        # A non-final middleware cannot be appended after the terminal boundary.
        DynamicFunctionMiddleware(config, builder)
        expected_error = "not the last middleware"
    for _ in range(2):
        with pytest.raises(ValueError, match=expected_error):
            await builder.get_llm("shared", "test")


@pytest.mark.parametrize("target", ["auto", "explicit"])
@pytest.mark.parametrize("policy", ["deny", "timeout"])
async def test_workflow_builder_composes_observation_and_policy(shared_llm_configs, target, policy):
    """An earlier observer must not hide clients from a later enforcement middleware."""
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.middleware.timeout.timeout_middleware_config import TimeoutMiddlewareConfig

    llm_config, deny_config, observe_config = shared_llm_configs
    flags = {"register_llms": True} if target == "auto" else {"llms": ["shared"]}
    config = deny_config(**flags) if policy == "deny" else TimeoutMiddlewareConfig(timeout=0.01, **flags)
    expected_error = PermissionError if policy == "deny" else TimeoutError
    async with WorkflowBuilder() as builder:
        await builder.add_llm("shared", llm_config())
        observer = await builder.add_middleware("observer", observe_config(**flags))
        enforcement = await builder.add_middleware("policy", config)
        for _ in range(3):
            client = await builder.get_llm("shared", "_middleware_shared_client_test")
            with pytest.raises(expected_error):
                await client.ainvoke("hi")
        assert len(observer.pre_invoke_calls) == 3
        assert len(enforcement.get_registered_keys()) == 3
