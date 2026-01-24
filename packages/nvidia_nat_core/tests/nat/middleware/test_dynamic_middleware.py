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
from nat.builder.function import Function
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.authentication import AuthResult
from nat.data_models.component import ComponentGroup
from nat.data_models.function import FunctionBaseConfig
from nat.memory.interfaces import MemoryEditor
from nat.middleware.dynamic.dynamic_function_middleware import DynamicFunctionMiddleware
from nat.middleware.dynamic.dynamic_middleware_config import DynamicMiddlewareConfig
from nat.middleware.utils.workflow_inventory import DiscoveredComponent
from nat.middleware.utils.workflow_inventory import DiscoveredFunction
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem
from nat.retriever.interface import Retriever
from nat.retriever.models import RetrieverOutput

# ==================== Fixtures ====================


@pytest.fixture
def mock_builder():
    """Create a mock builder with all required methods."""
    builder = Mock()
    builder._functions = {}
    builder.get_llm = AsyncMock()
    builder.get_embedder = AsyncMock()
    builder.get_retriever = AsyncMock()
    builder.get_memory_client = AsyncMock()
    builder.get_object_store_client = AsyncMock()
    builder.get_auth_provider = AsyncMock()
    builder.get_function = AsyncMock()
    builder.get_function_config = Mock()
    return builder


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


# ==================== Registration Tests ====================


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
    fake_registered = RegisteredComponentMethod(key="fake.method",
                                                component_instance=Mock(),
                                                function_name="method",
                                                original_callable=lambda: None)

    with pytest.raises(ValueError, match=r"'fake\.method' is not registered"):
        middleware.unregister(fake_registered)
