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

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from nat.builder.context import ActiveFunctionContextManager
from nat.builder.context import AIQContext
from nat.builder.context import AIQContextState
from nat.builder.context import Context
from nat.builder.context import ContextState
from nat.builder.context import Singleton
from nat.builder.intermediate_step_manager import IntermediateStepManager
from nat.builder.user_interaction_manager import UserInteractionManager
from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive import InteractionPrompt
from nat.data_models.invocation_node import InvocationNode
from nat.runtime.user_metadata import RequestAttributes
from nat.utils.reactive.subject import Subject

# --------------------------------------------------------------------------- #
# Test Fixtures and Helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def context_state():
    """Create a fresh ContextState instance for testing."""
    # Reset singleton instance to ensure clean state
    if hasattr(ContextState, 'instance'):
        ContextState.instance = None
    return ContextState()


@pytest.fixture
def context(context_state):
    """Create a Context instance with a fresh ContextState."""
    return Context(context_state)


@pytest.fixture
def mock_auth_config():
    """Create a mock AuthProviderBaseConfig."""
    return MagicMock(spec=AuthProviderBaseConfig)


@pytest.fixture
def mock_authenticated_context():
    """Create a mock AuthenticatedContext."""
    return MagicMock(spec=AuthenticatedContext)


@pytest.fixture
def mock_interaction_prompt():
    """Create a mock InteractionPrompt."""
    return MagicMock(spec=InteractionPrompt)


@pytest.fixture
def mock_human_response():
    """Create a mock HumanResponse."""
    return MagicMock(spec=HumanResponse)


# --------------------------------------------------------------------------- #
# Test Singleton Metaclass
# --------------------------------------------------------------------------- #


class TestSingleton:
    """Test the Singleton metaclass behavior."""

    def test_singleton_creates_single_instance(self):
        """Test that Singleton metaclass creates only one instance."""

        class TestClass(metaclass=Singleton):

            def __init__(self, value=None):
                self.value = value

        instance1 = TestClass("first")
        instance2 = TestClass("second")

        assert instance1 is instance2
        assert instance1.value == "first"  # First initialization wins
        assert instance2.value == "first"

    def test_singleton_different_classes_different_instances(self):
        """Test that different classes with Singleton metaclass have different instances."""

        class TestClass1(metaclass=Singleton):
            pass

        class TestClass2(metaclass=Singleton):
            pass

        instance1 = TestClass1()
        instance2 = TestClass2()

        assert instance1 is not instance2
        assert type(instance1) is not type(instance2)


# --------------------------------------------------------------------------- #
# Test ActiveFunctionContextManager
# --------------------------------------------------------------------------- #


class TestActiveFunctionContextManager:
    """Test the ActiveFunctionContextManager class."""

    def test_initialization(self):
        """Test ActiveFunctionContextManager initialization."""
        manager = ActiveFunctionContextManager()
        assert manager.output is None

    def test_set_and_get_output(self):
        """Test setting and getting output."""
        manager = ActiveFunctionContextManager()
        test_output = {"result": "test"}

        manager.set_output(test_output)
        assert manager.output == test_output

    def test_output_can_be_any_type(self):
        """Test that output can be any type."""
        manager = ActiveFunctionContextManager()

        # Test with different types
        test_values = ["string", 42, [1, 2, 3], {"key": "value"}, None, True]

        for value in test_values:
            manager.set_output(value)
            assert manager.output == value


# --------------------------------------------------------------------------- #
# Test ContextState
# --------------------------------------------------------------------------- #


class TestContextState:
    """Test the ContextState class."""

    def test_singleton_behavior(self, context_state):
        """Test that ContextState follows singleton pattern."""
        state1 = ContextState.get()
        state2 = ContextState.get()
        assert state1 is state2

    def test_initialization_sets_context_vars(self, context_state):
        """Test that initialization creates all required ContextVars."""
        assert hasattr(context_state, 'conversation_id')
        assert hasattr(context_state, 'user_message_id')
        assert hasattr(context_state, 'input_message')
        assert hasattr(context_state, 'user_manager')
        assert hasattr(context_state, '_metadata')
        assert hasattr(context_state, '_event_stream')
        assert hasattr(context_state, '_active_function')
        assert hasattr(context_state, '_active_span_id_stack')
        assert hasattr(context_state, 'user_input_callback')
        assert hasattr(context_state, 'user_auth_callback')

    def test_context_vars_default_values(self, context_state):
        """Test default values of ContextVars."""
        assert context_state.conversation_id.get() is None
        assert context_state.user_message_id.get() is None
        assert context_state.input_message.get() is None
        assert context_state.user_manager.get() is None
        assert context_state._metadata.get() is None
        assert context_state._event_stream.get() is None
        assert context_state._active_function.get() is None
        assert context_state._active_span_id_stack.get() is None
        assert context_state.user_input_callback.get() is not None  # Has default
        assert context_state.user_auth_callback.get() is None

    def test_metadata_property_lazy_initialization(self, context_state):
        """Test that metadata property creates RequestAttributes when None."""
        # Initially None
        assert context_state._metadata.get() is None

        # Accessing property should create instance
        metadata = context_state.metadata.get()
        assert isinstance(metadata, RequestAttributes)

        # Should be the same instance on subsequent calls
        metadata2 = context_state.metadata.get()
        assert metadata is metadata2

    def test_active_function_property_lazy_initialization(self, context_state):
        """Test that active_function property creates root InvocationNode when None."""
        # Initially None
        assert context_state._active_function.get() is None

        # Accessing property should create root node
        active_func = context_state.active_function.get()
        assert isinstance(active_func, InvocationNode)
        assert active_func.function_id == "root"
        assert active_func.function_name == "root"

        # Should be the same instance on subsequent calls
        active_func2 = context_state.active_function.get()
        assert active_func is active_func2

    def test_event_stream_property_lazy_initialization(self, context_state):
        """Test that event_stream property creates Subject when None."""
        # Initially None
        assert context_state._event_stream.get() is None

        # Accessing property should create Subject
        stream = context_state.event_stream.get()
        assert isinstance(stream, Subject)

        # Should be the same instance on subsequent calls
        stream2 = context_state.event_stream.get()
        assert stream is stream2

    def test_active_span_id_stack_property_lazy_initialization(self, context_state):
        """Test that active_span_id_stack property creates default stack when None."""
        # Initially None
        assert context_state._active_span_id_stack.get() is None

        # Accessing property should create default stack
        stack = context_state.active_span_id_stack.get()
        assert isinstance(stack, list)
        assert stack == ["root"]

        # Should be the same instance on subsequent calls
        stack2 = context_state.active_span_id_stack.get()
        assert stack is stack2


# --------------------------------------------------------------------------- #
# Test Context
# --------------------------------------------------------------------------- #


class TestContext:
    """Test the Context class."""

    def test_initialization(self, context_state):
        """Test Context initialization."""
        context = Context(context_state)
        assert context._context_state is context_state

    def test_input_message_property(self, context):
        """Test input_message property."""
        # Initially None
        assert context.input_message is None

        # Set value and test
        test_message = "test message"
        context._context_state.input_message.set(test_message)
        assert context.input_message == test_message

    def test_user_manager_property(self, context):
        """Test user_manager property."""
        # Initially None
        assert context.user_manager is None

        # Set value and test
        test_manager = MagicMock()
        context._context_state.user_manager.set(test_manager)
        assert context.user_manager is test_manager

    def test_metadata_property(self, context):
        """Test metadata property."""
        metadata = context.metadata
        assert isinstance(metadata, RequestAttributes)

    def test_user_interaction_manager_property(self, context):
        """Test user_interaction_manager property."""
        manager = context.user_interaction_manager
        assert isinstance(manager, UserInteractionManager)

    def test_intermediate_step_manager_property(self, context):
        """Test intermediate_step_manager property."""
        manager = context.intermediate_step_manager
        assert isinstance(manager, IntermediateStepManager)

    def test_conversation_id_property(self, context):
        """Test conversation_id property."""
        # Initially None
        assert context.conversation_id is None

        # Set value and test
        test_id = "conv-123"
        context._context_state.conversation_id.set(test_id)
        assert context.conversation_id == test_id

    def test_user_message_id_property(self, context):
        """Test user_message_id property."""
        # Initially None
        assert context.user_message_id is None

        # Set value and test
        test_id = "msg-456"
        context._context_state.user_message_id.set(test_id)
        assert context.user_message_id == test_id

    def test_active_function_property(self, context):
        """Test active_function property."""
        active_func = context.active_function
        assert isinstance(active_func, InvocationNode)
        assert active_func.function_id == "root"
        assert active_func.function_name == "root"

    def test_active_span_id_property(self, context):
        """Test active_span_id property."""
        span_id = context.active_span_id
        assert span_id == "root"

        # Test with modified stack
        context._context_state.active_span_id_stack.get().append("child-span")
        assert context.active_span_id == "child-span"

    def test_user_auth_callback_property_success(self, context, mock_auth_config, mock_authenticated_context):
        """Test user_auth_callback property when callback is set."""

        async def mock_callback(config, flow_type):
            return mock_authenticated_context

        context._context_state.user_auth_callback.set(mock_callback)
        callback = context.user_auth_callback
        assert callback is mock_callback

    def test_user_auth_callback_property_not_set(self, context):
        """Test user_auth_callback property when callback is not set."""
        with pytest.raises(RuntimeError, match="User authentication callback is not set in the context"):
            _ = context.user_auth_callback

    def test_get_static_method(self):
        """Test Context.get() static method."""
        context = Context.get()
        assert isinstance(context, Context)
        assert isinstance(context._context_state, ContextState)


# --------------------------------------------------------------------------- #
# Test push_active_function Context Manager
# --------------------------------------------------------------------------- #


class TestPushActiveFunctionContextManager:
    """Test the push_active_function context manager."""

    def test_push_active_function_basic_functionality(self, context):
        """Test basic functionality of push_active_function."""
        function_name = "test_function"
        input_data = {"param": "value"}

        # Test that context manager works and returns correct manager
        with context.push_active_function(function_name, input_data) as manager:
            assert isinstance(manager, ActiveFunctionContextManager)

            # Check that active function was set
            active_func = context.active_function
            assert active_func.function_name == function_name
            assert active_func.parent_id == "root"
            assert active_func.parent_name == "root"

            # Set output for testing
            test_output = "test result"
            manager.set_output(test_output)
            assert manager.output == test_output

    def test_push_active_function_with_exception(self, context):
        """Test function execution with exception in push_active_function."""
        function_name = "failing_function"
        input_data = {"param": "value"}
        test_exception = ValueError("Test error")

        # Exception should be re-raised
        with pytest.raises(ValueError, match="Test error"):
            with context.push_active_function(function_name, input_data):
                raise test_exception

    def test_push_active_function_restores_previous_function(self, context):
        """Test that push_active_function restores the previous active function."""
        # Set initial active function
        initial_func = InvocationNode(function_id="initial", function_name="initial_func")
        context._context_state.active_function.set(initial_func)

        function_name = "nested_function"
        input_data = {}

        with context.push_active_function(function_name, input_data):
            # Inside context, should have new function
            active_func = context.active_function
            assert active_func.function_name == function_name
            assert active_func.parent_id == "initial"
            assert active_func.parent_name == "initial_func"

        # After context, should restore initial function
        restored_func = context.active_function
        assert restored_func is initial_func

    def test_push_active_function_nested_calls(self, context):
        """Test nested push_active_function calls."""
        with context.push_active_function("func1", {"data": 1}) as manager1:
            func1 = context.active_function
            assert func1.function_name == "func1"
            assert func1.parent_name == "root"

            with context.push_active_function("func2", {"data": 2}) as manager2:
                func2 = context.active_function
                assert func2.function_name == "func2"
                assert func2.parent_name == "func1"

                # Both managers should be different instances
                assert manager1 is not manager2

            # Should restore to func1
            restored_func1 = context.active_function
            assert restored_func1 is func1

        # Should restore to root
        root_func = context.active_function
        assert root_func.function_name == "root"

    def test_push_active_function_with_none_input_data(self, context):
        """Test push_active_function with None input data."""
        function_name = "test_function"
        with context.push_active_function(function_name, None) as manager:
            assert isinstance(manager, ActiveFunctionContextManager)
            active_func = context.active_function
            assert active_func.function_name == function_name


# --------------------------------------------------------------------------- #
# Test Context in Multi-threading Environment
# --------------------------------------------------------------------------- #


class TestContextMultiThreading:
    """Test Context behavior in multi-threading scenarios."""

    def test_context_vars_isolation_between_threads(self):
        """Test that ContextVars maintain isolation between threads."""
        results = {}

        def worker(thread_id):
            context = Context.get()
            # Set thread-specific values
            context._context_state.conversation_id.set(f"conv-{thread_id}")
            context._context_state.user_message_id.set(f"msg-{thread_id}")

            # Small delay to ensure threads overlap
            time.sleep(0.01)

            # Read values back
            results[thread_id] = {
                'conversation_id': context.conversation_id, 'user_message_id': context.user_message_id
            }

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i, ))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify each thread maintained its own values
        for i in range(5):
            assert results[i]['conversation_id'] == f"conv-{i}"
            assert results[i]['user_message_id'] == f"msg-{i}"

    @pytest.mark.asyncio
    async def test_context_vars_isolation_in_async_tasks(self):
        """Test that ContextVars maintain isolation in async tasks."""
        results = {}

        async def async_worker(task_id):
            context = Context.get()
            # Set task-specific values
            context._context_state.conversation_id.set(f"conv-{task_id}")
            context._context_state.user_message_id.set(f"msg-{task_id}")

            # Small delay to ensure tasks overlap
            await asyncio.sleep(0.01)

            # Read values back
            results[task_id] = {'conversation_id': context.conversation_id, 'user_message_id': context.user_message_id}

        # Start multiple async tasks
        tasks = []
        for i in range(5):
            task = asyncio.create_task(async_worker(i))
            tasks.append(task)

        # Wait for all tasks
        await asyncio.gather(*tasks)

        # Verify each task maintained its own values
        for i in range(5):
            assert results[i]['conversation_id'] == f"conv-{i}"
            assert results[i]['user_message_id'] == f"msg-{i}"


# --------------------------------------------------------------------------- #
# Test Integration with Managers
# --------------------------------------------------------------------------- #


class TestContextManagerIntegration:
    """Test Context integration with various managers."""

    def test_user_interaction_manager_integration(self, context, mock_interaction_prompt, mock_human_response):
        """Test integration with UserInteractionManager."""

        # Set up mock callback
        async def mock_callback(prompt):
            return mock_human_response

        context._context_state.user_input_callback.set(mock_callback)

        # Get manager and verify it uses the context
        user_manager = context.user_interaction_manager
        assert isinstance(user_manager, UserInteractionManager)

        # The manager should have access to the context state
        assert user_manager._context_state is context._context_state

    def test_intermediate_step_manager_integration(self, context):
        """Test integration with IntermediateStepManager."""
        manager = context.intermediate_step_manager
        assert isinstance(manager, IntermediateStepManager)

        # The manager should have access to the context state
        assert manager._context_state is context._context_state


# --------------------------------------------------------------------------- #
# Test Compatibility Aliases
# --------------------------------------------------------------------------- #


class TestCompatibilityAliases:
    """Test compatibility aliases with previous releases."""

    def test_aiq_context_state_alias(self):
        """Test that AIQContextState is an alias for ContextState."""
        assert AIQContextState is ContextState

    def test_aiq_context_alias(self):
        """Test that AIQContext is an alias for Context."""
        assert AIQContext is Context

    def test_aliases_work_as_expected(self, context_state):
        """Test that aliases work as expected in practice."""
        # Create instances using aliases
        aiq_context_state = AIQContextState()
        aiq_context = AIQContext(context_state)

        # Should be the same types as the original classes
        assert isinstance(aiq_context_state, ContextState)
        assert isinstance(aiq_context, Context)


# --------------------------------------------------------------------------- #
# Test Edge Cases and Error Conditions
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_context_with_none_context_state(self):
        """Test Context behavior with None context state."""
        # This should work but might cause issues when accessing properties
        context = Context(None)  # type: ignore

        # Accessing properties should raise AttributeError
        with pytest.raises(AttributeError):
            _ = context.input_message

    def test_active_span_id_with_empty_stack(self, context):
        """Test active_span_id behavior with empty stack."""
        # Manually set empty stack (this shouldn't happen in normal usage)
        context._context_state._active_span_id_stack.set([])

        # Should raise IndexError when trying to access last element
        with pytest.raises(IndexError):
            _ = context.active_span_id

    def test_context_state_singleton_thread_safety(self):
        """Test that ContextState singleton is thread-safe."""
        instances = []

        def create_instance():
            instances.append(ContextState.get())

        # Create instances from multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All instances should be the same
        first_instance = instances[0]
        for instance in instances[1:]:
            assert instance is first_instance


# --------------------------------------------------------------------------- #
# Performance and Stress Tests
# --------------------------------------------------------------------------- #


class TestPerformance:
    """Test performance characteristics of Context classes."""

    def test_context_creation_performance(self):
        """Test that Context creation is reasonably fast."""
        start_time = time.time()

        # Create many contexts
        contexts = []
        for _ in range(1000):
            contexts.append(Context.get())

        end_time = time.time()

        # Should complete in reasonable time (adjust threshold as needed)
        assert end_time - start_time < 1.0  # Less than 1 second

        # All contexts should reference the same ContextState (singleton)
        first_state = contexts[0]._context_state
        for context in contexts[1:]:
            assert context._context_state is first_state

    def test_nested_push_active_function_performance(self, context):
        """Test performance of nested push_active_function calls."""
        start_time = time.time()

        # Create nested function calls
        with context.push_active_function("func1", {}):
            with context.push_active_function("func2", {}):
                with context.push_active_function("func3", {}):
                    with context.push_active_function("func4", {}):
                        with context.push_active_function("func5", {}):
                            pass

        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 1.0  # Less than 1 second (relaxed for CI)
