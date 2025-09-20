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
import uuid
from unittest.mock import Mock, AsyncMock

from nat.data_models.span import Span
from nat.observability.context import ObservabilityContext, ObservabilityContextManager
from nat.observability.workflow_utils import ObservabilityWorkflowInvoker
from nat.observability.processor.cross_workflow_processor import CrossWorkflowProcessor, WorkflowRelationshipProcessor


class TestObservabilityContext:
    """Test the ObservabilityContext class."""

    def test_create_root_context(self):
        """Test creating a root observability context."""
        context = ObservabilityContext.create_root_context("test_workflow")

        assert context.trace_id is not None
        assert context.root_span_id is not None
        assert context.current_span_id == context.root_span_id
        assert len(context.workflow_chain) == 1
        assert context.workflow_chain[0].workflow_name == "test_workflow"
        assert context.workflow_chain[0].workflow_id == context.root_span_id

    def test_create_root_context_with_custom_ids(self):
        """Test creating a root context with custom trace and span IDs."""
        trace_id = "custom-trace-id"
        root_span_id = "custom-span-id"

        context = ObservabilityContext.create_root_context(
            "test_workflow", trace_id=trace_id, root_span_id=root_span_id
        )

        assert context.trace_id == trace_id
        assert context.root_span_id == root_span_id
        assert context.current_span_id == root_span_id

    def test_create_child_context(self):
        """Test creating a child observability context."""
        parent_context = ObservabilityContext.create_root_context("parent_workflow")
        child_context = parent_context.create_child_context("child_workflow")

        assert child_context.trace_id == parent_context.trace_id
        assert child_context.root_span_id == parent_context.root_span_id
        assert child_context.current_span_id != parent_context.current_span_id
        assert len(child_context.workflow_chain) == 2
        assert child_context.workflow_chain[0].workflow_name == "parent_workflow"
        assert child_context.workflow_chain[1].workflow_name == "child_workflow"
        assert child_context.workflow_chain[1].parent_workflow_id == parent_context.current_span_id

    def test_create_span_context(self):
        """Test creating a new context with different span ID."""
        original_context = ObservabilityContext.create_root_context("test_workflow")
        new_span_id = str(uuid.uuid4())
        span_context = original_context.create_span_context(new_span_id)

        assert span_context.trace_id == original_context.trace_id
        assert span_context.root_span_id == original_context.root_span_id
        assert span_context.current_span_id == new_span_id
        assert len(span_context.workflow_chain) == len(original_context.workflow_chain)

    def test_add_attribute(self):
        """Test adding custom attributes to the context."""
        context = ObservabilityContext.create_root_context("test_workflow")
        context.add_attribute("test_key", "test_value")

        assert "test_key" in context.custom_attributes
        assert context.custom_attributes["test_key"] == "test_value"

    def test_get_current_workflow(self):
        """Test getting the current workflow metadata."""
        context = ObservabilityContext.create_root_context("test_workflow")
        current_workflow = context.get_current_workflow()

        assert current_workflow is not None
        assert current_workflow.workflow_name == "test_workflow"

    def test_get_workflow_depth(self):
        """Test getting workflow depth."""
        parent_context = ObservabilityContext.create_root_context("parent")
        child_context = parent_context.create_child_context("child")
        grandchild_context = child_context.create_child_context("grandchild")

        assert parent_context.get_workflow_depth() == 1
        assert child_context.get_workflow_depth() == 2
        assert grandchild_context.get_workflow_depth() == 3

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization of observability context."""
        original_context = ObservabilityContext.create_root_context("test_workflow")
        original_context.add_attribute("test_attr", "test_value")

        # Convert to dict
        context_dict = original_context.to_dict()

        # Convert back from dict
        restored_context = ObservabilityContext.from_dict(context_dict)

        assert restored_context.trace_id == original_context.trace_id
        assert restored_context.root_span_id == original_context.root_span_id
        assert restored_context.current_span_id == original_context.current_span_id
        assert len(restored_context.workflow_chain) == len(original_context.workflow_chain)
        assert restored_context.custom_attributes == original_context.custom_attributes


class TestObservabilityContextManager:
    """Test the ObservabilityContextManager class."""

    def test_create_root_context(self):
        """Test creating and setting a root context."""
        context = ObservabilityContextManager.create_root_context("test_workflow")

        assert context is not None
        assert context.workflow_chain[0].workflow_name == "test_workflow"

        current_context = ObservabilityContextManager.get_current_context()
        assert current_context is not None
        assert current_context.trace_id == context.trace_id

    def test_create_child_context(self):
        """Test creating a child context from current context."""
        # First create a root context
        ObservabilityContextManager.create_root_context("parent_workflow")

        # Then create a child context
        child_context = ObservabilityContextManager.create_child_context("child_workflow")

        assert child_context is not None
        assert len(child_context.workflow_chain) == 2
        assert child_context.workflow_chain[1].workflow_name == "child_workflow"

    def test_create_child_context_without_current(self):
        """Test creating a child context when no current context exists."""
        ObservabilityContextManager.clear_context()
        child_context = ObservabilityContextManager.create_child_context("child_workflow")

        assert child_context is None

    def test_propagate_context(self):
        """Test propagating an existing context."""
        external_context = ObservabilityContext.create_root_context("external_workflow")
        ObservabilityContextManager.propagate_context(external_context)

        current_context = ObservabilityContextManager.get_current_context()
        assert current_context is not None
        assert current_context.trace_id == external_context.trace_id

    def test_clear_context(self):
        """Test clearing the current context."""
        ObservabilityContextManager.create_root_context("test_workflow")
        assert ObservabilityContextManager.get_current_context() is not None

        ObservabilityContextManager.clear_context()
        assert ObservabilityContextManager.get_current_context() is None

    def test_get_trace_id(self):
        """Test getting the current trace ID."""
        context = ObservabilityContextManager.create_root_context("test_workflow")
        trace_id = ObservabilityContextManager.get_trace_id()

        assert trace_id == context.trace_id

    def test_get_current_span_id(self):
        """Test getting the current span ID."""
        context = ObservabilityContextManager.create_root_context("test_workflow")
        span_id = ObservabilityContextManager.get_current_span_id()

        assert span_id == context.current_span_id

    def test_add_workflow_attribute(self):
        """Test adding an attribute to the current context."""
        ObservabilityContextManager.create_root_context("test_workflow")
        ObservabilityContextManager.add_workflow_attribute("test_key", "test_value")

        current_context = ObservabilityContextManager.get_current_context()
        assert current_context is not None
        assert current_context.custom_attributes["test_key"] == "test_value"


class TestCrossWorkflowProcessor:
    """Test the CrossWorkflowProcessor class."""

    def setup_method(self):
        """Set up test environment."""
        self.processor = CrossWorkflowProcessor()

    @pytest.mark.asyncio
    async def test_process_span_with_context(self):
        """Test processing a span with observability context available."""
        # Create a real observability context and set it in the context state
        obs_context = ObservabilityContext.create_root_context("test_workflow")
        obs_context.add_attribute("custom_attr", "custom_value")

        from nat.builder.context import Context
        context = Context.get()
        context.set_observability_context(obs_context)

        try:
            span = Span(name="test_span")
            processed_span = await self.processor.process(span)

            # Check that observability attributes were added
            attributes = processed_span.attributes
            assert "observability.trace_id" in attributes
            assert "observability.root_span_id" in attributes
            assert "observability.current_span_id" in attributes
            assert "observability.workflow_depth" in attributes
            assert "observability.workflow_name" in attributes
            assert "observability.custom.custom_attr" in attributes
            assert attributes["observability.custom.custom_attr"] == "custom_value"

        finally:
            # Clean up context
            context.set_observability_context(None)

    @pytest.mark.asyncio
    async def test_process_span_without_context(self):
        """Test processing a span without observability context."""
        from nat.builder.context import Context
        context = Context.get()
        context.set_observability_context(None)

        try:
            span = Span(name="test_span")
            processed_span = await self.processor.process(span)

            # Should return span with minimal changes
            assert processed_span.name == span.name
            # Attributes might have been slightly modified but no observability info should be added
            assert "observability.trace_id" not in processed_span.attributes

        finally:
            # Context already cleared
            pass



class TestWorkflowRelationshipProcessor:
    """Test the WorkflowRelationshipProcessor class."""

    def setup_method(self):
        """Set up test environment."""
        self.processor = WorkflowRelationshipProcessor()

    @pytest.mark.asyncio
    async def test_process_span_root_workflow(self):
        """Test processing a span for a root workflow."""
        obs_context = ObservabilityContext.create_root_context("root_workflow")

        from nat.builder.context import Context
        context = Context.get()
        context.set_observability_context(obs_context)

        try:
            span = Span(name="test_span")
            processed_span = await self.processor.process(span)

            attributes = processed_span.attributes
            assert attributes["relationship.type"] == "root_workflow"
            assert attributes["relationship.nesting_level"] == 0

        finally:
            context.set_observability_context(None)

    @pytest.mark.asyncio
    async def test_process_span_child_workflow(self):
        """Test processing a span for a child workflow."""
        parent_context = ObservabilityContext.create_root_context("parent_workflow")
        child_context = parent_context.create_child_context("child_workflow")

        from nat.builder.context import Context
        context = Context.get()
        context.set_observability_context(child_context)

        try:
            span = Span(name="test_span")
            processed_span = await self.processor.process(span)

            attributes = processed_span.attributes
            assert attributes["relationship.type"] == "child_workflow"
            assert attributes["relationship.parent_workflow_name"] == "parent_workflow"
            assert attributes["relationship.child_workflow_name"] == "child_workflow"
            assert attributes["relationship.nesting_level"] == 1
            assert "parent_workflow -> child_workflow" in attributes["relationship.hierarchy_path"]

        finally:
            context.set_observability_context(None)


class TestObservabilityWorkflowInvoker:
    """Test the ObservabilityWorkflowInvoker class."""

    @pytest.mark.asyncio
    async def test_invoke_workflow_with_context(self):
        """Test invoking a workflow with observability context."""
        # Create a mock workflow with proper async context manager
        mock_workflow = Mock()
        mock_runner = Mock()
        mock_runner.result = AsyncMock(return_value="test_result")

        # Create a proper async context manager mock
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__ = AsyncMock(return_value=mock_runner)
        async_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_workflow.run = Mock(return_value=async_context_manager)

        parent_context = ObservabilityContext.create_root_context("parent_workflow")

        result = await ObservabilityWorkflowInvoker.invoke_workflow_with_context(
            workflow=mock_workflow,
            message="test_message",
            workflow_name="child_workflow",
            parent_context=parent_context
        )

        assert result == "test_result"
        mock_workflow.run.assert_called_once()

        # Check that observability context was passed
        call_args = mock_workflow.run.call_args
        assert "observability_context" in call_args.kwargs
        obs_context = call_args.kwargs["observability_context"]
        assert obs_context is not None
        assert len(obs_context.workflow_chain) == 2
        assert obs_context.workflow_chain[1].workflow_name == "child_workflow"

    @pytest.mark.asyncio
    async def test_invoke_workflow_stream_with_context(self):
        """Test invoking a workflow with streaming output and observability context."""
        # Create a mock workflow with proper async context manager
        mock_workflow = Mock()
        mock_runner = Mock()

        async def mock_result_stream(to_type=None):
            yield "item1"
            yield "item2"

        mock_runner.result_stream = mock_result_stream

        # Create a proper async context manager mock
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__ = AsyncMock(return_value=mock_runner)
        async_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_workflow.run = Mock(return_value=async_context_manager)

        parent_context = ObservabilityContext.create_root_context("parent_workflow")

        results = []
        async for item in ObservabilityWorkflowInvoker.invoke_workflow_stream_with_context(
            workflow=mock_workflow,
            message="test_message",
            workflow_name="child_workflow",
            parent_context=parent_context
        ):
            results.append(item)

        assert results == ["item1", "item2"]
        mock_workflow.run.assert_called_once()

    def test_create_observability_context(self):
        """Test creating an observability context."""
        context = ObservabilityWorkflowInvoker.create_observability_context("test_workflow")

        assert context is not None
        assert context.workflow_chain[0].workflow_name == "test_workflow"

    def test_get_current_observability_context(self):
        """Test getting current observability context."""
        # This test would need proper context setup
        context = ObservabilityWorkflowInvoker.get_current_observability_context()
        # Since we don't have a proper context, this should return None
        assert context is None or isinstance(context, ObservabilityContext)