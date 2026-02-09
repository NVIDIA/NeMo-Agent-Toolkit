# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

"""Integration tests for A365 telemetry exporter with mocked A365 SDK."""

import uuid
from unittest.mock import Mock, patch

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags

from nat.builder.context import ContextState
from nat.plugins.a365.telemetry.a365_exporter import A365OtelExporter, _ReadableSpanAdapter
from nat.plugins.opentelemetry.otel_span import OtelSpan
from nat.plugins.opentelemetry.otel_span import InstrumentationScope


def create_mock_otel_span(
    name: str = "test_span",
    trace_id: int | None = None,
    span_id: int | None = None,
    parent: OtelSpan | None = None,
    attributes: dict | None = None,
    events: list | None = None,
    links: list | None = None,
) -> OtelSpan:
    """Create a mock OtelSpan for testing."""
    if trace_id is None:
        trace_id = uuid.uuid4().int
    if span_id is None:
        span_id = uuid.uuid4().int >> 64

    context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(1),  # SAMPLED
    )

    span = OtelSpan(
        name=name,
        context=context,
        parent=parent,
        attributes=attributes or {},
        events=events or [],
        links=links or [],
        kind=SpanKind.INTERNAL,
        resource=Resource.create({"service.name": "test_service"}),
        instrumentation_scope=InstrumentationScope("test", "1.0.0"),
    )

    return span


@pytest.fixture
def mock_context_state():
    """Create a mock ContextState for testing."""
    return Mock(spec=ContextState)


@pytest.fixture
def mock_token_resolver():
    """Create a mock token resolver function."""
    def resolver(agent_id: str, tenant_id: str) -> str:
        return f"mock_token_for_{agent_id}_{tenant_id}"

    return resolver


@pytest.fixture
def a365_exporter(mock_context_state, mock_token_resolver):
    """Create an A365OtelExporter instance for testing."""
    # Patch the A365 exporter where it's imported in our module
    with patch(
        "nat.plugins.a365.telemetry.a365_exporter.Agent365Exporter"
    ) as mock_exporter_class:
        # Mock the A365 exporter class
        mock_exporter_instance = Mock()
        mock_exporter_instance.export = Mock(return_value=SpanExportResult.SUCCESS)
        mock_exporter_class.return_value = mock_exporter_instance

        exporter = A365OtelExporter(
            agent_id="test-agent-123",
            tenant_id="test-tenant-456",
            token_resolver=mock_token_resolver,
            cluster_category="prod",
            use_s2s_endpoint=False,
            suppress_invoke_agent_input=False,
            context_state=mock_context_state,
        )

        # Store reference to the mock instance
        exporter._mock_a365_exporter_instance = mock_exporter_instance

        yield exporter


class TestA365ExporterIntegration:
    """Integration tests for A365OtelExporter."""

    @pytest.mark.asyncio
    async def test_export_empty_spans(self, a365_exporter):
        """Test that exporting empty spans list does nothing."""
        await a365_exporter.export_otel_spans([])

        # Should not call the A365 exporter
        a365_exporter._mock_a365_exporter_instance.export.assert_not_called()

    @pytest.mark.asyncio
    async def test_export_single_span(self, a365_exporter):
        """Test exporting a single span."""
        span = create_mock_otel_span(name="test_span_1")

        await a365_exporter.export_otel_spans([span])

        # Verify A365 exporter was called
        a365_exporter._mock_a365_exporter_instance.export.assert_called_once()

        # Get the call arguments
        call_args = a365_exporter._mock_a365_exporter_instance.export.call_args
        readable_spans = call_args[0][0]  # First positional argument

        # Verify we got a list of ReadableSpan-like objects
        assert len(readable_spans) == 1
        readable_span = readable_spans[0]

        # Verify it's a _ReadableSpanAdapter
        assert isinstance(readable_span, _ReadableSpanAdapter)

        # Verify span attributes were set correctly
        assert readable_span.name == "test_span_1"
        assert readable_span.attributes["tenant.id"] == "test-tenant-456"
        assert readable_span.attributes["gen_ai.agent.id"] == "test-agent-123"
        assert readable_span.context.trace_id == span.get_span_context().trace_id
        assert readable_span.context.span_id == span.get_span_context().span_id

    @pytest.mark.asyncio
    async def test_export_multiple_spans(self, a365_exporter):
        """Test exporting multiple spans."""
        span1 = create_mock_otel_span(name="span_1")
        span2 = create_mock_otel_span(name="span_2")
        span3 = create_mock_otel_span(name="span_3")

        await a365_exporter.export_otel_spans([span1, span2, span3])

        # Verify A365 exporter was called once with all spans
        a365_exporter._mock_a365_exporter_instance.export.assert_called_once()

        call_args = a365_exporter._mock_a365_exporter_instance.export.call_args
        readable_spans = call_args[0][0]

        assert len(readable_spans) == 3
        assert all(isinstance(rs, _ReadableSpanAdapter) for rs in readable_spans)
        assert readable_spans[0].name == "span_1"
        assert readable_spans[1].name == "span_2"
        assert readable_spans[2].name == "span_3"

    @pytest.mark.asyncio
    async def test_span_conversion_with_attributes(self, a365_exporter):
        """Test that span attributes are preserved and tenant/agent IDs are added."""
        span = create_mock_otel_span(
            name="test_span",
            attributes={"custom.attr": "value", "another.attr": 42}
        )

        await a365_exporter.export_otel_spans([span])

        call_args = a365_exporter._mock_a365_exporter_instance.export.call_args
        readable_span = call_args[0][0][0]

        # Verify original attributes are preserved
        assert readable_span.attributes["custom.attr"] == "value"
        assert readable_span.attributes["another.attr"] == 42

        # Verify A365-specific attributes are added
        assert readable_span.attributes["tenant.id"] == "test-tenant-456"
        assert readable_span.attributes["gen_ai.agent.id"] == "test-agent-123"

    @pytest.mark.asyncio
    async def test_span_conversion_with_events(self, a365_exporter):
        """Test that span events are converted correctly."""
        from opentelemetry.sdk.trace import Event as OtelEvent

        events = [
            OtelEvent(name="event1", timestamp=1234567890, attributes={"key": "value"}),
            OtelEvent(name="event2", timestamp=1234567891, attributes={}),
        ]

        span = create_mock_otel_span(name="test_span", events=events)

        await a365_exporter.export_otel_spans([span])

        call_args = a365_exporter._mock_a365_exporter_instance.export.call_args
        readable_span = call_args[0][0][0]

        assert len(readable_span.events) == 2
        assert readable_span.events[0].name == "event1"
        assert readable_span.events[1].name == "event2"

    @pytest.mark.asyncio
    async def test_span_conversion_with_parent(self, a365_exporter):
        """Test that parent span context is converted correctly."""
        parent_span = create_mock_otel_span(name="parent_span")
        child_span = create_mock_otel_span(name="child_span", parent=parent_span)

        await a365_exporter.export_otel_spans([child_span])

        call_args = a365_exporter._mock_a365_exporter_instance.export.call_args
        readable_span = call_args[0][0][0]

        # Parent should be converted to SpanContext
        assert readable_span.parent is not None
        assert readable_span.parent.trace_id == parent_span.get_span_context().trace_id
        assert readable_span.parent.span_id == parent_span.get_span_context().span_id

    @pytest.mark.asyncio
    async def test_export_error_handling(self, a365_exporter):
        """Test that export errors are caught and logged without re-raising."""
        span = create_mock_otel_span(name="test_span")

        # Make the A365 exporter raise an exception
        a365_exporter._mock_a365_exporter_instance.export.side_effect = Exception("Export failed")

        # Should not raise, but log the error
        with patch("nat.plugins.a365.telemetry.a365_exporter.logger") as mock_logger:
            await a365_exporter.export_otel_spans([span])

            # Verify error was logged
            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args
            assert "Error exporting spans to A365" in error_call[0][0]
            assert "Export failed" in str(error_call)

    @pytest.mark.asyncio
    async def test_exporter_initialization_with_config(self, mock_context_state, mock_token_resolver):
        """Test that exporter is initialized with correct A365 SDK configuration."""
        with patch(
            "nat.plugins.a365.telemetry.a365_exporter.Agent365Exporter"
        ) as mock_exporter_class:
            mock_exporter_instance = Mock()
            mock_exporter_class.return_value = mock_exporter_instance

            exporter = A365OtelExporter(
                agent_id="custom-agent",
                tenant_id="custom-tenant",
                token_resolver=mock_token_resolver,
                cluster_category="dev",
                use_s2s_endpoint=True,
                suppress_invoke_agent_input=True,
                context_state=mock_context_state,
            )

            # Verify A365 exporter was created with correct parameters
            mock_exporter_class.assert_called_once_with(
                token_resolver=mock_token_resolver,
                cluster_category="dev",
                use_s2s_endpoint=True,
            )

            assert exporter._agent_id == "custom-agent"
            assert exporter._tenant_id == "custom-tenant"

    @pytest.mark.asyncio
    async def test_async_executor_bridge(self, a365_exporter):
        """Test that async/sync bridge works correctly."""
        span = create_mock_otel_span(name="test_span")

        # Verify the export method is synchronous (no async/await in the mock)
        await a365_exporter.export_otel_spans([span])

        # The call should have been made via run_in_executor
        # We can verify this by checking that export was called
        a365_exporter._mock_a365_exporter_instance.export.assert_called_once()

        # Verify it was called with the correct argument type
        call_args = a365_exporter._mock_a365_exporter_instance.export.call_args
        readable_spans = call_args[0][0]
        assert isinstance(readable_spans, list)
        assert len(readable_spans) == 1
