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
"""
Integration tests for OTLPSpanAdapterExporter that validate actual export behavior.

These tests complement the unit tests by validating real export functionality
without mocking the underlying OTLP exporter.
"""

import asyncio
import uuid
from datetime import datetime

import pytest
import pytest_httpserver
from werkzeug import Request
from werkzeug import Response

from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepType
from aiq.data_models.intermediate_step import StreamEventData
from aiq.data_models.invocation_node import InvocationNode
from aiq.plugins.opentelemetry.otlp_span_adapter_exporter import OTLPSpanAdapterExporter


def create_test_intermediate_step(parent_id="root",
                                  function_name="test_function",
                                  function_id="test_id",
                                  **payload_kwargs):
    """Helper function to create IntermediateStep with proper structure for tests."""
    payload = IntermediateStepPayload(**payload_kwargs)
    function_ancestry = InvocationNode(function_name=function_name, function_id=function_id, parent_id=None)
    return IntermediateStep(parent_id=parent_id, function_ancestry=function_ancestry, payload=payload)


class TestOTLPSpanAdapterExporterIntegration:
    """Integration tests that validate actual span export behavior."""

    @pytest.fixture
    def mock_otlp_server(self):
        """Create a mock OTLP HTTP server to receive exported spans."""
        server = pytest_httpserver.HTTPServer(host="127.0.0.1", port=0)
        server.start()

        # Track received requests
        server.received_spans = []
        server.received_headers = []

        def trace_handler(request: Request):
            """Handle OTLP trace requests."""
            # Store received data for validation
            server.received_spans.append(request.data)
            server.received_headers.append(dict(request.headers))

            # Return success response
            return Response(status=200, response="{}")

        server.expect_request("/v1/traces", method="POST").respond_with_handler(trace_handler)

        yield server
        server.stop()

    @pytest.fixture
    def sample_events(self):
        """Create sample start and end events for testing."""
        test_uuid = str(uuid.uuid4())

        start_event = create_test_intermediate_step(parent_id="root",
                                                    function_name="test_llm_call",
                                                    function_id="func_123",
                                                    event_type=IntermediateStepType.LLM_START,
                                                    framework=LLMFrameworkEnum.LANGCHAIN,
                                                    name="test_llm_call",
                                                    event_timestamp=datetime.now().timestamp(),
                                                    data=StreamEventData(input="Test input"),
                                                    metadata={"key": "value"},
                                                    UUID=test_uuid)

        end_event = create_test_intermediate_step(parent_id="root",
                                                  function_name="test_llm_call",
                                                  function_id="func_123",
                                                  event_type=IntermediateStepType.LLM_END,
                                                  framework=LLMFrameworkEnum.LANGCHAIN,
                                                  name="test_llm_call",
                                                  event_timestamp=datetime.now().timestamp(),
                                                  data=StreamEventData(output="Test output"),
                                                  metadata={"key": "value"},
                                                  UUID=test_uuid)

        return start_event, end_event

    async def test_actual_span_export_to_mock_server(self, mock_otlp_server, sample_events):
        """Test that spans are actually exported to a real HTTP endpoint."""
        start_event, end_event = sample_events

        # Create exporter pointing to mock server
        endpoint = f"http://127.0.0.1:{mock_otlp_server.port}/v1/traces"
        headers = {"Authorization": "Bearer test-token", "Custom-Header": "test-value"}

        exporter = OTLPSpanAdapterExporter(
            endpoint=endpoint,
            headers=headers,
            batch_size=1,  # Force immediate export
            flush_interval=0.1,
            resource_attributes={"service.name": "test-service"})

        async with exporter.start():
            # Process events to create and export spans
            exporter.export(start_event)
            exporter.export(end_event)

            # Wait for async export to complete
            await exporter._wait_for_tasks()

            # Give a small buffer for HTTP request to complete
            await asyncio.sleep(0.1)

        # Validate that actual HTTP request was received
        assert len(mock_otlp_server.received_spans) >= 1, "No spans were exported to the server"

        # Validate request headers were passed correctly
        received_headers = mock_otlp_server.received_headers[0]
        assert received_headers.get("Authorization") == "Bearer test-token"
        assert received_headers.get("Custom-Header") == "test-value"
        assert received_headers.get("Content-Type") == "application/x-protobuf"

        # Validate that span data was sent (protobuf format)
        span_data = mock_otlp_server.received_spans[0]
        assert len(span_data) > 0, "Exported span data is empty"
        assert isinstance(span_data, bytes), "Span data should be protobuf bytes"

    async def test_export_error_handling_with_real_endpoint(self, sample_events):
        """Test error handling when exporting to an unreachable endpoint."""
        start_event, end_event = sample_events

        # Create exporter with unreachable endpoint
        exporter = OTLPSpanAdapterExporter(
            endpoint="http://127.0.0.1:99999/v1/traces",  # Unreachable port
            batch_size=1,
            flush_interval=0.1)

        async with exporter.start():
            exporter.export(start_event)
            exporter.export(end_event)

            # Wait for export attempt (should fail but not crash)
            await exporter._wait_for_tasks()
            await asyncio.sleep(0.1)

        # Test passes if no exception was raised - error should be logged internally

    async def test_span_batching_with_real_export(self, mock_otlp_server):
        """Test that span batching works with actual HTTP export."""
        batch_size = 3

        # Create exporter with batching
        endpoint = f"http://127.0.0.1:{mock_otlp_server.port}/v1/traces"
        exporter = OTLPSpanAdapterExporter(
            endpoint=endpoint,
            batch_size=batch_size,
            flush_interval=10.0  # Long interval to test batching trigger
        )

        async with exporter.start():
            # Create multiple spans to trigger batch export
            for i in range(batch_size):
                start_event = create_test_intermediate_step(parent_id="root",
                                                            function_name=f"test_function_{i}",
                                                            function_id=f"func_{i}",
                                                            event_type=IntermediateStepType.LLM_START,
                                                            framework=LLMFrameworkEnum.LANGCHAIN,
                                                            name=f"test_call_{i}",
                                                            event_timestamp=datetime.now().timestamp(),
                                                            data=StreamEventData(input=f"Input {i}"),
                                                            UUID=f"uuid_{i}")

                end_event = create_test_intermediate_step(parent_id="root",
                                                          function_name=f"test_function_{i}",
                                                          function_id=f"func_{i}",
                                                          event_type=IntermediateStepType.LLM_END,
                                                          framework=LLMFrameworkEnum.LANGCHAIN,
                                                          name=f"test_call_{i}",
                                                          event_timestamp=datetime.now().timestamp(),
                                                          data=StreamEventData(output=f"Output {i}"),
                                                          UUID=f"uuid_{i}")

                exporter.export(start_event)
                exporter.export(end_event)

            # Wait for batch processing
            await exporter._wait_for_tasks()
            await asyncio.sleep(0.1)

        # Validate that batch export occurred
        assert len(mock_otlp_server.received_spans) >= 1, "Batch export did not occur"

    async def test_basic_export_functionality(self, mock_otlp_server, sample_events):
        """Test basic OTLP export functionality."""
        start_event, end_event = sample_events

        # Create exporter with basic configuration
        endpoint = f"http://127.0.0.1:{mock_otlp_server.port}/v1/traces"
        exporter = OTLPSpanAdapterExporter(endpoint=endpoint, batch_size=1)

        async with exporter.start():
            exporter.export(start_event)
            exporter.export(end_event)
            await exporter._wait_for_tasks()
            await asyncio.sleep(0.1)

        # Validate that spans were exported
        assert len(mock_otlp_server.received_spans) >= 1
        received_headers = mock_otlp_server.received_headers[0]
        assert received_headers.get("Content-Type") == "application/x-protobuf"
