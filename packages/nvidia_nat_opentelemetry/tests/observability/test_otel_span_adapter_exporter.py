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

import uuid
from datetime import datetime
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from nat.builder.context import ContextState
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.invocation_node import InvocationNode
from nat.plugins.opentelemetry.otel_span import OtelSpan
from nat.plugins.opentelemetry.otlp_span_adapter_exporter import OTLPSpanAdapterExporter


def create_test_intermediate_step(parent_id="root",
                                  function_name="test_function",
                                  function_id="test_id",
                                  **payload_kwargs):
    """Helper function to create IntermediateStep with proper structure for tests."""
    payload = IntermediateStepPayload(**payload_kwargs)
    function_ancestry = InvocationNode(function_name=function_name, function_id=function_id, parent_id=None)
    return IntermediateStep(parent_id=parent_id, function_ancestry=function_ancestry, payload=payload)


class TestOTLPSpanAdapterExporter:
    """Test suite for OTLPSpanAdapterExporter functionality."""

    @pytest.fixture
    def mock_context_state(self):
        """Create a mock AIQContextState for testing."""
        return Mock(spec=ContextState)

    @pytest.fixture
    def basic_exporter_config(self):
        """Basic configuration for the exporter."""
        return {
            "endpoint": "https://api.example.com/v1/traces",
            "headers": {
                "Authorization": "Bearer test-token"
            },
            "batch_size": 50,
            "flush_interval": 5.0
        }

    @pytest.fixture
    def sample_start_event(self):
        """Create a sample START event."""
        test_uuid = str(uuid.uuid4())
        return create_test_intermediate_step(parent_id="root",
                                             function_name="test_llm_call",
                                             function_id="func_123",
                                             event_type=IntermediateStepType.LLM_START,
                                             framework=LLMFrameworkEnum.LANGCHAIN,
                                             name="test_llm_call",
                                             event_timestamp=datetime.now().timestamp(),
                                             data=StreamEventData(input="Test input"),
                                             metadata={"key": "value"},
                                             UUID=test_uuid)

    @pytest.fixture
    def sample_end_event(self):
        """Create a sample END event."""
        test_uuid = str(uuid.uuid4())
        return create_test_intermediate_step(parent_id="root",
                                             function_name="test_llm_call",
                                             function_id="func_123",
                                             event_type=IntermediateStepType.LLM_END,
                                             framework=LLMFrameworkEnum.LANGCHAIN,
                                             name="test_llm_call",
                                             event_timestamp=datetime.now().timestamp(),
                                             data=StreamEventData(output="Test output"),
                                             metadata={"key": "value"},
                                             UUID=test_uuid)

    @pytest.fixture
    def mock_otel_span(self):
        """Create a mock OtelSpan for testing."""
        span = Mock(spec=OtelSpan)
        span.set_resource = Mock()
        return span

    def test_initialization_with_required_params(self, basic_exporter_config):
        """Test OTLPSpanAdapterExporter initialization with required parameters."""
        exporter = OTLPSpanAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                           headers=basic_exporter_config["headers"])

        assert exporter is not None
        assert hasattr(exporter, '_exporter')
        assert isinstance(exporter._exporter, OTLPSpanExporter)

    def test_initialization_with_all_params(self, mock_context_state, basic_exporter_config):
        """Test OTLPSpanAdapterExporter initialization with all parameters."""
        resource_attributes = {"service.name": "test-service", "service.version": "1.0"}

        exporter = OTLPSpanAdapterExporter(context_state=mock_context_state,
                                           endpoint=basic_exporter_config["endpoint"],
                                           headers=basic_exporter_config["headers"],
                                           batch_size=basic_exporter_config["batch_size"],
                                           flush_interval=basic_exporter_config["flush_interval"],
                                           max_queue_size=500,
                                           drop_on_overflow=True,
                                           shutdown_timeout=15.0,
                                           resource_attributes=resource_attributes)

        assert exporter is not None
        assert hasattr(exporter, '_exporter')
        assert isinstance(exporter._exporter, OTLPSpanExporter)
        assert exporter._resource.attributes["service.name"] == "test-service"
        assert exporter._resource.attributes["service.version"] == "1.0"

    def test_initialization_with_otlp_kwargs(self, basic_exporter_config):
        """Test OTLPSpanAdapterExporter initialization with core OTLP parameters only."""
        exporter = OTLPSpanAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                           headers=basic_exporter_config["headers"])

        assert exporter is not None
        assert isinstance(exporter._exporter, OTLPSpanExporter)

    def test_initialization_without_headers(self, basic_exporter_config):
        """Test OTLPSpanAdapterExporter initialization without headers."""
        exporter = OTLPSpanAdapterExporter(endpoint=basic_exporter_config["endpoint"])

        assert exporter is not None
        assert isinstance(exporter._exporter, OTLPSpanExporter)

    def test_initialization_with_empty_resource_attributes(self, basic_exporter_config):
        """Test OTLPSpanAdapterExporter initialization with empty resource attributes."""
        exporter = OTLPSpanAdapterExporter(endpoint=basic_exporter_config["endpoint"], resource_attributes={})

        assert exporter is not None
        assert exporter._resource.attributes == {}

    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporter')
    async def test_export_otel_spans_success(self, mock_otlp_exporter_class, basic_exporter_config, mock_otel_span):
        """Test successful export of OtelSpans."""
        # Setup mock
        mock_otlp_exporter = Mock()
        mock_otlp_exporter.export = Mock()
        mock_otlp_exporter_class.return_value = mock_otlp_exporter

        exporter = OTLPSpanAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                           headers=basic_exporter_config["headers"])

        spans = [mock_otel_span]

        # Test export
        await exporter.export_otel_spans(spans)

        # Verify the OTLP exporter was called
        mock_otlp_exporter.export.assert_called_once_with(spans)

    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporter')
    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.logger')
    async def test_export_otel_spans_with_exception(self,
                                                    mock_logger,
                                                    mock_otlp_exporter_class,
                                                    basic_exporter_config,
                                                    mock_otel_span):
        """Test export of OtelSpans with exception handling."""
        # Setup mock to raise exception
        mock_otlp_exporter = Mock()
        mock_otlp_exporter.export = Mock(side_effect=Exception("Network error"))
        mock_otlp_exporter_class.return_value = mock_otlp_exporter

        exporter = OTLPSpanAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                           headers=basic_exporter_config["headers"])

        spans = [mock_otel_span]

        # Test export - should not raise exception
        await exporter.export_otel_spans(spans)

        # Verify error was logged
        mock_logger.error.assert_called_once()
        assert "Error exporting spans" in str(mock_logger.error.call_args)

    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporter')
    async def test_export_multiple_spans(self, mock_otlp_exporter_class, basic_exporter_config):
        """Test export of multiple OtelSpans."""
        # Setup mock
        mock_otlp_exporter = Mock()
        mock_otlp_exporter.export = Mock()
        mock_otlp_exporter_class.return_value = mock_otlp_exporter

        exporter = OTLPSpanAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                           headers=basic_exporter_config["headers"])

        spans = [Mock(spec=OtelSpan) for _ in range(3)]
        for span in spans:
            span.set_resource = Mock()

        # Test export
        await exporter.export_otel_spans(spans)

        # Verify the OTLP exporter was called with all spans
        mock_otlp_exporter.export.assert_called_once_with(spans)

    async def test_end_to_end_span_processing(self, basic_exporter_config, sample_start_event, sample_end_event):
        """Test end-to-end span processing from IntermediateStep to export."""
        with patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporter') \
                as mock_otlp_exporter_class:
            # Setup mock
            mock_otlp_exporter = Mock()
            mock_otlp_exporter.export = Mock()
            mock_otlp_exporter_class.return_value = mock_otlp_exporter

            exporter = OTLPSpanAdapterExporter(
                endpoint=basic_exporter_config["endpoint"],
                headers=basic_exporter_config["headers"],
                batch_size=1,  # Force immediate processing
                flush_interval=0.1)

            # Use same UUID for start and end events to create a complete span
            sample_end_event.payload.UUID = sample_start_event.payload.UUID

            async with exporter.start():
                # Process start event
                exporter.export(sample_start_event)

                # Process end event
                exporter.export(sample_end_event)

                # Wait for async processing
                await exporter._wait_for_tasks()

            # Verify that export was called (span was processed and exported)
            mock_otlp_exporter.export.assert_called()

            # Verify the exported spans have the correct structure
            call_args = mock_otlp_exporter.export.call_args
            exported_spans = call_args[0][0]  # First positional argument
            assert len(exported_spans) >= 1
            assert all(hasattr(span, 'set_resource') for span in exported_spans)

    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporter')
    async def test_batching_behavior(self, mock_otlp_exporter_class, basic_exporter_config):
        """Test that batching works correctly with the OTLP exporter."""
        # Setup mock
        mock_otlp_exporter = Mock()
        mock_otlp_exporter.export = Mock()
        mock_otlp_exporter_class.return_value = mock_otlp_exporter

        batch_size = 3
        exporter = OTLPSpanAdapterExporter(
            endpoint=basic_exporter_config["endpoint"],
            headers=basic_exporter_config["headers"],
            batch_size=batch_size,
            flush_interval=10.0  # Long interval to test batching
        )

        async with exporter.start():
            # Create multiple complete spans (start + end events)
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

        # Verify that export was called (batching should trigger export)
        mock_otlp_exporter.export.assert_called()

    def test_inheritance_structure(self, basic_exporter_config):
        """Test that OTLPSpanAdapterExporter has the correct inheritance structure."""
        from nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin import OTLPSpanExporterMixin
        from nat.plugins.opentelemetry.otel_span_exporter import OtelSpanExporter

        exporter = OTLPSpanAdapterExporter(endpoint=basic_exporter_config["endpoint"])

        assert isinstance(exporter, OTLPSpanExporterMixin)
        assert isinstance(exporter, OtelSpanExporter)
        assert hasattr(exporter, 'export_otel_spans')
        assert hasattr(exporter, 'export_processed')

    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporter')
    def test_otlp_exporter_initialization_with_headers(self, mock_otlp_exporter_class, basic_exporter_config):
        """Test that the internal OTLP exporter is initialized with correct headers."""
        headers = basic_exporter_config["headers"]
        endpoint = basic_exporter_config["endpoint"]

        OTLPSpanAdapterExporter(endpoint=endpoint, headers=headers)

        # Verify OTLPSpanExporter was initialized with correct parameters
        mock_otlp_exporter_class.assert_called_once_with(endpoint=endpoint, headers=headers)

    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporter')
    def test_otlp_exporter_initialization_without_headers(self, mock_otlp_exporter_class, basic_exporter_config):
        """Test that the internal OTLP exporter is initialized correctly without headers."""
        endpoint = basic_exporter_config["endpoint"]

        OTLPSpanAdapterExporter(endpoint=endpoint)

        # Verify OTLPSpanExporter was initialized with correct parameters
        mock_otlp_exporter_class.assert_called_once_with(endpoint=endpoint, headers=None)

    def test_missing_endpoint_parameter(self):
        """Test that missing endpoint parameter raises appropriate error."""
        with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'endpoint'"):
            OTLPSpanAdapterExporter()  # pylint: disable=missing-kwoa # type: ignore[call-arg]

    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporter')
    async def test_resource_attributes_applied_to_spans(self,
                                                        mock_otlp_exporter_class,
                                                        basic_exporter_config,
                                                        mock_otel_span):
        """Test that resource attributes are properly applied to spans before export."""
        # Setup mock
        mock_otlp_exporter = Mock()
        mock_otlp_exporter.export = Mock()
        mock_otlp_exporter_class.return_value = mock_otlp_exporter

        resource_attributes = {"service.name": "test-service"}
        exporter = OTLPSpanAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                           resource_attributes=resource_attributes)

        # Test export_processed method (which sets resource attributes)
        await exporter.export_processed(mock_otel_span)

        # Verify resource was set on the span
        mock_otel_span.set_resource.assert_called_once_with(exporter._resource)

        # Verify export was called
        mock_otlp_exporter.export.assert_called_once()
