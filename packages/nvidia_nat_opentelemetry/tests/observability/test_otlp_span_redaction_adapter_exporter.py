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
from starlette.datastructures import Headers

from nat.builder.context import ContextState
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.span import Span
from nat.observability.mixin.tagging_config_mixin import PrivacyLevel
from nat.observability.processor.header_redaction_processor import HeaderRedactionProcessor
from nat.observability.processor.span_tagging_processor import SpanTaggingProcessor
from nat.plugins.opentelemetry import OTLPSpanAdapterExporter
from nat.plugins.opentelemetry import OTLPSpanHeaderRedactionAdapterExporter
from nat.plugins.opentelemetry.otel_span import OtelSpan


def create_test_intermediate_step(parent_id="root",
                                  function_name="test_function",
                                  function_id="test_id",
                                  **payload_kwargs):
    """Helper function to create IntermediateStep with proper structure for tests."""
    payload = IntermediateStepPayload(**payload_kwargs)
    function_ancestry = InvocationNode(function_name=function_name, function_id=function_id, parent_id=None)
    return IntermediateStep(parent_id=parent_id, function_ancestry=function_ancestry, payload=payload)


class TestOTLPSpanHeaderRedactionAdapterExporterInitialization:
    """Test suite for OTLPSpanHeaderRedactionAdapterExporter initialization."""

    @pytest.fixture
    def mock_context_state(self):
        """Create a mock ContextState for testing."""
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
    def sample_redaction_callback(self):
        """Sample redaction callback for testing."""

        def should_redact(auth_key: str) -> bool:
            return auth_key in ["sensitive_user", "test_user", "admin"]

        return should_redact

    def test_initialization_with_minimal_params(self, basic_exporter_config):
        """Test initialization with only required parameters."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"])

        assert exporter is not None
        assert hasattr(exporter, '_exporter')
        assert isinstance(exporter._exporter, OTLPSpanExporter)
        assert isinstance(exporter, OTLPSpanAdapterExporter)

    def test_initialization_with_redaction_params(self, basic_exporter_config, sample_redaction_callback):
        """Test initialization with redaction parameters."""
        redaction_attributes = ["user.email", "user.ssn", "request.body"]

        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          headers=basic_exporter_config["headers"],
                                                          redaction_attributes=redaction_attributes,
                                                          redaction_header="x-user-id",
                                                          redaction_callback=sample_redaction_callback,
                                                          redaction_enabled=True,
                                                          force_redaction=False)

        assert exporter is not None
        # Verify that the redaction processor was added
        # The processor should be at position 0
        # Since the processors are private, we test behavior later

    def test_initialization_with_privacy_tagging_params(self, basic_exporter_config):
        """Test initialization with privacy tagging parameters."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          privacy_tag_key="privacy.level",
                                                          privacy_tag_value=PrivacyLevel.HIGH)

        assert exporter is not None
        assert isinstance(exporter, OTLPSpanAdapterExporter)

    def test_initialization_with_all_privacy_levels(self, basic_exporter_config):
        """Test initialization with different privacy levels."""
        privacy_levels = [PrivacyLevel.NONE, PrivacyLevel.LOW, PrivacyLevel.MEDIUM, PrivacyLevel.HIGH]

        for privacy_level in privacy_levels:
            exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                              privacy_tag_key="privacy.level",
                                                              privacy_tag_value=privacy_level)
            assert exporter is not None

    def test_initialization_with_all_parameters(self,
                                                mock_context_state,
                                                basic_exporter_config,
                                                sample_redaction_callback):
        """Test initialization with all parameters."""
        resource_attributes = {"service.name": "test-service", "service.version": "1.0"}
        redaction_attributes = ["user.email", "session.token"]

        exporter = OTLPSpanHeaderRedactionAdapterExporter(
            # Base exporter args
            context_state=mock_context_state,
            batch_size=basic_exporter_config["batch_size"],
            flush_interval=basic_exporter_config["flush_interval"],
            max_queue_size=500,
            drop_on_overflow=True,
            shutdown_timeout=15.0,
            resource_attributes=resource_attributes,
            # Redaction args
            redaction_attributes=redaction_attributes,
            redaction_header="x-auth-user",
            redaction_callback=sample_redaction_callback,
            redaction_enabled=True,
            force_redaction=False,
            privacy_tag_key="privacy.level",
            privacy_tag_value=PrivacyLevel.HIGH,
            # OTLP args
            endpoint=basic_exporter_config["endpoint"],
            headers=basic_exporter_config["headers"])

        assert exporter is not None
        assert hasattr(exporter, '_exporter')
        assert isinstance(exporter._exporter, OTLPSpanExporter)
        assert exporter._resource.attributes["service.name"] == "test-service"
        assert exporter._resource.attributes["service.version"] == "1.0"

    def test_initialization_with_force_redaction(self, basic_exporter_config):
        """Test initialization with force_redaction=True."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          redaction_attributes=["secret_data"],
                                                          force_redaction=True,
                                                          redaction_enabled=True)

        assert exporter is not None

    def test_initialization_without_privacy_tag_value(self, basic_exporter_config):
        """Test initialization with privacy_tag_key but no privacy_tag_value."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          privacy_tag_key="privacy.level",
                                                          privacy_tag_value=None)

        assert exporter is not None

    def test_missing_endpoint_parameter(self):
        """Test that missing endpoint parameter raises appropriate error."""
        with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'endpoint'"):
            OTLPSpanHeaderRedactionAdapterExporter()


class TestOTLPSpanHeaderRedactionAdapterExporterProcessors:
    """Test suite for processor addition and configuration."""

    @pytest.fixture
    def basic_exporter_config(self):
        """Basic configuration for the exporter."""
        return {"endpoint": "https://api.example.com/v1/traces", "headers": {"Authorization": "Bearer test-token"}}

    @pytest.fixture
    def sample_redaction_callback(self):
        """Sample redaction callback for testing."""

        def should_redact(auth_key: str) -> bool:
            return auth_key in ["sensitive_user", "admin"]

        return should_redact

    @patch('nat.plugins.opentelemetry.otlp_span_adapter_exporter.OTLPSpanAdapterExporter.add_processor')
    def test_processor_addition_order(self, mock_add_processor, basic_exporter_config, sample_redaction_callback):
        """Test that processors are added in the correct order."""
        redaction_attributes = ["user.email"]

        OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                               redaction_attributes=redaction_attributes,
                                               redaction_header="x-auth-user",
                                               redaction_callback=sample_redaction_callback,
                                               redaction_enabled=True,
                                               privacy_tag_key="privacy.level",
                                               privacy_tag_value=PrivacyLevel.MEDIUM)

        # Verify add_processor was called 4 times total:
        # - 2 from parent OtelSpanExporter (SpanToOtelProcessor, OtelSpanBatchProcessor)
        # - 2 from our class (HeaderRedactionProcessor, SpanTaggingProcessor)
        assert mock_add_processor.call_count == 4

        # Find our redaction processor call (should have name="header_redaction")
        redaction_calls = [
            call for call in mock_add_processor.call_args_list
            if len(call) > 1 and call[1].get("name") == "header_redaction"
        ]
        assert len(redaction_calls) == 1
        redaction_call = redaction_calls[0]
        assert redaction_call[1]["position"] == 0
        assert isinstance(redaction_call[0][0], HeaderRedactionProcessor)

        # Find our tagging processor call (should have name="span_privacy_tagging")
        tagging_calls = [
            call for call in mock_add_processor.call_args_list
            if len(call) > 1 and call[1].get("name") == "span_privacy_tagging"
        ]
        assert len(tagging_calls) == 1
        tagging_call = tagging_calls[0]
        assert tagging_call[1]["position"] == 1
        assert isinstance(tagging_call[0][0], SpanTaggingProcessor)

    @patch('nat.plugins.opentelemetry.otlp_span_adapter_exporter.OTLPSpanAdapterExporter.add_processor')
    def test_header_redaction_processor_configuration(self,
                                                      mock_add_processor,
                                                      basic_exporter_config,
                                                      sample_redaction_callback):
        """Test that HeaderRedactionProcessor is configured correctly."""
        redaction_attributes = ["user.email", "user.phone"]
        redaction_header = "x-user-auth"

        OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                               redaction_attributes=redaction_attributes,
                                               redaction_header=redaction_header,
                                               redaction_callback=sample_redaction_callback,
                                               redaction_enabled=True,
                                               force_redaction=False)

        # Find the HeaderRedactionProcessor call by name
        redaction_calls = [
            call for call in mock_add_processor.call_args_list
            if len(call) > 1 and call[1].get("name") == "header_redaction"
        ]
        assert len(redaction_calls) == 1

        header_processor = redaction_calls[0][0][0]
        assert isinstance(header_processor, HeaderRedactionProcessor)
        assert header_processor.attributes == redaction_attributes
        assert header_processor.header == redaction_header
        assert header_processor.callback == sample_redaction_callback
        assert header_processor.enabled
        assert not header_processor.force_redact

    @patch('nat.plugins.opentelemetry.otlp_span_adapter_exporter.OTLPSpanAdapterExporter.add_processor')
    def test_span_tagging_processor_configuration(self, mock_add_processor, basic_exporter_config):
        """Test that SpanTaggingProcessor is configured correctly."""
        privacy_tag_key = "privacy.level"
        privacy_level = PrivacyLevel.HIGH

        OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                               privacy_tag_key=privacy_tag_key,
                                               privacy_tag_value=privacy_level)

        # Find the SpanTaggingProcessor call by name
        tagging_calls = [
            call for call in mock_add_processor.call_args_list
            if len(call) > 1 and call[1].get("name") == "span_privacy_tagging"
        ]
        assert len(tagging_calls) == 1

        tagging_processor = tagging_calls[0][0][0]
        assert isinstance(tagging_processor, SpanTaggingProcessor)
        assert tagging_processor.tag_key == privacy_tag_key
        assert tagging_processor.tag_value == privacy_level.value  # Should use .value

    @patch('nat.plugins.opentelemetry.otlp_span_adapter_exporter.OTLPSpanAdapterExporter.add_processor')
    def test_processors_added_with_none_values(self, mock_add_processor, basic_exporter_config):
        """Test that processors are still added even when optional values are None."""
        OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                               redaction_attributes=None,
                                               redaction_header=None,
                                               redaction_callback=None,
                                               privacy_tag_key=None,
                                               privacy_tag_value=None)

        # Should add 4 processors total (2 from parent + 2 from our class)
        assert mock_add_processor.call_count == 4

        # Find HeaderRedactionProcessor call
        redaction_calls = [
            call for call in mock_add_processor.call_args_list
            if len(call) > 1 and call[1].get("name") == "header_redaction"
        ]
        assert len(redaction_calls) == 1
        header_processor = redaction_calls[0][0][0]
        assert isinstance(header_processor, HeaderRedactionProcessor)

        # Find SpanTaggingProcessor call
        tagging_calls = [
            call for call in mock_add_processor.call_args_list
            if len(call) > 1 and call[1].get("name") == "span_privacy_tagging"
        ]
        assert len(tagging_calls) == 1
        tagging_processor = tagging_calls[0][0][0]
        assert isinstance(tagging_processor, SpanTaggingProcessor)
        assert tagging_processor.tag_key is None
        assert tagging_processor.tag_value is None


class TestOTLPSpanHeaderRedactionAdapterExporterRedaction:
    """Test suite for redaction functionality."""

    @pytest.fixture
    def basic_exporter_config(self):
        """Basic configuration for the exporter."""
        return {"endpoint": "https://api.example.com/v1/traces", "headers": {"Authorization": "Bearer test-token"}}

    @pytest.fixture
    def sample_span(self):
        """Create a sample span for testing."""
        return Span(name="test_span",
                    attributes={
                        "user.email": "user@example.com",
                        "user.phone": "123-456-7890",
                        "request.id": "req_123",
                        "system.info": "safe_data"
                    })

    def test_redaction_callback_functionality(self, basic_exporter_config):
        """Test different redaction callback scenarios."""

        # Callback that redacts for specific users
        def redact_for_test_users(auth_key: str) -> bool:
            return auth_key.startswith("test_") or auth_key == "admin"

        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          redaction_attributes=["user.email", "user.phone"],
                                                          redaction_header="x-user-id",
                                                          redaction_callback=redact_for_test_users,
                                                          redaction_enabled=True)

        assert exporter is not None

    def test_force_redaction_configuration(self, basic_exporter_config):
        """Test force_redaction=True configuration."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          redaction_attributes=["sensitive_data"],
                                                          force_redaction=True,
                                                          redaction_enabled=True)

        assert exporter is not None

    def test_redaction_disabled_configuration(self, basic_exporter_config):
        """Test with redaction_enabled=False."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          redaction_attributes=["user.email"],
                                                          redaction_header="x-user-id",
                                                          redaction_enabled=False)

        assert exporter is not None

    def test_default_redaction_value_configuration(self, basic_exporter_config):
        """Test that default redaction value is correctly set."""

        def test_redaction_callback(auth_key: str) -> bool:
            return auth_key in ["test_user", "admin"]

        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          redaction_attributes=["user.email"],
                                                          redaction_header="x-user-id",
                                                          redaction_callback=test_redaction_callback,
                                                          redaction_enabled=True)

        # Find the HeaderRedactionProcessor in the processors (should be at position 0)
        header_processor = None
        for processor in exporter._processors:
            if isinstance(processor, HeaderRedactionProcessor):
                header_processor = processor
                break

        assert header_processor is not None
        assert isinstance(header_processor, HeaderRedactionProcessor)
        assert header_processor.redaction_value == "[REDACTED]"  # Default value

    def test_custom_redaction_value_configuration(self, basic_exporter_config):
        """Test that custom redaction value is correctly passed through."""

        def test_redaction_callback(auth_key: str) -> bool:
            return auth_key in ["test_user", "admin"]

        custom_redaction_value = "***HIDDEN***"

        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          redaction_attributes=["user.email"],
                                                          redaction_header="x-user-id",
                                                          redaction_callback=test_redaction_callback,
                                                          redaction_enabled=True,
                                                          redaction_value=custom_redaction_value)

        # Find the HeaderRedactionProcessor in the processors (should be at position 0)
        header_processor = None
        for processor in exporter._processors:
            if isinstance(processor, HeaderRedactionProcessor):
                header_processor = processor
                break

        assert header_processor is not None
        assert isinstance(header_processor, HeaderRedactionProcessor)
        assert header_processor.redaction_value == custom_redaction_value

    @patch('nat.observability.processor.header_redaction_processor.Context.get')
    async def test_redaction_value_end_to_end(self, mock_context_get, basic_exporter_config):
        """Test that custom redaction values work end-to-end in span processing."""
        # Setup context with headers that trigger redaction
        headers = Headers({"x-user-id": "sensitive_user"})
        metadata = Mock()
        metadata.headers = headers
        context = Mock()
        context.metadata = metadata
        mock_context_get.return_value = context

        def should_redact_sensitive_users(auth_key: str) -> bool:
            return auth_key == "sensitive_user"

        custom_redaction_value = "***CLASSIFIED***"

        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          redaction_attributes=["user.email", "user.ssn"],
                                                          redaction_header="x-user-id",
                                                          redaction_callback=should_redact_sensitive_users,
                                                          redaction_enabled=True,
                                                          redaction_value=custom_redaction_value)

        # Create a span with sensitive data
        span = Span(
            name="test_operation",
            attributes={
                "user.email": "sensitive@example.com",
                "user.ssn": "123-45-6789",
                "user.name": "John Doe",  # Not in redaction list
                "request.id": "req_123"  # Not in redaction list
            })

        # Process the span through the redaction processor
        header_processor = None
        for processor in exporter._processors:
            if isinstance(processor, HeaderRedactionProcessor):
                header_processor = processor
                break

        assert header_processor is not None
        processed_span = await header_processor.process(span)

        # Verify redaction occurred with custom value
        assert processed_span.attributes["user.email"] == custom_redaction_value
        assert processed_span.attributes["user.ssn"] == custom_redaction_value
        # Non-redacted fields should remain unchanged
        assert processed_span.attributes["user.name"] == "John Doe"
        assert processed_span.attributes["request.id"] == "req_123"

    @patch('nat.observability.processor.header_redaction_processor.Context.get')
    async def test_default_redaction_value_end_to_end(self, mock_context_get, basic_exporter_config):
        """Test that default redaction value works end-to-end in span processing."""
        # Setup context with headers that trigger redaction
        headers = Headers({"x-user-id": "test_user"})
        metadata = Mock()
        metadata.headers = headers
        context = Mock()
        context.metadata = metadata
        mock_context_get.return_value = context

        def should_redact_test_users(auth_key: str) -> bool:
            return auth_key == "test_user"

        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          redaction_attributes=["user.email"],
                                                          redaction_header="x-user-id",
                                                          redaction_callback=should_redact_test_users,
                                                          redaction_enabled=True)
        # No redaction_value specified - should use default "[REDACTED]"

        # Create a span with sensitive data
        span = Span(name="test_operation", attributes={"user.email": "user@example.com", "public.data": "safe_value"})

        # Process the span through the redaction processor
        header_processor = None
        for processor in exporter._processors:
            if isinstance(processor, HeaderRedactionProcessor):
                header_processor = processor
                break

        assert header_processor is not None
        processed_span = await header_processor.process(span)

        # Verify redaction occurred with default value
        assert processed_span.attributes["user.email"] == "[REDACTED]"
        # Non-redacted fields should remain unchanged
        assert processed_span.attributes["public.data"] == "safe_value"


class TestOTLPSpanHeaderRedactionAdapterExporterPrivacyTagging:
    """Test suite for privacy tagging functionality."""

    @pytest.fixture
    def basic_exporter_config(self):
        """Basic configuration for the exporter."""
        return {"endpoint": "https://api.example.com/v1/traces"}

    def test_privacy_level_none(self, basic_exporter_config):
        """Test privacy tagging with PrivacyLevel.NONE."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          privacy_tag_key="privacy.level",
                                                          privacy_tag_value=PrivacyLevel.NONE)

        assert exporter is not None

    def test_privacy_level_low(self, basic_exporter_config):
        """Test privacy tagging with PrivacyLevel.LOW."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          privacy_tag_key="privacy.level",
                                                          privacy_tag_value=PrivacyLevel.LOW)

        assert exporter is not None

    def test_privacy_level_medium(self, basic_exporter_config):
        """Test privacy tagging with PrivacyLevel.MEDIUM."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          privacy_tag_key="privacy.level",
                                                          privacy_tag_value=PrivacyLevel.MEDIUM)

        assert exporter is not None

    def test_privacy_level_high(self, basic_exporter_config):
        """Test privacy tagging with PrivacyLevel.HIGH."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          privacy_tag_key="privacy.level",
                                                          privacy_tag_value=PrivacyLevel.HIGH)

        assert exporter is not None

    def test_custom_privacy_tag_key(self, basic_exporter_config):
        """Test with custom privacy tag key."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          privacy_tag_key="custom.privacy.classification",
                                                          privacy_tag_value=PrivacyLevel.MEDIUM)

        assert exporter is not None

    def test_privacy_tagging_without_tag_key(self, basic_exporter_config):
        """Test privacy tagging with only tag_value but no tag_key."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          privacy_tag_key=None,
                                                          privacy_tag_value=PrivacyLevel.HIGH)

        assert exporter is not None

    def test_privacy_tagging_without_tag_value(self, basic_exporter_config):
        """Test privacy tagging with only tag_key but no tag_value."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          privacy_tag_key="privacy.level",
                                                          privacy_tag_value=None)

        assert exporter is not None


class TestOTLPSpanHeaderRedactionAdapterExporterIntegration:
    """Test suite for integration scenarios."""

    @pytest.fixture
    def mock_context_state(self):
        """Create a mock ContextState for testing."""
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
                                             data=StreamEventData(input="Test input with sensitive data"),
                                             metadata={
                                                 "user.email": "user@example.com", "key": "value"
                                             },
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
                                             data=StreamEventData(output="Test output with results"),
                                             metadata={
                                                 "user.email": "user@example.com", "key": "value"
                                             },
                                             UUID=test_uuid)

    @pytest.fixture
    def sample_redaction_callback(self):
        """Sample redaction callback for testing."""

        def should_redact(auth_key: str) -> bool:
            return auth_key in ["sensitive_user", "test_user"]

        return should_redact

    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporter')
    async def test_end_to_end_with_redaction_and_tagging(
        self,
        mock_otlp_exporter_class,
        basic_exporter_config,
        sample_start_event,
        sample_end_event,
        sample_redaction_callback,
    ):
        """Test end-to-end processing with both redaction and privacy tagging enabled."""
        # Setup mock
        mock_otlp_exporter = Mock()
        mock_otlp_exporter.export = Mock()
        mock_otlp_exporter_class.return_value = mock_otlp_exporter

        exporter = OTLPSpanHeaderRedactionAdapterExporter(
            endpoint=basic_exporter_config["endpoint"],
            headers=basic_exporter_config["headers"],
            redaction_attributes=["user.email"],
            redaction_header="x-user-id",
            redaction_callback=sample_redaction_callback,
            redaction_enabled=True,
            privacy_tag_key="privacy.level",
            privacy_tag_value=PrivacyLevel.HIGH,
            batch_size=1,  # Force immediate processing
            flush_interval=0.1)

        # Use same UUID for start and end events to create a complete span
        sample_end_event.payload.UUID = sample_start_event.payload.UUID

        async with exporter.start():
            # Process events
            exporter.export(sample_start_event)
            exporter.export(sample_end_event)

            # Wait for async processing
            await exporter.wait_for_tasks()

        # Verify that export was called
        mock_otlp_exporter.export.assert_called()

    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporter')
    async def test_redaction_only_configuration(
        self,
        mock_otlp_exporter_class,
        basic_exporter_config,
        sample_start_event,
        sample_end_event,
        sample_redaction_callback,
    ):
        """Test configuration with only redaction enabled."""
        # Setup mock
        mock_otlp_exporter = Mock()
        mock_otlp_exporter.export = Mock()
        mock_otlp_exporter_class.return_value = mock_otlp_exporter

        exporter = OTLPSpanHeaderRedactionAdapterExporter(
            endpoint=basic_exporter_config["endpoint"],
            redaction_attributes=["user.email", "user.ssn"],
            redaction_header="x-auth-token",
            redaction_callback=sample_redaction_callback,
            redaction_enabled=True,
            force_redaction=False,
            # No privacy tagging configured
            privacy_tag_key=None,
            privacy_tag_value=None,
            batch_size=1,
            flush_interval=0.1)

        # Use same UUID for start and end events
        sample_end_event.payload.UUID = sample_start_event.payload.UUID

        async with exporter.start():
            exporter.export(sample_start_event)
            exporter.export(sample_end_event)
            await exporter.wait_for_tasks()

        mock_otlp_exporter.export.assert_called()

    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporter')
    async def test_privacy_tagging_only_configuration(
        self,
        mock_otlp_exporter_class,
        basic_exporter_config,
        sample_start_event,
        sample_end_event,
    ):
        """Test configuration with only privacy tagging enabled."""
        # Setup mock
        mock_otlp_exporter = Mock()
        mock_otlp_exporter.export = Mock()
        mock_otlp_exporter_class.return_value = mock_otlp_exporter

        exporter = OTLPSpanHeaderRedactionAdapterExporter(
            endpoint=basic_exporter_config["endpoint"],
            # No redaction configured
            redaction_attributes=None,
            redaction_header=None,
            redaction_callback=None,
            redaction_enabled=False,
            force_redaction=False,
            # Only privacy tagging
            privacy_tag_key="compliance.level",
            privacy_tag_value=PrivacyLevel.MEDIUM,
            batch_size=1,
            flush_interval=0.1)

        # Use same UUID for start and end events
        sample_end_event.payload.UUID = sample_start_event.payload.UUID

        async with exporter.start():
            exporter.export(sample_start_event)
            exporter.export(sample_end_event)
            await exporter.wait_for_tasks()

        mock_otlp_exporter.export.assert_called()


class TestOTLPSpanHeaderRedactionAdapterExporterInheritance:
    """Test suite for inheritance and interface compatibility."""

    @pytest.fixture
    def basic_exporter_config(self):
        """Basic configuration for the exporter."""
        return {"endpoint": "https://api.example.com/v1/traces", "headers": {"Authorization": "Bearer test-token"}}

    def test_inheritance_structure(self, basic_exporter_config):
        """Test that OTLPSpanHeaderRedactionAdapterExporter inherits from correct classes."""
        from nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin import OTLPSpanExporterMixin
        from nat.plugins.opentelemetry.otel_span_exporter import OtelSpanExporter

        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"])

        # Should inherit from base classes
        assert isinstance(exporter, OTLPSpanAdapterExporter)
        assert isinstance(exporter, OTLPSpanExporterMixin)
        assert isinstance(exporter, OtelSpanExporter)

    def test_method_availability(self, basic_exporter_config):
        """Test that inherited methods are available."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"])

        # Should have all expected methods from parent classes
        assert hasattr(exporter, 'export')
        assert hasattr(exporter, 'export_otel_spans')
        assert hasattr(exporter, 'export_processed')
        assert hasattr(exporter, 'add_processor')
        assert hasattr(exporter, 'start')
        assert hasattr(exporter, 'wait_for_tasks')

    def test_processor_management_methods(self, basic_exporter_config):
        """Test processor management methods are inherited correctly."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"])

        # Should have processor management capabilities
        assert hasattr(exporter, 'add_processor')
        assert callable(getattr(exporter, 'add_processor'))

    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporter')
    def test_otlp_exporter_initialization(self, mock_otlp_exporter_class, basic_exporter_config):
        """Test that the underlying OTLP exporter is properly initialized."""
        headers = basic_exporter_config["headers"]
        endpoint = basic_exporter_config["endpoint"]

        OTLPSpanHeaderRedactionAdapterExporter(endpoint=endpoint,
                                               headers=headers,
                                               redaction_enabled=True,
                                               privacy_tag_value=PrivacyLevel.LOW)

        # Verify OTLPSpanExporter was initialized with correct parameters
        mock_otlp_exporter_class.assert_called_once_with(endpoint=endpoint, headers=headers)


class TestOTLPSpanHeaderRedactionAdapterExporterEdgeCases:
    """Test suite for edge cases and error scenarios."""

    @pytest.fixture
    def basic_exporter_config(self):
        """Basic configuration for the exporter."""
        return {"endpoint": "https://api.example.com/v1/traces"}

    def test_empty_redaction_attributes_list(self, basic_exporter_config):
        """Test with empty redaction attributes list."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(
            endpoint=basic_exporter_config["endpoint"],
            redaction_attributes=[],  # Empty list
            redaction_enabled=True)

        assert exporter is not None

    def test_complex_redaction_callback(self, basic_exporter_config):
        """Test with complex redaction callback logic."""

        def complex_callback(auth_key: str) -> bool:
            # Complex logic with multiple conditions
            if not auth_key:
                return False

            # Redact for test environments
            if auth_key.startswith("test_"):
                return True

            # Redact for admin users in specific environments
            if "admin" in auth_key and "prod" not in auth_key:
                return True

            return False

        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          redaction_attributes=["sensitive_field"],
                                                          redaction_header="x-environment-user",
                                                          redaction_callback=complex_callback,
                                                          redaction_enabled=True)

        assert exporter is not None

    def test_multiple_redaction_attributes(self, basic_exporter_config):
        """Test with multiple redaction attributes."""
        redaction_attributes = [
            "user.email", "user.phone", "user.ssn", "payment.card_number", "auth.session_token", "internal.debug_info"
        ]

        exporter = OTLPSpanHeaderRedactionAdapterExporter(
            endpoint=basic_exporter_config["endpoint"],
            redaction_attributes=redaction_attributes,
            redaction_enabled=True,
            force_redaction=True  # Always redact these sensitive fields
        )

        assert exporter is not None

    def test_inheritance_with_super_call(self, basic_exporter_config):
        """Test that super().__init__ is called correctly with all parameters."""
        mock_context_state = Mock(spec=ContextState)
        resource_attributes = {"service.name": "test-service"}

        # This should not raise any errors about missing parameters
        exporter = OTLPSpanHeaderRedactionAdapterExporter(context_state=mock_context_state,
                                                          batch_size=75,
                                                          flush_interval=3.0,
                                                          max_queue_size=800,
                                                          drop_on_overflow=True,
                                                          shutdown_timeout=20.0,
                                                          resource_attributes=resource_attributes,
                                                          endpoint=basic_exporter_config["endpoint"],
                                                          redaction_enabled=True,
                                                          privacy_tag_value=PrivacyLevel.LOW)

        assert exporter is not None
        assert exporter._resource.attributes["service.name"] == "test-service"

    def test_redaction_callback_none_handling(self, basic_exporter_config):
        """Test handling when redaction_callback is None."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(
            endpoint=basic_exporter_config["endpoint"],
            redaction_attributes=["user.data"],
            redaction_header="x-auth",
            redaction_callback=None,  # Explicitly None
            redaction_enabled=True)

        assert exporter is not None
        # The HeaderRedactionProcessor should handle None callback by using default_callback

    def test_combined_force_redaction_and_privacy_tagging(self, basic_exporter_config):
        """Test combining force_redaction with privacy tagging."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(
            endpoint=basic_exporter_config["endpoint"],
            redaction_attributes=["sensitive_data", "user_info"],
            force_redaction=True,  # Always redact
            redaction_enabled=True,
            privacy_tag_key="security.classification",
            privacy_tag_value=PrivacyLevel.HIGH)

        assert exporter is not None


class TestOTLPSpanHeaderRedactionAdapterExporterExportFunctionality:
    """Test suite for export functionality with processing pipeline."""

    @pytest.fixture
    def basic_exporter_config(self):
        """Basic configuration for the exporter."""
        return {"endpoint": "https://api.example.com/v1/traces", "headers": {"Authorization": "Bearer test-token"}}

    @pytest.fixture
    def mock_otel_span(self):
        """Create a mock OtelSpan for testing."""
        span = Mock(spec=OtelSpan)
        span.set_resource = Mock()
        return span

    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporter')
    async def test_export_otel_spans_with_processing(self,
                                                     mock_otlp_exporter_class,
                                                     basic_exporter_config,
                                                     mock_otel_span):
        """Test export of OtelSpans through the processing pipeline."""
        # Setup mock
        mock_otlp_exporter = Mock()
        mock_otlp_exporter.export = Mock()
        mock_otlp_exporter_class.return_value = mock_otlp_exporter

        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          headers=basic_exporter_config["headers"],
                                                          redaction_attributes=["sensitive_field"],
                                                          redaction_enabled=True,
                                                          privacy_tag_key="privacy.level",
                                                          privacy_tag_value=PrivacyLevel.MEDIUM)

        spans = [mock_otel_span]

        # Test export
        await exporter.export_otel_spans(spans)

        # Verify the OTLP exporter was called
        mock_otlp_exporter.export.assert_called_once_with(spans)

    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporter')
    async def test_export_processed_with_resource_attributes(
        self,
        mock_otlp_exporter_class,
        basic_exporter_config,
        mock_otel_span,
    ):
        """Test that export_processed applies resource attributes and processing."""
        # Setup mock
        mock_otlp_exporter = Mock()
        mock_otlp_exporter.export = Mock()
        mock_otlp_exporter_class.return_value = mock_otlp_exporter

        resource_attributes = {"service.name": "redacted-service"}
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          resource_attributes=resource_attributes,
                                                          privacy_tag_key="privacy.level",
                                                          privacy_tag_value=PrivacyLevel.LOW)

        # Test export_processed method
        await exporter.export_processed(mock_otel_span)

        # Verify resource was set on the span
        mock_otel_span.set_resource.assert_called_once_with(exporter._resource)

        # Verify export was called
        mock_otlp_exporter.export.assert_called_once()

    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporter')
    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.logger')
    async def test_export_with_exception_handling(
        self,
        mock_logger,
        mock_otlp_exporter_class,
        basic_exporter_config,
        mock_otel_span,
    ):
        """Test export with exception handling from the underlying exporter."""
        # Setup mock to raise exception
        mock_otlp_exporter = Mock()
        mock_otlp_exporter.export = Mock(side_effect=Exception("Export failed"))
        mock_otlp_exporter_class.return_value = mock_otlp_exporter

        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint=basic_exporter_config["endpoint"],
                                                          redaction_enabled=True,
                                                          privacy_tag_value=PrivacyLevel.HIGH)

        spans = [mock_otel_span]

        # Test export - should not raise exception
        await exporter.export_otel_spans(spans)

        # Verify error was logged (inherited behavior)
        mock_logger.error.assert_called_once()
        assert "Error exporting spans" in str(mock_logger.error.call_args)


class TestOTLPSpanHeaderRedactionAdapterExporterBatching:
    """Test suite for batching behavior with redaction and tagging."""

    @pytest.fixture
    def basic_exporter_config(self):
        """Basic configuration for the exporter."""
        return {"endpoint": "https://api.example.com/v1/traces", "headers": {"Authorization": "Bearer test-token"}}

    @patch('nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporter')
    async def test_batching_with_redaction_and_tagging(self, mock_otlp_exporter_class, basic_exporter_config):
        """Test that batching works correctly with redaction and tagging processors."""
        # Setup mock
        mock_otlp_exporter = Mock()
        mock_otlp_exporter.export = Mock()
        mock_otlp_exporter_class.return_value = mock_otlp_exporter

        batch_size = 3
        exporter = OTLPSpanHeaderRedactionAdapterExporter(
            endpoint=basic_exporter_config["endpoint"],
            headers=basic_exporter_config["headers"],
            batch_size=batch_size,
            flush_interval=10.0,  # Long interval to test batching
            redaction_attributes=["user.email"],
            redaction_enabled=True,
            privacy_tag_key="privacy.level",
            privacy_tag_value=PrivacyLevel.MEDIUM)

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
                                                            metadata={"user.email": f"user{i}@example.com"},
                                                            UUID=f"uuid_{i}")

                end_event = create_test_intermediate_step(parent_id="root",
                                                          function_name=f"test_function_{i}",
                                                          function_id=f"func_{i}",
                                                          event_type=IntermediateStepType.LLM_END,
                                                          framework=LLMFrameworkEnum.LANGCHAIN,
                                                          name=f"test_call_{i}",
                                                          event_timestamp=datetime.now().timestamp(),
                                                          data=StreamEventData(output=f"Output {i}"),
                                                          metadata={"user.email": f"user{i}@example.com"},
                                                          UUID=f"uuid_{i}")

                exporter.export(start_event)
                exporter.export(end_event)

            # Wait for batch processing
            await exporter.wait_for_tasks()

        # Verify that export was called (batching should trigger export)
        mock_otlp_exporter.export.assert_called()


class TestOTLPSpanHeaderRedactionAdapterExporterRealWorldScenarios:
    """Test suite for real-world usage scenarios."""

    @pytest.fixture
    def basic_exporter_config(self):
        """Basic configuration for the exporter."""
        return {"endpoint": "https://api.example.com/v1/traces", "headers": {"Authorization": "Bearer test-token"}}

    def test_datadog_integration_scenario(self):
        """Test configuration for DataDog OTLP endpoint integration."""

        def datadog_redaction_callback(auth_key: str) -> bool:
            # Redact for non-production environments
            return auth_key in ["dev", "staging", "test"]

        exporter = OTLPSpanHeaderRedactionAdapterExporter(
            endpoint="https://api.datadoghq.com/api/v1/traces",
            headers={"DD-API-KEY": "fake-datadog-key"},
            redaction_attributes=["user.email", "user.ip", "request.body"],
            redaction_header="x-environment",
            redaction_callback=datadog_redaction_callback,
            redaction_enabled=True,
            privacy_tag_key="privacy.level",
            privacy_tag_value=PrivacyLevel.MEDIUM,
            batch_size=100,
            flush_interval=5.0)

        assert exporter is not None

    def test_jaeger_integration_scenario(self):
        """Test configuration for Jaeger OTLP endpoint integration."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(endpoint="http://jaeger-collector:14268/api/traces",
                                                          redaction_attributes=["auth.token", "user.credentials"],
                                                          redaction_header="authorization",
                                                          redaction_enabled=True,
                                                          force_redaction=False,
                                                          privacy_tag_key="compliance.level",
                                                          privacy_tag_value=PrivacyLevel.HIGH,
                                                          resource_attributes={
                                                              "service.name": "nemo-agent-toolkit",
                                                              "service.version": "1.0.0",
                                                              "deployment.environment": "production"
                                                          })

        assert exporter is not None

    def test_custom_otlp_backend_scenario(self):
        """Test configuration for custom OTLP-compatible backend."""

        def enterprise_redaction_callback(auth_key: str) -> bool:
            # Enterprise-specific redaction logic
            return auth_key.startswith("external_") or auth_key.endswith("_guest")

        exporter = OTLPSpanHeaderRedactionAdapterExporter(
            endpoint="https://enterprise-traces.company.com/otlp/v1/traces",
            headers={
                "Authorization": "Bearer enterprise-token", "X-Tenant-ID": "prod-tenant-123"
            },
            redaction_attributes=[
                "user.pii.email", "user.pii.phone", "payment.sensitive_data", "internal.proprietary_info"
            ],
            redaction_header="x-user-classification",
            redaction_callback=enterprise_redaction_callback,
            redaction_enabled=True,
            force_redaction=False,
            privacy_tag_key="enterprise.privacy.classification",
            privacy_tag_value=PrivacyLevel.HIGH,
            batch_size=200,
            flush_interval=2.0,
            max_queue_size=2000,
            resource_attributes={
                "service.name": "enterprise-agent",
                "service.version": "2.1.0",
                "enterprise.tenant.id": "prod-tenant-123"
            })

        assert exporter is not None

    def test_high_volume_scenario_configuration(self):
        """Test configuration optimized for high volume scenarios."""
        exporter = OTLPSpanHeaderRedactionAdapterExporter(
            endpoint="https://api.example.com/v1/traces",
            batch_size=500,  # Large batch size for high volume
            flush_interval=1.0,  # Frequent flushes
            max_queue_size=5000,
            drop_on_overflow=True,  # Drop spans if overwhelmed
            shutdown_timeout=30.0,
            redaction_attributes=["user.data"],
            redaction_enabled=True,
            privacy_tag_key="volume.classification",
            privacy_tag_value=PrivacyLevel.LOW  # Lower privacy for high-volume data
        )

        assert exporter is not None

    def test_development_environment_scenario(self):
        """Test configuration for development environment with detailed logging."""

        def dev_redaction_callback(auth_key: str) -> bool:
            # In development, only redact for specific test cases
            return auth_key == "redaction_test_user"

        exporter = OTLPSpanHeaderRedactionAdapterExporter(
            endpoint="http://localhost:4318/v1/traces",  # Local development endpoint
            redaction_attributes=["test.sensitive_field"],
            redaction_header="x-test-user",
            redaction_callback=dev_redaction_callback,
            redaction_enabled=True,
            force_redaction=False,
            privacy_tag_key="dev.privacy.level",
            privacy_tag_value=PrivacyLevel.NONE,  # Development environment
            batch_size=10,  # Small batches for easier debugging
            flush_interval=1.0,  # Fast flushes for immediate feedback
            resource_attributes={
                "service.name": "nat-dev", "environment": "development", "developer": "test-user"
            })

        assert exporter is not None
