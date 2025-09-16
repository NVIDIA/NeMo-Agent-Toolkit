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

from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from starlette.datastructures import Headers

from nat.builder.context import Context
from nat.data_models.span import Span
from nat.data_models.span import SpanContext
from nat.observability.processor.redaction import SpanHeaderRedactionProcessor
from nat.runtime.user_metadata import RequestAttributes


def default_callback(_data: dict[str, Any]) -> bool:
    """Default callback that always returns False."""
    return False


@pytest.fixture(name="sample_span")
def sample_span():
    """Create a sample span for testing."""
    span_context = SpanContext(span_id=123, trace_id=456)
    return Span(name="test_operation",
                context=span_context,
                parent=None,
                start_time=1000000,
                end_time=2000000,
                attributes={
                    "user_id": "user123",
                    "session_token": "secret_token",
                    "api_key": "api_secret",
                    "normal_field": "normal_value"
                },
                events=[])


@pytest.fixture(name="mock_context_with_headers")
def mock_context_with_headers():
    """Create a mock context with headers."""
    headers = Headers({"authorization": "Bearer token123", "x-api-key": "key456"})
    metadata = Mock(spec=RequestAttributes)
    metadata.headers = headers

    context = Mock(spec=Context)
    context.metadata = metadata
    return context


@pytest.fixture(name="mock_context_no_headers")
def mock_context_no_headers():
    """Create a mock context without headers."""
    metadata = Mock(spec=RequestAttributes)
    metadata.headers = None

    context = Mock(spec=Context)
    context.metadata = metadata
    return context


class TestSpanHeaderRedactionProcessorInitialization:
    """Test SpanHeaderRedactionProcessor initialization."""

    def test_default_initialization(self):
        """Test initialization parameters with mandatory callback."""
        processor = SpanHeaderRedactionProcessor(headers=[], attributes=[], callback=default_callback)

        assert processor.attributes == []
        assert processor.headers == []
        assert processor.callback is default_callback
        assert processor.enabled is True
        assert processor.force_redact is False
        assert processor.redaction_value == "[REDACTED]"

    def test_initialization_with_attributes(self):
        """Test initialization with custom attributes."""
        attributes = ["user_id", "session_token"]
        processor = SpanHeaderRedactionProcessor(headers=[], attributes=attributes, callback=default_callback)

        assert processor.attributes == attributes
        assert processor.headers == []
        assert processor.callback is default_callback
        assert processor.enabled is True
        assert processor.force_redact is False
        assert processor.redaction_value == "[REDACTED]"

    def test_initialization_with_single_header(self):
        """Test initialization with single header."""
        processor = SpanHeaderRedactionProcessor(headers=["authorization"], attributes=[], callback=default_callback)

        assert processor.attributes == []
        assert processor.headers == ["authorization"]
        assert processor.callback is default_callback
        assert processor.enabled is True
        assert processor.force_redact is False

    def test_initialization_with_multiple_headers(self):
        """Test initialization with multiple headers."""
        headers = ["authorization", "x-api-key", "x-user-id"]
        processor = SpanHeaderRedactionProcessor(headers=headers, attributes=[], callback=default_callback)

        assert processor.attributes == []
        assert processor.headers == headers
        assert processor.callback is default_callback
        assert processor.enabled is True
        assert processor.force_redact is False

    def test_initialization_with_callback(self):
        """Test initialization with custom callback."""

        def custom_callback(data: dict[str, Any]) -> bool:
            auth = data.get("authorization", "")
            return "admin" in auth

        processor = SpanHeaderRedactionProcessor(headers=[], attributes=[], callback=custom_callback)

        assert processor.attributes == []
        assert processor.headers == []
        assert processor.callback is custom_callback
        assert processor.enabled is True
        assert processor.force_redact is False
        assert processor.redaction_value == "[REDACTED]"

    def test_initialization_with_redaction_tag(self):
        """Test initialization with redaction_tag parameter."""
        processor = SpanHeaderRedactionProcessor(headers=[],
                                                 attributes=[],
                                                 callback=default_callback,
                                                 redaction_tag="redacted")

        assert processor.redaction_tag == "redacted"

    def test_initialization_with_all_parameters(self):
        """Test initialization with all parameters specified."""
        attributes = ["user_id", "api_key"]
        headers = ["x-api-key", "authorization"]

        def callback(data: dict[str, Any]) -> bool:
            api_key = data.get("x-api-key", "")
            return len(api_key) > 10

        processor = SpanHeaderRedactionProcessor(headers=headers,
                                                 attributes=attributes,
                                                 callback=callback,
                                                 enabled=False,
                                                 force_redact=True,
                                                 redaction_value="[CUSTOM]",
                                                 redaction_tag="was_redacted")

        assert processor.attributes == attributes
        assert processor.headers == headers
        assert processor.callback is callback
        assert processor.enabled is False
        assert processor.force_redact is True
        assert processor.redaction_value == "[CUSTOM]"
        assert processor.redaction_tag == "was_redacted"


class TestSpanHeaderRedactionProcessorExtractDataFromContext:
    """Test extract_data_from_context method."""

    @patch('nat.builder.context.Context.get')
    def test_extract_data_with_headers(self, mock_context_get):
        """Test extracting data when headers exist."""
        headers = Headers({"authorization": "Bearer token123", "x-api-key": "key456"})
        metadata = Mock(spec=RequestAttributes)
        metadata.headers = headers
        context = Mock(spec=Context)
        context.metadata = metadata
        mock_context_get.return_value = context

        processor = SpanHeaderRedactionProcessor(headers=["authorization", "x-api-key"],
                                                 attributes=[],
                                                 callback=default_callback)

        result = processor.extract_data_from_context()

        expected = {"authorization": "Bearer token123", "x-api-key": "key456"}
        assert result == expected

    @patch('nat.builder.context.Context.get')
    def test_extract_data_with_missing_headers(self, mock_context_get):
        """Test extracting data when some headers are missing."""
        headers = Headers({"authorization": "Bearer token123"})
        metadata = Mock(spec=RequestAttributes)
        metadata.headers = headers
        context = Mock(spec=Context)
        context.metadata = metadata
        mock_context_get.return_value = context

        processor = SpanHeaderRedactionProcessor(headers=["authorization", "missing-header"],
                                                 attributes=[],
                                                 callback=default_callback)

        result = processor.extract_data_from_context()

        expected = {"authorization": "Bearer token123", "missing-header": None}
        assert result == expected

    @patch('nat.builder.context.Context.get')
    def test_extract_data_with_no_headers_in_context(self, mock_context_get):
        """Test extracting data when context has no headers."""
        metadata = Mock(spec=RequestAttributes)
        metadata.headers = None
        context = Mock(spec=Context)
        context.metadata = metadata
        mock_context_get.return_value = context

        processor = SpanHeaderRedactionProcessor(headers=["authorization"], attributes=[], callback=default_callback)

        result = processor.extract_data_from_context()

        assert result is None

    @patch('nat.builder.context.Context.get')
    def test_extract_data_with_empty_headers_list(self, mock_context_get):
        """Test extracting data when headers list is empty."""
        headers = Headers({"authorization": "Bearer token123"})
        metadata = Mock(spec=RequestAttributes)
        metadata.headers = headers
        context = Mock(spec=Context)
        context.metadata = metadata
        mock_context_get.return_value = context

        processor = SpanHeaderRedactionProcessor(headers=[], attributes=[], callback=default_callback)

        result = processor.extract_data_from_context()

        assert result is None


class TestSpanHeaderRedactionProcessorValidateData:
    """Test validate_data method."""

    def test_validate_data_with_valid_headers(self):
        """Test validation with valid header data."""
        processor = SpanHeaderRedactionProcessor(headers=[], attributes=[], callback=default_callback)

        data = {"authorization": "Bearer token123", "x-api-key": "key456"}

        result = processor.validate_data(data)
        assert result is True

    def test_validate_data_with_some_none_values(self):
        """Test validation when some headers are None but not all."""
        processor = SpanHeaderRedactionProcessor(headers=[], attributes=[], callback=default_callback)

        data = {"authorization": "Bearer token123", "missing-header": None}

        result = processor.validate_data(data)
        assert result is True

    def test_validate_data_with_all_none_values(self):
        """Test validation when all headers are None."""
        processor = SpanHeaderRedactionProcessor(headers=[], attributes=[], callback=default_callback)

        data = {"authorization": None, "x-api-key": None}

        result = processor.validate_data(data)
        assert result is False

    def test_validate_data_with_empty_dict(self):
        """Test validation with empty data dictionary."""
        processor = SpanHeaderRedactionProcessor(headers=[], attributes=[], callback=default_callback)

        data = {}

        result = processor.validate_data(data)
        assert result is False

    def test_validate_data_with_empty_string_values(self):
        """Test validation with empty string values (should be valid)."""
        processor = SpanHeaderRedactionProcessor(headers=[], attributes=[], callback=default_callback)

        data = {"authorization": "", "x-api-key": "key456"}

        result = processor.validate_data(data)
        assert result is True


class TestSpanHeaderRedactionProcessorRedactItem:
    """Test redact_item method."""

    async def test_redact_item_with_single_attribute(self, sample_span):
        """Test redacting a single attribute."""
        processor = SpanHeaderRedactionProcessor(headers=[], attributes=["user_id"], callback=default_callback)

        # Create a copy to avoid mutating the fixture
        test_span = Span(name=sample_span.name,
                         context=sample_span.context,
                         parent=sample_span.parent,
                         start_time=sample_span.start_time,
                         end_time=sample_span.end_time,
                         attributes=sample_span.attributes.copy(),
                         events=sample_span.events)

        result = await processor.redact_item(test_span)

        assert result.attributes["user_id"] == "[REDACTED]"
        assert result.attributes["session_token"] == "secret_token"  # Unchanged
        assert result.attributes["api_key"] == "api_secret"  # Unchanged
        assert result.attributes["normal_field"] == "normal_value"  # Unchanged

    async def test_redact_item_with_multiple_attributes(self, sample_span):
        """Test redacting multiple attributes."""
        processor = SpanHeaderRedactionProcessor(headers=[],
                                                 attributes=["user_id", "session_token", "api_key"],
                                                 callback=default_callback)

        # Create a copy to avoid mutating the fixture
        test_span = Span(name=sample_span.name,
                         context=sample_span.context,
                         parent=sample_span.parent,
                         start_time=sample_span.start_time,
                         end_time=sample_span.end_time,
                         attributes=sample_span.attributes.copy(),
                         events=sample_span.events)

        result = await processor.redact_item(test_span)

        assert result.attributes["user_id"] == "[REDACTED]"
        assert result.attributes["session_token"] == "[REDACTED]"
        assert result.attributes["api_key"] == "[REDACTED]"
        assert result.attributes["normal_field"] == "normal_value"  # Unchanged

    async def test_redact_item_with_redaction_tag(self, sample_span):
        """Test redacting with redaction_tag set."""
        processor = SpanHeaderRedactionProcessor(headers=[],
                                                 attributes=["user_id"],
                                                 callback=default_callback,
                                                 redaction_tag="was_redacted")

        # Create a copy to avoid mutating the fixture
        test_span = Span(name=sample_span.name,
                         context=sample_span.context,
                         parent=sample_span.parent,
                         start_time=sample_span.start_time,
                         end_time=sample_span.end_time,
                         attributes=sample_span.attributes.copy(),
                         events=sample_span.events)

        result = await processor.redact_item(test_span)

        assert result.attributes["user_id"] == "[REDACTED]"
        assert result.attributes["was_redacted"] is True

    async def test_redact_item_with_custom_redaction_value(self, sample_span):
        """Test redacting with custom redaction value."""
        processor = SpanHeaderRedactionProcessor(headers=[],
                                                 attributes=["user_id"],
                                                 callback=default_callback,
                                                 redaction_value="[CUSTOM_REDACTED]")

        # Create a copy to avoid mutating the fixture
        test_span = Span(name=sample_span.name,
                         context=sample_span.context,
                         parent=sample_span.parent,
                         start_time=sample_span.start_time,
                         end_time=sample_span.end_time,
                         attributes=sample_span.attributes.copy(),
                         events=sample_span.events)

        result = await processor.redact_item(test_span)

        assert result.attributes["user_id"] == "[CUSTOM_REDACTED]"


class TestSpanHeaderRedactionProcessorIntegration:
    """Test integration scenarios with SpanHeaderRedactionProcessor."""

    @patch('nat.builder.context.Context.get')
    async def test_full_redaction_flow_with_headers(self, mock_context_get, sample_span):
        """Test complete redaction flow with headers and callback."""

        def admin_callback(data: dict[str, Any]) -> bool:
            auth = data.get("authorization", "")
            return "admin" in auth

        processor = SpanHeaderRedactionProcessor(attributes=["user_id", "session_token"],
                                                 headers=["authorization"],
                                                 callback=admin_callback,
                                                 enabled=True,
                                                 force_redact=False)

        headers = Headers({"authorization": "Bearer admin_token_123"})
        metadata = Mock(spec=RequestAttributes)
        metadata.headers = headers
        context = Mock(spec=Context)
        context.metadata = metadata
        mock_context_get.return_value = context

        # Create a copy to avoid mutating the fixture
        test_span = Span(name=sample_span.name,
                         context=sample_span.context,
                         parent=sample_span.parent,
                         start_time=sample_span.start_time,
                         end_time=sample_span.end_time,
                         attributes=sample_span.attributes.copy(),
                         events=sample_span.events)

        result = await processor.process(test_span)

        # Verify redaction occurred
        assert result.attributes["user_id"] == "[REDACTED]"
        assert result.attributes["session_token"] == "[REDACTED]"
        assert result.attributes["api_key"] == "api_secret"  # Not in redaction list
        assert result.attributes["normal_field"] == "normal_value"

    @patch('nat.builder.context.Context.get')
    async def test_no_redaction_flow_with_user_token(self, mock_context_get, sample_span):
        """Test no redaction when callback returns False."""

        def admin_only_callback(data: dict[str, Any]) -> bool:
            auth = data.get("authorization", "")
            return "admin" in auth

        processor = SpanHeaderRedactionProcessor(attributes=["user_id", "session_token"],
                                                 headers=["authorization"],
                                                 callback=admin_only_callback,
                                                 enabled=True,
                                                 force_redact=False)

        headers = Headers({"authorization": "Bearer user_token_123"})  # No "admin"
        metadata = Mock(spec=RequestAttributes)
        metadata.headers = headers
        context = Mock(spec=Context)
        context.metadata = metadata
        mock_context_get.return_value = context

        # Create a copy to avoid mutating the fixture
        test_span = Span(name=sample_span.name,
                         context=sample_span.context,
                         parent=sample_span.parent,
                         start_time=sample_span.start_time,
                         end_time=sample_span.end_time,
                         attributes=sample_span.attributes.copy(),
                         events=sample_span.events)

        original_attributes = dict(test_span.attributes)
        result = await processor.process(test_span)

        # No redaction should have occurred
        assert result.attributes == original_attributes

    @patch('nat.builder.context.Context.get')
    async def test_force_redact_overrides_everything(self, mock_context_get, sample_span):
        """Test that force_redact=True overrides all other conditions."""

        def never_redact_callback(_data: dict[str, Any]) -> bool:
            return False

        processor = SpanHeaderRedactionProcessor(
            attributes=["user_id"],
            headers=["nonexistent_header"],  # Header that doesn't exist
            callback=never_redact_callback,  # Callback that never redacts
            enabled=False,  # Disabled
            force_redact=True  # But force redact is True
        )

        # Context with no headers
        metadata = Mock(spec=RequestAttributes)
        metadata.headers = None
        context = Mock(spec=Context)
        context.metadata = metadata
        mock_context_get.return_value = context

        # Create a copy to avoid mutating the fixture
        test_span = Span(name=sample_span.name,
                         context=sample_span.context,
                         parent=sample_span.parent,
                         start_time=sample_span.start_time,
                         end_time=sample_span.end_time,
                         attributes=sample_span.attributes.copy(),
                         events=sample_span.events)

        result = await processor.process(test_span)

        # Should still redact due to force_redact=True
        assert result.attributes["user_id"] == "[REDACTED]"


class TestSpanHeaderRedactionProcessorTypeIntrospection:
    """Test type introspection capabilities."""

    def test_span_header_redaction_processor_types(self):
        """Test type introspection for span header redaction processor."""
        processor = SpanHeaderRedactionProcessor(headers=[], attributes=[], callback=default_callback)

        assert processor.input_type is Span
        assert processor.output_type is Span

        # Test Pydantic-based validation methods (preferred approach)
        span_context = SpanContext(span_id=123, trace_id=456)
        test_span = Span(name="test", context=span_context)
        assert processor.validate_input_type(test_span)
        assert not processor.validate_input_type("not_a_span")
        assert processor.validate_output_type(test_span)
