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

from unittest.mock import Mock
from unittest.mock import patch

import pytest
from starlette.datastructures import Headers

from nat.builder.context import Context
from nat.data_models.span import Span
from nat.data_models.span import SpanContext
from nat.observability.processor.header_redaction_processor import HeaderRedactionProcessor
from nat.observability.processor.header_redaction_processor import default_callback
from nat.runtime.user_metadata import RequestAttributes


@pytest.fixture
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


@pytest.fixture
def mock_context_with_headers():
    """Create a mock context with headers."""
    headers = Headers({"authorization": "Bearer token123", "x-api-key": "key456"})
    metadata = Mock(spec=RequestAttributes)
    metadata.headers = headers

    context = Mock(spec=Context)
    context.metadata = metadata
    return context


@pytest.fixture
def mock_context_no_headers():
    """Create a mock context without headers."""
    metadata = Mock(spec=RequestAttributes)
    metadata.headers = None

    context = Mock(spec=Context)
    context.metadata = metadata
    return context


class TestHeaderRedactionProcessorInitialization:
    """Test HeaderRedactionProcessor initialization."""

    def test_default_initialization(self):
        """Test default initialization parameters."""
        processor = HeaderRedactionProcessor()

        assert processor.attributes == []
        assert processor.header is None
        assert processor.callback is default_callback
        assert processor.enabled is True
        assert processor.force_redact is False

    def test_initialization_with_attributes(self):
        """Test initialization with custom attributes."""
        attributes = ["user_id", "session_token"]
        processor = HeaderRedactionProcessor(attributes=attributes)

        assert processor.attributes == attributes
        assert processor.header is None
        assert processor.callback is default_callback
        assert processor.enabled is True
        assert processor.force_redact is False

    def test_initialization_with_header(self):
        """Test initialization with custom header."""
        processor = HeaderRedactionProcessor(header="authorization")

        assert processor.attributes == []
        assert processor.header == "authorization"
        assert processor.callback is default_callback
        assert processor.enabled is True
        assert processor.force_redact is False

    def test_initialization_with_callback(self):
        """Test initialization with custom callback."""

        def custom_callback(auth_key: str) -> bool:
            return "admin" in auth_key

        processor = HeaderRedactionProcessor(callback=custom_callback)

        assert processor.attributes == []
        assert processor.header is None
        assert processor.callback is custom_callback
        assert processor.enabled is True
        assert processor.force_redact is False

    def test_initialization_with_enabled_false(self):
        """Test initialization with enabled=False."""
        processor = HeaderRedactionProcessor(enabled=False)

        assert processor.attributes == []
        assert processor.header is None
        assert processor.callback is default_callback
        assert processor.enabled is False
        assert processor.force_redact is False

    def test_initialization_with_force_redact_true(self):
        """Test initialization with force_redact=True."""
        processor = HeaderRedactionProcessor(force_redact=True)

        assert processor.attributes == []
        assert processor.header is None
        assert processor.callback is default_callback
        assert processor.enabled is True
        assert processor.force_redact is True

    def test_initialization_with_all_parameters(self):
        """Test initialization with all parameters specified."""
        attributes = ["user_id", "api_key"]

        def callback(auth_key: str) -> bool:
            return len(auth_key) > 10

        processor = HeaderRedactionProcessor(attributes=attributes,
                                             header="x-api-key",
                                             callback=callback,
                                             enabled=False,
                                             force_redact=True)

        assert processor.attributes == attributes
        assert processor.header == "x-api-key"
        assert processor.callback is callback
        assert processor.enabled is False
        assert processor.force_redact is True


class TestHeaderRedactionProcessorShouldRedact:
    """Test should_redact method of HeaderRedactionProcessor."""

    def test_should_redact_with_force_redact_true_always_returns_true(self):
        """Test that force_redact=True always returns True regardless of other conditions."""
        processor = HeaderRedactionProcessor(
            enabled=False,  # Even with enabled=False
            header=None,  # Even with no header
            force_redact=True)

        # Create context without headers
        context = Mock(spec=Context)
        metadata = Mock(spec=RequestAttributes)
        metadata.headers = None
        context.metadata = metadata

        span = Mock(spec=Span)

        result = processor.should_redact(span, context)
        assert result is True

    def test_should_redact_with_enabled_false_returns_false(self):
        """Test that enabled=False returns False (unless force_redact=True)."""
        processor = HeaderRedactionProcessor(enabled=False, header="authorization", force_redact=False)

        # Create context with headers
        headers = Headers({"authorization": "Bearer token123"})
        metadata = Mock(spec=RequestAttributes)
        metadata.headers = headers
        context = Mock(spec=Context)
        context.metadata = metadata

        span = Mock(spec=Span)

        result = processor.should_redact(span, context)
        assert result is False

    def test_should_redact_with_no_headers_returns_false(self, mock_context_no_headers):
        """Test that missing headers returns False."""
        processor = HeaderRedactionProcessor(header="authorization", enabled=True, force_redact=False)

        span = Mock(spec=Span)

        result = processor.should_redact(span, mock_context_no_headers)
        assert result is False

    def test_should_redact_with_no_header_key_returns_false(self, mock_context_with_headers):
        """Test that missing header key returns False."""
        processor = HeaderRedactionProcessor(
            header=None,  # No header specified
            enabled=True,
            force_redact=False)

        span = Mock(spec=Span)

        result = processor.should_redact(span, mock_context_with_headers)
        assert result is False

    def test_should_redact_with_missing_header_value_returns_false(self):
        """Test that missing header value returns False."""
        processor = HeaderRedactionProcessor(
            header="missing-header",  # Header not in the Headers
            enabled=True,
            force_redact=False)

        headers = Headers({"authorization": "Bearer token123"})
        metadata = Mock(spec=RequestAttributes)
        metadata.headers = headers
        context = Mock(spec=Context)
        context.metadata = metadata

        span = Mock(spec=Span)

        result = processor.should_redact(span, context)
        assert result is False

    def test_should_redact_with_empty_header_value_returns_false(self):
        """Test that empty header value returns False."""
        processor = HeaderRedactionProcessor(header="authorization", enabled=True, force_redact=False)

        headers = Headers({"authorization": ""})  # Empty value
        metadata = Mock(spec=RequestAttributes)
        metadata.headers = headers
        context = Mock(spec=Context)
        context.metadata = metadata

        span = Mock(spec=Span)

        result = processor.should_redact(span, context)
        assert result is False

    def test_should_redact_with_callback_returning_true(self):
        """Test that callback returning True results in redaction."""

        def always_redact_callback(auth_key: str) -> bool:
            return True

        processor = HeaderRedactionProcessor(header="authorization",
                                             callback=always_redact_callback,
                                             enabled=True,
                                             force_redact=False)

        headers = Headers({"authorization": "Bearer token123"})
        metadata = Mock(spec=RequestAttributes)
        metadata.headers = headers
        context = Mock(spec=Context)
        context.metadata = metadata

        span = Mock(spec=Span)

        result = processor.should_redact(span, context)
        assert result is True

    def test_should_redact_with_callback_returning_false(self):
        """Test that callback returning False results in no redaction."""

        def never_redact_callback(auth_key: str) -> bool:
            return False

        processor = HeaderRedactionProcessor(header="authorization",
                                             callback=never_redact_callback,
                                             enabled=True,
                                             force_redact=False)

        headers = Headers({"authorization": "Bearer token123"})
        metadata = Mock(spec=RequestAttributes)
        metadata.headers = headers
        context = Mock(spec=Context)
        context.metadata = metadata

        span = Mock(spec=Span)

        result = processor.should_redact(span, context)
        assert result is False

    def test_should_redact_with_conditional_callback(self):
        """Test callback with conditional logic."""

        def conditional_callback(auth_key: str) -> bool:
            return "admin" in auth_key or len(auth_key) > 20

        processor = HeaderRedactionProcessor(header="x-api-key",
                                             callback=conditional_callback,
                                             enabled=True,
                                             force_redact=False)

        context = Mock(spec=Context)
        metadata = Mock(spec=RequestAttributes)
        context.metadata = metadata
        span = Mock(spec=Span)

        # Test with admin key (should redact)
        headers1 = Headers({"x-api-key": "admin_key_123"})
        metadata.headers = headers1
        result1 = processor.should_redact(span, context)
        assert result1 is True

        # Test with long key (should redact)
        headers2 = Headers({"x-api-key": "very_long_api_key_that_exceeds_twenty_chars"})
        metadata.headers = headers2
        result2 = processor.should_redact(span, context)
        assert result2 is True

        # Test with short non-admin key (should not redact)
        headers3 = Headers({"x-api-key": "short_key"})
        metadata.headers = headers3
        result3 = processor.should_redact(span, context)
        assert result3 is False


class TestHeaderRedactionProcessorRedactItem:
    """Test redact_item method of HeaderRedactionProcessor."""

    def test_redact_item_with_single_attribute(self, sample_span):
        """Test redacting a single attribute."""
        processor = HeaderRedactionProcessor(attributes=["user_id"])

        # Create a copy to avoid mutating the fixture
        test_span = Span(name=sample_span.name,
                         context=sample_span.context,
                         parent=sample_span.parent,
                         start_time=sample_span.start_time,
                         end_time=sample_span.end_time,
                         attributes=sample_span.attributes.copy(),
                         events=sample_span.events)

        result = processor.redact_item(test_span)

        assert result.attributes["user_id"] == "[REDACTED]"
        assert result.attributes["session_token"] == "secret_token"  # Unchanged
        assert result.attributes["api_key"] == "api_secret"  # Unchanged
        assert result.attributes["normal_field"] == "normal_value"  # Unchanged

    def test_redact_item_with_multiple_attributes(self, sample_span):
        """Test redacting multiple attributes."""
        processor = HeaderRedactionProcessor(attributes=["user_id", "session_token", "api_key"])

        # Create a copy to avoid mutating the fixture
        test_span = Span(name=sample_span.name,
                         context=sample_span.context,
                         parent=sample_span.parent,
                         start_time=sample_span.start_time,
                         end_time=sample_span.end_time,
                         attributes=sample_span.attributes.copy(),
                         events=sample_span.events)

        result = processor.redact_item(test_span)

        assert result.attributes["user_id"] == "[REDACTED]"
        assert result.attributes["session_token"] == "[REDACTED]"
        assert result.attributes["api_key"] == "[REDACTED]"
        assert result.attributes["normal_field"] == "normal_value"  # Unchanged

    def test_redact_item_with_missing_attributes(self, sample_span):
        """Test redacting attributes that don't exist in the span."""
        processor = HeaderRedactionProcessor(attributes=["nonexistent_field", "user_id"])

        # Create a copy to avoid mutating the fixture
        test_span = Span(name=sample_span.name,
                         context=sample_span.context,
                         parent=sample_span.parent,
                         start_time=sample_span.start_time,
                         end_time=sample_span.end_time,
                         attributes=sample_span.attributes.copy(),
                         events=sample_span.events)

        result = processor.redact_item(test_span)

        # Should redact existing field
        assert result.attributes["user_id"] == "[REDACTED]"
        # Should not create nonexistent field
        assert "nonexistent_field" not in result.attributes
        # Other fields should be unchanged
        assert result.attributes["session_token"] == "secret_token"

    def test_redact_item_with_empty_attributes_list(self, sample_span):
        """Test redacting with empty attributes list."""
        processor = HeaderRedactionProcessor(attributes=[])

        # Create a copy to avoid mutating the fixture
        test_span = Span(name=sample_span.name,
                         context=sample_span.context,
                         parent=sample_span.parent,
                         start_time=sample_span.start_time,
                         end_time=sample_span.end_time,
                         attributes=sample_span.attributes.copy(),
                         events=sample_span.events)

        original_attributes = test_span.attributes.copy()
        result = processor.redact_item(test_span)

        # No attributes should be changed
        assert result.attributes == original_attributes

    def test_redact_item_modifies_original_span(self, sample_span):
        """Test that redact_item modifies the original span object."""
        processor = HeaderRedactionProcessor(attributes=["user_id"])

        # Create a copy to avoid mutating the fixture
        test_span = Span(name=sample_span.name,
                         context=sample_span.context,
                         parent=sample_span.parent,
                         start_time=sample_span.start_time,
                         end_time=sample_span.end_time,
                         attributes=sample_span.attributes.copy(),
                         events=sample_span.events)

        original_user_id = test_span.attributes["user_id"]
        result = processor.redact_item(test_span)

        # Should return the same span object
        assert result is test_span
        # Should have modified the original
        assert test_span.attributes["user_id"] == "[REDACTED]"
        assert original_user_id != "[REDACTED]"

    def test_redact_item_preserves_other_span_fields(self, sample_span):
        """Test that redact_item preserves all non-attribute fields."""
        processor = HeaderRedactionProcessor(attributes=["user_id"])

        # Create a copy to avoid mutating the fixture
        test_span = Span(name=sample_span.name,
                         context=sample_span.context,
                         parent=sample_span.parent,
                         start_time=sample_span.start_time,
                         end_time=sample_span.end_time,
                         attributes=sample_span.attributes.copy(),
                         events=sample_span.events)

        original_name = test_span.name
        original_context = test_span.context
        original_start_time = test_span.start_time
        original_end_time = test_span.end_time
        original_events = test_span.events

        result = processor.redact_item(test_span)

        # All non-attribute fields should be unchanged
        assert result.name == original_name
        assert result.context == original_context
        assert result.start_time == original_start_time
        assert result.end_time == original_end_time
        assert result.events == original_events


class TestHeaderRedactionProcessorLRUCache:
    """Test LRU cache behavior in HeaderRedactionProcessor."""

    def test_lru_cache_avoids_redundant_callback_calls(self):
        """Test that LRU cache avoids redundant callback executions."""
        call_count = 0

        def counting_callback(auth_key: str) -> bool:
            nonlocal call_count
            call_count += 1
            return "admin" in auth_key

        processor = HeaderRedactionProcessor(header="authorization",
                                             callback=counting_callback,
                                             enabled=True,
                                             force_redact=False)

        # Call _should_redact_impl multiple times with same auth_key
        result1 = processor._should_redact_impl("admin_token")
        result2 = processor._should_redact_impl("admin_token")
        result3 = processor._should_redact_impl("admin_token")

        assert result1 is True
        assert result2 is True
        assert result3 is True
        # Callback should only be called once due to caching
        assert call_count == 1

        # Call with different auth_key
        result4 = processor._should_redact_impl("user_token")
        assert result4 is False
        # Should call callback again for new key
        assert call_count == 2

    def test_lru_cache_respects_different_auth_keys(self):
        """Test that LRU cache properly handles different auth keys."""

        def selective_callback(auth_key: str) -> bool:
            return auth_key.startswith("admin")

        processor = HeaderRedactionProcessor(header="authorization",
                                             callback=selective_callback,
                                             enabled=True,
                                             force_redact=False)

        # Test different auth keys
        assert processor._should_redact_impl("admin_key1") is True
        assert processor._should_redact_impl("user_key1") is False
        assert processor._should_redact_impl("admin_key2") is True
        assert processor._should_redact_impl("user_key2") is False

        # Calling same keys again should use cache
        assert processor._should_redact_impl("admin_key1") is True
        assert processor._should_redact_impl("user_key1") is False

    def test_lru_cache_info_accessible(self):
        """Test that LRU cache info is accessible for monitoring."""
        processor = HeaderRedactionProcessor(callback=lambda x: True)

        # Clear the cache to start fresh (LRU cache is shared across instances)
        processor._should_redact_impl.cache_clear()

        # Check initial cache info after clearing
        cache_info = processor._should_redact_impl.cache_info()
        assert cache_info.hits == 0
        assert cache_info.misses == 0
        assert cache_info.maxsize == 128  # Default maxsize

        # Make some calls
        processor._should_redact_impl("key1")
        processor._should_redact_impl("key2")
        processor._should_redact_impl("key1")  # Cache hit

        cache_info = processor._should_redact_impl.cache_info()
        assert cache_info.hits == 1
        assert cache_info.misses == 2


class TestHeaderRedactionProcessorIntegration:
    """Test integration scenarios with HeaderRedactionProcessor."""

    @patch('nat.observability.processor.header_redaction_processor.Context.get')
    async def test_full_redaction_flow_with_headers(self, mock_context_get, sample_span):
        """Test complete redaction flow with headers and callback."""

        def admin_callback(auth_key: str) -> bool:
            return "admin" in auth_key

        processor = HeaderRedactionProcessor(attributes=["user_id", "session_token"],
                                             header="authorization",
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

    @patch('nat.observability.processor.header_redaction_processor.Context.get')
    async def test_no_redaction_flow_with_user_token(self, mock_context_get, sample_span):
        """Test no redaction when callback returns False."""

        def admin_only_callback(auth_key: str) -> bool:
            return "admin" in auth_key

        processor = HeaderRedactionProcessor(attributes=["user_id", "session_token"],
                                             header="authorization",
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

        original_attributes = test_span.attributes.copy()
        result = await processor.process(test_span)

        # No redaction should have occurred
        assert result.attributes == original_attributes

    @patch('nat.observability.processor.header_redaction_processor.Context.get')
    async def test_disabled_processor_never_redacts(self, mock_context_get, sample_span):
        """Test that disabled processor never redacts."""

        def always_redact_callback(auth_key: str) -> bool:
            return True

        processor = HeaderRedactionProcessor(
            attributes=["user_id"],
            header="authorization",
            callback=always_redact_callback,
            enabled=False,  # Disabled
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

        original_attributes = test_span.attributes.copy()
        result = await processor.process(test_span)

        # No redaction should have occurred
        assert result.attributes == original_attributes

    @patch('nat.observability.processor.header_redaction_processor.Context.get')
    async def test_force_redact_overrides_everything(self, mock_context_get, sample_span):
        """Test that force_redact=True overrides all other conditions."""

        def never_redact_callback(auth_key: str) -> bool:
            return False

        processor = HeaderRedactionProcessor(
            attributes=["user_id"],
            header="nonexistent_header",  # Header that doesn't exist
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


class TestHeaderRedactionProcessorErrorHandling:
    """Test error handling in HeaderRedactionProcessor."""

    def test_should_redact_with_callback_error_propagates(self):
        """Test that callback errors are propagated."""

        def failing_callback(auth_key: str) -> bool:
            raise ValueError("Callback failed")

        processor = HeaderRedactionProcessor(header="authorization",
                                             callback=failing_callback,
                                             enabled=True,
                                             force_redact=False)

        headers = Headers({"authorization": "Bearer token123"})
        metadata = Mock(spec=RequestAttributes)
        metadata.headers = headers
        context = Mock(spec=Context)
        context.metadata = metadata

        span = Mock(spec=Span)

        with pytest.raises(ValueError, match="Callback failed"):
            processor.should_redact(span, context)

    def test_should_redact_with_context_metadata_error(self):
        """Test error handling when context.metadata access fails."""
        processor = HeaderRedactionProcessor(header="authorization", enabled=True, force_redact=False)

        # Create context that raises error when accessing metadata
        context = Mock(spec=Context)

        # Create a metadata mock that raises error when headers attribute is accessed
        class ErrorMetadata:

            @property
            def headers(self):
                raise AttributeError("Metadata access failed")

        context.metadata = ErrorMetadata()

        span = Mock(spec=Span)

        with pytest.raises(AttributeError, match="Metadata access failed"):
            processor.should_redact(span, context)

    def test_redact_item_with_attribute_access_error(self):
        """Test error handling when attribute access fails."""
        processor = HeaderRedactionProcessor(attributes=["user_id"])

        # Create span with attributes that raise error
        span = Mock(spec=Span)
        span.attributes = Mock()
        span.attributes.__contains__ = Mock(side_effect=RuntimeError("Attribute access failed"))

        with pytest.raises(RuntimeError, match="Attribute access failed"):
            processor.redact_item(span)


class TestHeaderRedactionProcessorDefaultCallback:
    """Test the default callback functionality."""

    def test_default_callback_always_returns_false(self):
        """Test that default_callback always returns False."""
        assert default_callback("any_key") is False
        assert default_callback("admin_key") is False
        assert default_callback("") is False
        assert default_callback("very_long_key_with_special_chars!@#") is False

    def test_default_callback_used_when_none_provided(self):
        """Test that default callback is used when none provided."""
        processor = HeaderRedactionProcessor()
        assert processor.callback is default_callback

        processor2 = HeaderRedactionProcessor(callback=None)
        assert processor2.callback is default_callback


class TestHeaderRedactionProcessorTypeIntrospection:
    """Test type introspection capabilities."""

    def test_header_redaction_processor_types(self):
        """Test type introspection for header redaction processor."""
        processor = HeaderRedactionProcessor()

        assert processor.input_type is Span
        assert processor.output_type is Span
        assert processor.input_class is Span
        assert processor.output_class is Span


class TestHeaderRedactionProcessorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_should_redact_with_case_sensitive_headers(self):
        """Test header matching is case-insensitive (Starlette Headers behavior)."""
        processor = HeaderRedactionProcessor(
            header="Authorization",  # Capital A
            callback=lambda x: True,
            enabled=True,
            force_redact=False)

        # Headers with lowercase key
        headers = Headers({"authorization": "Bearer token123"})  # lowercase
        metadata = Mock(spec=RequestAttributes)
        metadata.headers = headers
        context = Mock(spec=Context)
        context.metadata = metadata

        span = Mock(spec=Span)

        # Should match despite case difference (Starlette Headers are case-insensitive)
        result = processor.should_redact(span, context)
        assert result is True

    def test_should_redact_with_multiple_header_values(self):
        """Test behavior with multiple header values."""
        processor = HeaderRedactionProcessor(header="authorization",
                                             callback=lambda x: "token1" in x,
                                             enabled=True,
                                             force_redact=False)

        # Headers.get() returns the first value for duplicate keys
        headers = Headers({"authorization": "Bearer token1"})
        metadata = Mock(spec=RequestAttributes)
        metadata.headers = headers
        context = Mock(spec=Context)
        context.metadata = metadata

        span = Mock(spec=Span)

        result = processor.should_redact(span, context)
        # Should use first header value which contains "token1"
        assert result is True

    def test_redact_item_with_none_attribute_values(self):
        """Test redacting attributes with None values."""
        processor = HeaderRedactionProcessor(attributes=["nullable_field"])

        span = Span(name="test", attributes={"nullable_field": None, "other_field": "value"}, events=[])

        result = processor.redact_item(span)

        # Should redact even None values
        assert result.attributes["nullable_field"] == "[REDACTED]"
        assert result.attributes["other_field"] == "value"

    def test_redact_item_with_non_string_attribute_values(self):
        """Test redacting non-string attribute values."""
        processor = HeaderRedactionProcessor(attributes=["numeric_field", "bool_field", "list_field"])

        span = Span(name="test",
                    attributes={
                        "numeric_field": 12345, "bool_field": True, "list_field": [1, 2, 3], "other_field": "preserve"
                    },
                    events=[])

        result = processor.redact_item(span)

        # Should redact all specified attributes regardless of type
        assert result.attributes["numeric_field"] == "[REDACTED]"
        assert result.attributes["bool_field"] == "[REDACTED]"
        assert result.attributes["list_field"] == "[REDACTED]"
        assert result.attributes["other_field"] == "preserve"


class TestHeaderRedactionProcessorPerformance:
    """Test performance-related aspects."""

    def test_should_redact_early_returns(self):
        """Test that should_redact has early return optimizations."""
        call_count = 0

        def counting_callback(auth_key: str) -> bool:
            nonlocal call_count
            call_count += 1
            return True

        # Test early return for force_redact=True
        processor = HeaderRedactionProcessor(
            callback=counting_callback,
            force_redact=True,
            enabled=False  # This should be ignored due to force_redact
        )

        context = Mock(spec=Context)
        span = Mock(spec=Span)

        result = processor.should_redact(span, context)
        assert result is True
        # Callback should not have been called due to early return
        assert call_count == 0

        # Test early return for enabled=False
        processor2 = HeaderRedactionProcessor(callback=counting_callback, force_redact=False, enabled=False)

        result2 = processor2.should_redact(span, context)
        assert result2 is False
        # Callback should still not have been called
        assert call_count == 0

    def test_redact_item_efficiency_with_no_attributes(self, sample_span):
        """Test that redact_item is efficient when no attributes specified."""
        processor = HeaderRedactionProcessor(attributes=[])

        # Create a copy to avoid mutating the fixture
        test_span = Span(name=sample_span.name,
                         context=sample_span.context,
                         parent=sample_span.parent,
                         start_time=sample_span.start_time,
                         end_time=sample_span.end_time,
                         attributes=sample_span.attributes.copy(),
                         events=sample_span.events)

        original_attributes = test_span.attributes.copy()
        result = processor.redact_item(test_span)

        # Should return same object
        assert result is test_span
        # Attributes should be completely unchanged
        assert result.attributes == original_attributes
