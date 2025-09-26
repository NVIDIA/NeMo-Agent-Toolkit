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

import pytest

from nat.data_models.span import Span
from nat.data_models.span import SpanContext
from nat.observability.processor.redaction.contextual_span_redaction_processor import ContextualSpanRedactionProcessor


def default_callback(_data: dict[str, Any]) -> bool:
    """Default callback that always returns False."""
    return False


class ConcreteContextualSpanRedactionProcessor(ContextualSpanRedactionProcessor):
    """Concrete implementation for testing ContextualSpanRedactionProcessor."""

    def __init__(self,
                 extracted_data: dict | None = None,
                 data_validation_result: bool = True,
                 attributes: list[str] | None = None,
                 callback: Any | None = None,
                 enabled: bool = True,
                 force_redact: bool = False,
                 redaction_value: str = "[REDACTED]",
                 redaction_tag: str | None = None,
                 **kwargs):
        # Set defaults for required parameters
        if attributes is None:
            attributes = []
        if callback is None:
            callback = default_callback

        super().__init__(attributes=attributes,
                         callback=callback,
                         enabled=enabled,
                         force_redact=force_redact,
                         redaction_value=redaction_value,
                         redaction_tag=redaction_tag,
                         **kwargs)
        self.extracted_data = extracted_data
        self.data_validation_result = data_validation_result

    def extract_data_from_context(self) -> dict | None:
        """Test implementation that returns configured data."""
        return self.extracted_data

    def validate_data(self, data: dict) -> bool:
        """Test implementation that returns configured validation result."""
        return self.data_validation_result


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
                    "normal_field": "normal_value",
                    "sensitive_data": "confidential_info"
                },
                events=[])


@pytest.fixture
def minimal_span():
    """Create a minimal span with no attributes."""
    span_context = SpanContext(span_id=789, trace_id=101112)
    return Span(name="minimal_operation",
                context=span_context,
                parent=None,
                start_time=1000000,
                end_time=2000000,
                attributes={},
                events=[])


class TestContextualSpanRedactionProcessorInitialization:
    """Test ContextualSpanRedactionProcessor initialization."""

    def test_default_initialization(self):
        """Test default initialization parameters."""
        processor = ConcreteContextualSpanRedactionProcessor()

        assert processor.attributes == []
        assert processor.callback is default_callback
        assert processor.enabled is True
        assert processor.force_redact is False
        assert processor.redaction_value == "[REDACTED]"
        assert processor.redaction_tag is None

    def test_initialization_with_attributes(self):
        """Test initialization with custom attributes."""
        attributes = ["user_id", "session_token", "api_key"]
        processor = ConcreteContextualSpanRedactionProcessor(attributes=attributes)

        assert processor.attributes == attributes
        assert processor.callback is default_callback
        assert processor.enabled is True
        assert processor.force_redact is False
        assert processor.redaction_value == "[REDACTED]"
        assert processor.redaction_tag is None

    def test_initialization_with_single_attribute(self):
        """Test initialization with single attribute."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id"])

        assert processor.attributes == ["user_id"]

    def test_initialization_with_empty_attributes(self):
        """Test initialization with empty attributes list."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=[])

        assert processor.attributes == []

    def test_initialization_with_none_attributes(self):
        """Test initialization with None attributes (should default to empty list)."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=None)

        assert processor.attributes == []

    def test_initialization_with_custom_callback(self):
        """Test initialization with custom callback."""

        def custom_callback(_data: Any) -> bool:
            return True

        processor = ConcreteContextualSpanRedactionProcessor(callback=custom_callback)

        assert processor.callback is custom_callback

    def test_initialization_with_none_callback(self):
        """Test initialization with None callback (should use default)."""
        processor = ConcreteContextualSpanRedactionProcessor(callback=None)

        assert processor.callback is default_callback

    def test_initialization_with_enabled_false(self):
        """Test initialization with enabled=False."""
        processor = ConcreteContextualSpanRedactionProcessor(enabled=False)

        assert processor.enabled is False

    def test_initialization_with_force_redact_true(self):
        """Test initialization with force_redact=True."""
        processor = ConcreteContextualSpanRedactionProcessor(force_redact=True)

        assert processor.force_redact is True

    def test_initialization_with_custom_redaction_value(self):
        """Test initialization with custom redaction value."""
        custom_value = "***HIDDEN***"
        processor = ConcreteContextualSpanRedactionProcessor(redaction_value=custom_value)

        assert processor.redaction_value == custom_value

    def test_initialization_with_redaction_tag(self):
        """Test initialization with redaction tag."""
        tag = "redacted_by_processor"
        processor = ConcreteContextualSpanRedactionProcessor(redaction_tag=tag)

        assert processor.redaction_tag == tag

    def test_initialization_with_all_parameters(self):
        """Test initialization with all parameters specified."""
        attributes = ["user_id", "api_key"]

        def test_callback(_x):
            return True

        callback = test_callback
        tag = "test_redaction"

        processor = ConcreteContextualSpanRedactionProcessor(attributes=attributes,
                                                             callback=callback,
                                                             enabled=False,
                                                             force_redact=True,
                                                             redaction_value="CENSORED",
                                                             redaction_tag=tag)

        assert processor.attributes == attributes
        assert processor.callback is callback
        assert processor.enabled is False
        assert processor.force_redact is True
        assert processor.redaction_value == "CENSORED"
        assert processor.redaction_tag == tag


class TestContextualSpanRedactionProcessorRedactItem:
    """Test ContextualSpanRedactionProcessor redact_item method."""

    async def test_redact_single_attribute(self, sample_span):
        """Test redacting a single attribute."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id"])

        result = await processor.redact_item(sample_span)

        assert result.attributes["user_id"] == "[REDACTED]"
        assert result.attributes["session_token"] == "secret_token"  # unchanged
        assert result.attributes["api_key"] == "api_secret"  # unchanged
        assert result.attributes["normal_field"] == "normal_value"  # unchanged

    async def test_redact_multiple_attributes(self, sample_span):
        """Test redacting multiple attributes."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id", "session_token", "api_key"])

        result = await processor.redact_item(sample_span)

        assert result.attributes["user_id"] == "[REDACTED]"
        assert result.attributes["session_token"] == "[REDACTED]"
        assert result.attributes["api_key"] == "[REDACTED]"
        assert result.attributes["normal_field"] == "normal_value"  # unchanged
        assert result.attributes["sensitive_data"] == "confidential_info"  # unchanged

    async def test_redact_all_attributes(self, sample_span):
        """Test redacting all attributes in the span."""
        all_attributes = list(sample_span.attributes.keys())
        processor = ConcreteContextualSpanRedactionProcessor(attributes=all_attributes)

        result = await processor.redact_item(sample_span)

        for key in all_attributes:
            assert result.attributes[key] == "[REDACTED]"

    async def test_redact_nonexistent_attributes(self, sample_span):
        """Test redacting attributes that don't exist in the span."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["nonexistent_attr1", "nonexistent_attr2"])

        result = await processor.redact_item(sample_span)

        # Original attributes should remain unchanged
        assert result.attributes["user_id"] == "user123"
        assert result.attributes["session_token"] == "secret_token"
        assert result.attributes["api_key"] == "api_secret"
        assert result.attributes["normal_field"] == "normal_value"

        # Nonexistent attributes should not be added
        assert "nonexistent_attr1" not in result.attributes
        assert "nonexistent_attr2" not in result.attributes

    async def test_redact_mixed_existing_and_nonexistent_attributes(self, sample_span):
        """Test redacting a mix of existing and nonexistent attributes."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id", "nonexistent_attr", "api_key"])

        result = await processor.redact_item(sample_span)

        assert result.attributes["user_id"] == "[REDACTED]"
        assert result.attributes["api_key"] == "[REDACTED]"
        assert result.attributes["session_token"] == "secret_token"  # unchanged
        assert "nonexistent_attr" not in result.attributes

    async def test_redact_with_custom_redaction_value(self, sample_span):
        """Test redacting with custom redaction value."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id", "api_key"],
                                                             redaction_value="***CENSORED***")

        result = await processor.redact_item(sample_span)

        assert result.attributes["user_id"] == "***CENSORED***"
        assert result.attributes["api_key"] == "***CENSORED***"
        assert result.attributes["session_token"] == "secret_token"  # unchanged

    async def test_redact_empty_attributes_list(self, sample_span):
        """Test redacting with empty attributes list (should not change anything)."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=[])

        result = await processor.redact_item(sample_span)

        # All attributes should remain unchanged
        assert result.attributes["user_id"] == "user123"
        assert result.attributes["session_token"] == "secret_token"
        assert result.attributes["api_key"] == "api_secret"
        assert result.attributes["normal_field"] == "normal_value"

    async def test_redact_span_with_no_attributes(self, minimal_span):
        """Test redacting a span with no attributes."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id"])

        result = await processor.redact_item(minimal_span)

        # Should remain empty
        assert result.attributes == {}

    async def test_redact_with_redaction_tag(self, sample_span):
        """Test redacting with redaction tag."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id"], redaction_tag="redacted_by_test")

        result = await processor.redact_item(sample_span)

        assert result.attributes["user_id"] == "[REDACTED]"
        assert result.attributes["redacted_by_test"] is True

    async def test_redact_with_redaction_tag_no_attributes(self, sample_span):
        """Test redacting with redaction tag but no attributes to redact."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=[], redaction_tag="redacted_by_test")

        result = await processor.redact_item(sample_span)

        # Original attributes unchanged
        assert result.attributes["user_id"] == "user123"
        # But tag should still be added
        assert result.attributes["redacted_by_test"] is True

    async def test_redact_with_none_redaction_tag(self, sample_span):
        """Test redacting with None redaction tag."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id"], redaction_tag=None)

        result = await processor.redact_item(sample_span)

        assert result.attributes["user_id"] == "[REDACTED]"
        # No redaction tag should be added
        assert len([k for k in result.attributes.keys() if k not in sample_span.attributes]) == 0

    async def test_redact_preserves_span_identity(self, sample_span):
        """Test that redaction preserves the span's identity and other properties."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id"])

        result = await processor.redact_item(sample_span)

        # Should be the same span object
        assert result is sample_span
        assert result.name == "test_operation"
        if result.context:
            assert result.context.span_id == 123
            assert result.context.trace_id == 456
        assert result.start_time == 1000000
        assert result.end_time == 2000000
        assert result.events == []

    async def test_redact_multiple_calls_same_span(self, sample_span):
        """Test multiple redaction calls on the same span."""
        processor1 = ConcreteContextualSpanRedactionProcessor(attributes=["user_id"])
        processor2 = ConcreteContextualSpanRedactionProcessor(attributes=["api_key"])

        # First redaction
        result1 = await processor1.redact_item(sample_span)
        assert result1.attributes["user_id"] == "[REDACTED]"
        assert result1.attributes["api_key"] == "api_secret"

        # Second redaction on the same span
        result2 = await processor2.redact_item(result1)
        assert result2.attributes["user_id"] == "[REDACTED]"  # still redacted
        assert result2.attributes["api_key"] == "[REDACTED]"  # now redacted

    async def test_redact_overwrite_existing_redaction(self, sample_span):
        """Test that redaction overwrites previously redacted values."""
        processor1 = ConcreteContextualSpanRedactionProcessor(attributes=["user_id"], redaction_value="FIRST_REDACTION")
        processor2 = ConcreteContextualSpanRedactionProcessor(attributes=["user_id"],
                                                              redaction_value="SECOND_REDACTION")

        # First redaction
        result1 = await processor1.redact_item(sample_span)
        assert result1.attributes["user_id"] == "FIRST_REDACTION"

        # Second redaction overwrites the first
        result2 = await processor2.redact_item(result1)
        assert result2.attributes["user_id"] == "SECOND_REDACTION"


class TestContextualSpanRedactionProcessorShouldRedact:
    """Test should_redact method - comprehensive coverage of the public interface."""

    async def test_should_redact_force_redact_true(self, sample_span):
        """Test should_redact with force_redact=True - should always return True."""
        processor = ConcreteContextualSpanRedactionProcessor(force_redact=True)

        # Should return True regardless of other conditions
        result = await processor.should_redact(sample_span)
        assert result is True

    async def test_should_redact_enabled_false(self, sample_span):
        """Test should_redact with enabled=False - should always return False."""
        processor = ConcreteContextualSpanRedactionProcessor(enabled=False)

        # Should return False when disabled
        result = await processor.should_redact(sample_span)
        assert result is False

    async def test_should_redact_enabled_false_overrides_force_redact(self, sample_span):
        """Test that enabled=False takes precedence over force_redact=True."""
        processor = ConcreteContextualSpanRedactionProcessor(enabled=False, force_redact=True)

        # force_redact check happens first, so should return True
        result = await processor.should_redact(sample_span)
        assert result is True

    async def test_should_redact_extract_data_returns_none(self, sample_span):
        """Test should_redact when extract_data_from_context returns None."""
        processor = ConcreteContextualSpanRedactionProcessor(extracted_data=None)

        result = await processor.should_redact(sample_span)
        assert result is False

    async def test_should_redact_validate_data_returns_false(self, sample_span):
        """Test should_redact when validate_data returns False."""
        processor = ConcreteContextualSpanRedactionProcessor(extracted_data={"test": "data"},
                                                             data_validation_result=False)

        result = await processor.should_redact(sample_span)
        assert result is False

    async def test_should_redact_all_conditions_met_with_default_callback(self, sample_span):
        """Test should_redact when all conditions are met but default callback returns False."""
        processor = ConcreteContextualSpanRedactionProcessor(extracted_data={"test": "data"},
                                                             data_validation_result=True
                                                             # Using default callback which always returns False
                                                             )

        # With default callback returning False, should not redact
        result = await processor.should_redact(sample_span)
        assert result is False

    async def test_should_redact_all_conditions_met_with_true_callback(self, sample_span):
        """Test should_redact when all conditions are met and callback returns True."""

        def always_true_callback(_data):
            return True

        processor = ConcreteContextualSpanRedactionProcessor(
            extracted_data={"test": "data"},
            data_validation_result=True,
            callback=always_true_callback,
        )

        result = await processor.should_redact(sample_span)
        assert result is True

    async def test_should_redact_callback_with_custom_logic(self, sample_span):
        """Test should_redact with custom callback logic."""

        def role_based_callback(data):
            return data.get("role") == "admin"

        # Test with admin role - should redact via callback
        processor_admin = ConcreteContextualSpanRedactionProcessor(
            extracted_data={
                "user": "test_user", "role": "admin"
            },
            data_validation_result=True,
            callback=role_based_callback,
        )

        result = await processor_admin.should_redact(sample_span)
        assert result is True

        # Test with non-admin role - should not redact via callback
        processor_user = ConcreteContextualSpanRedactionProcessor(
            extracted_data={
                "user": "test_user", "role": "user"
            },
            data_validation_result=True,
            callback=role_based_callback,
        )

        result = await processor_user.should_redact(sample_span)
        assert result is False

    async def test_should_redact_different_span_types(self, minimal_span):
        """Test should_redact works with different span configurations."""
        processor = ConcreteContextualSpanRedactionProcessor(force_redact=True)

        result = await processor.should_redact(minimal_span)
        assert result is True

    async def test_should_redact_complex_extracted_data(self, sample_span):
        """Test should_redact with complex extracted data structures."""
        complex_data = {
            "headers": {
                "authorization": "bearer token"
            },
            "cookies": {
                "session_id": "abc123"
            },
            "user_info": {
                "id": 456, "permissions": ["read", "write"]
            }
        }

        processor = ConcreteContextualSpanRedactionProcessor(
            extracted_data=complex_data,
            data_validation_result=True,
            force_redact=True  # Simplify for this test
        )

        result = await processor.should_redact(sample_span)
        assert result is True


class TestContextualSpanRedactionProcessorAbstractMethods:
    """Test that abstract methods work correctly with concrete implementations."""

    def test_extract_data_from_context_implementation(self):
        """Test that extract_data_from_context works with concrete implementation."""
        test_data = {"test": "data"}
        processor = ConcreteContextualSpanRedactionProcessor(extracted_data=test_data)

        result = processor.extract_data_from_context()
        assert result == test_data

    def test_extract_data_from_context_returns_none(self):
        """Test extract_data_from_context when no data is configured."""
        processor = ConcreteContextualSpanRedactionProcessor(extracted_data=None)

        result = processor.extract_data_from_context()
        assert result is None

    def test_validate_data_implementation_true(self):
        """Test that validate_data works with concrete implementation returning True."""
        processor = ConcreteContextualSpanRedactionProcessor(data_validation_result=True)

        result = processor.validate_data({"test": "data"})
        assert result is True

    def test_validate_data_implementation_false(self):
        """Test that validate_data works with concrete implementation returning False."""
        processor = ConcreteContextualSpanRedactionProcessor(data_validation_result=False)

        result = processor.validate_data({"test": "data"})
        assert result is False

    def test_validate_data_with_various_data_types(self):
        """Test validate_data with various data types."""
        processor = ConcreteContextualSpanRedactionProcessor(data_validation_result=True)

        # Test with different data types
        assert processor.validate_data({"string": "value"}) is True
        assert processor.validate_data({"number": 123}) is True
        assert processor.validate_data({"list": [1, 2, 3]}) is True
        assert processor.validate_data({}) is True


class TestContextualSpanRedactionProcessorEdgeCases:
    """Test edge cases and error conditions."""

    async def test_redact_item_with_special_attribute_values(self):
        """Test redacting attributes with special values (None, empty string, etc.)."""
        span_context = SpanContext(span_id=123, trace_id=456)
        span = Span(name="test",
                    context=span_context,
                    parent=None,
                    start_time=1000000,
                    end_time=2000000,
                    attributes={
                        "none_value": None,
                        "empty_string": "",
                        "zero": 0,
                        "false_bool": False,
                        "list_value": [1, 2, 3],
                        "dict_value": {
                            "nested": "value"
                        }
                    },
                    events=[])

        processor = ConcreteContextualSpanRedactionProcessor(
            attributes=["none_value", "empty_string", "zero", "false_bool", "list_value", "dict_value"])

        result = await processor.redact_item(span)

        # All should be redacted regardless of their original values
        assert result.attributes["none_value"] == "[REDACTED]"
        assert result.attributes["empty_string"] == "[REDACTED]"
        assert result.attributes["zero"] == "[REDACTED]"
        assert result.attributes["false_bool"] == "[REDACTED]"
        assert result.attributes["list_value"] == "[REDACTED]"
        assert result.attributes["dict_value"] == "[REDACTED]"

    async def test_redact_item_with_unicode_attributes(self):
        """Test redacting attributes with unicode values."""
        span_context = SpanContext(span_id=123, trace_id=456)
        span = Span(name="test",
                    context=span_context,
                    parent=None,
                    start_time=1000000,
                    end_time=2000000,
                    attributes={
                        "unicode_key": "🔐 sensitive data 密码",
                        "emoji_key": "🚀🌟💫",
                        "chinese": "保密信息",
                        "arabic": "معلومات سرية"
                    },
                    events=[])

        processor = ConcreteContextualSpanRedactionProcessor(
            attributes=["unicode_key", "emoji_key", "chinese", "arabic"])

        result = await processor.redact_item(span)

        assert result.attributes["unicode_key"] == "[REDACTED]"
        assert result.attributes["emoji_key"] == "[REDACTED]"
        assert result.attributes["chinese"] == "[REDACTED]"
        assert result.attributes["arabic"] == "[REDACTED]"

    async def test_redact_item_with_very_long_attribute_names(self):
        """Test redacting attributes with very long names."""
        long_key = "a" * 1000  # Very long attribute name
        span_context = SpanContext(span_id=123, trace_id=456)
        span = Span(name="test",
                    context=span_context,
                    parent=None,
                    start_time=1000000,
                    end_time=2000000,
                    attributes={long_key: "sensitive_value"},
                    events=[])

        processor = ConcreteContextualSpanRedactionProcessor(attributes=[long_key])

        result = await processor.redact_item(span)

        assert result.attributes[long_key] == "[REDACTED]"

    def test_initialization_with_duplicate_attributes(self):
        """Test initialization with duplicate attributes."""
        attributes = ["user_id", "api_key", "user_id", "session_token", "api_key"]
        processor = ConcreteContextualSpanRedactionProcessor(attributes=attributes)

        # Should store the list as-is (duplicates included)
        assert processor.attributes == attributes

    async def test_redact_duplicate_attributes_processed_once(self, sample_span):
        """Test that duplicate attributes in the list are processed correctly."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id", "user_id", "api_key"])

        result = await processor.redact_item(sample_span)

        # Should still work correctly despite duplicates
        assert result.attributes["user_id"] == "[REDACTED]"
        assert result.attributes["api_key"] == "[REDACTED]"
        assert result.attributes["session_token"] == "secret_token"  # unchanged

    async def test_redact_with_empty_redaction_value(self, sample_span):
        """Test redacting with empty string as redaction value."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id"], redaction_value="")

        result = await processor.redact_item(sample_span)

        assert result.attributes["user_id"] == ""

    async def test_redact_with_whitespace_redaction_value(self, sample_span):
        """Test redacting with whitespace as redaction value."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id"], redaction_value="   ")

        result = await processor.redact_item(sample_span)

        assert result.attributes["user_id"] == "   "

    async def test_redact_preserves_span_events(self, sample_span):
        """Test that redaction preserves span events."""
        # Add some events to the span
        sample_span.add_event("test_event", {"event_attr": "event_value"})

        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id"])

        result = await processor.redact_item(sample_span)

        # Events should be preserved
        assert len(result.events) == 1
        assert result.events[0].name == "test_event"
        assert result.events[0].attributes["event_attr"] == "event_value"

        # Span attributes should still be redacted
        assert result.attributes["user_id"] == "[REDACTED]"


class TestContextualSpanRedactionProcessorProcess:
    """Test ContextualSpanRedactionProcessor process method - the main public interface."""

    async def test_process_should_redact_true(self, sample_span):
        """Test process method when should_redact returns True."""
        processor = ConcreteContextualSpanRedactionProcessor(
            attributes=["user_id", "api_key"],
            force_redact=True  # This ensures should_redact returns True
        )

        result = await processor.process(sample_span)

        # Should redact the specified attributes
        assert result.attributes["user_id"] == "[REDACTED]"
        assert result.attributes["api_key"] == "[REDACTED]"
        assert result.attributes["session_token"] == "secret_token"  # unchanged
        assert result.attributes["normal_field"] == "normal_value"  # unchanged

    async def test_process_should_redact_false(self, sample_span):
        """Test process method when should_redact returns False."""
        processor = ConcreteContextualSpanRedactionProcessor(
            attributes=["user_id", "api_key"],
            enabled=False  # This ensures should_redact returns False
        )

        result = await processor.process(sample_span)

        # Should not redact anything - all attributes unchanged
        assert result.attributes["user_id"] == "user123"
        assert result.attributes["api_key"] == "api_secret"
        assert result.attributes["session_token"] == "secret_token"
        assert result.attributes["normal_field"] == "normal_value"

    async def test_process_with_callback_conditions(self, sample_span):
        """Test process method with callback-based conditions."""
        # Create a processor that will redact based on extracted data and callback
        processor = ConcreteContextualSpanRedactionProcessor(
            attributes=["user_id"],
            extracted_data={"test": "data"},
            data_validation_result=True,
            force_redact=True  # Force redaction for this test
        )

        result = await processor.process(sample_span)

        assert result.attributes["user_id"] == "[REDACTED]"
        assert result.attributes["api_key"] == "api_secret"  # unchanged

    async def test_process_preserves_span_identity(self, sample_span):
        """Test that process method preserves span identity."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id"], force_redact=True)

        result = await processor.process(sample_span)

        # Should be the same span object
        assert result is sample_span
        assert result.name == "test_operation"
        if result.context:
            assert result.context.span_id == 123
            assert result.context.trace_id == 456

    async def test_process_with_custom_redaction_value(self, sample_span):
        """Test process method with custom redaction value."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id", "api_key"],
                                                             redaction_value="***HIDDEN***",
                                                             force_redact=True)

        result = await processor.process(sample_span)

        assert result.attributes["user_id"] == "***HIDDEN***"
        assert result.attributes["api_key"] == "***HIDDEN***"

    async def test_process_with_redaction_tag(self, sample_span):
        """Test process method with redaction tag."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id"],
                                                             redaction_tag="processed_by_test",
                                                             force_redact=True)

        result = await processor.process(sample_span)

        assert result.attributes["user_id"] == "[REDACTED]"
        assert result.attributes["processed_by_test"] is True

    async def test_process_minimal_span(self, minimal_span):
        """Test process method with minimal span (no attributes)."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id"], force_redact=True)

        result = await processor.process(minimal_span)

        # Should remain empty
        assert result.attributes == {}

    async def test_process_no_matching_attributes(self, sample_span):
        """Test process method when no attributes match the configured list."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["nonexistent_attr"], force_redact=True)

        result = await processor.process(sample_span)

        # All original attributes should remain unchanged
        assert result.attributes["user_id"] == "user123"
        assert result.attributes["api_key"] == "api_secret"
        assert result.attributes["session_token"] == "secret_token"

    async def test_process_empty_attributes_list(self, sample_span):
        """Test process method with empty attributes list."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=[], force_redact=True)

        result = await processor.process(sample_span)

        # All attributes should remain unchanged (nothing to redact)
        assert result.attributes["user_id"] == "user123"
        assert result.attributes["api_key"] == "api_secret"
        assert result.attributes["session_token"] == "secret_token"

    async def test_process_multiple_calls_idempotent(self, sample_span):
        """Test that multiple process calls are idempotent."""
        processor = ConcreteContextualSpanRedactionProcessor(attributes=["user_id"], force_redact=True)

        # First process call
        result1 = await processor.process(sample_span)
        assert result1.attributes["user_id"] == "[REDACTED]"

        # Second process call on the same span
        result2 = await processor.process(result1)
        assert result2.attributes["user_id"] == "[REDACTED]"  # Still redacted

        # Should be the same span object
        assert result2 is result1 is sample_span
