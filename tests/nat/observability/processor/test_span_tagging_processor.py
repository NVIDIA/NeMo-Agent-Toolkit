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

import logging
import os
from unittest.mock import patch

import pytest

from nat.data_models.span import Span
from nat.data_models.span import SpanContext
from nat.observability.processor.span_tagging_processor import SpanTaggingProcessor

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_span():
    """Create a sample span for testing."""
    span_context = SpanContext(span_id=123, trace_id=456)
    return Span(name="test_operation",
                context=span_context,
                parent=None,
                start_time=1000000,
                end_time=2000000,
                attributes={"existing_key": "existing_value"},
                events=[])


class TestSpanTaggingProcessorInitialization:
    """Test SpanTaggingProcessor initialization and configuration."""

    def test_default_initialization(self):
        """Test processor with default parameters."""
        processor = SpanTaggingProcessor()

        assert processor.tag_key is None
        assert processor.tag_value is None
        assert processor._span_prefix == "nat"  # Default value

    def test_custom_initialization_all_parameters(self):
        """Test processor with all custom parameters."""
        processor = SpanTaggingProcessor(tag_key="environment", tag_value="production", span_prefix="custom")

        assert processor.tag_key == "environment"
        assert processor.tag_value == "production"
        assert processor._span_prefix == "custom"

    def test_partial_initialization_tag_key_only(self):
        """Test processor with only tag_key provided."""
        processor = SpanTaggingProcessor(tag_key="service")

        assert processor.tag_key == "service"
        assert processor.tag_value is None
        assert processor._span_prefix == "nat"

    def test_partial_initialization_tag_value_only(self):
        """Test processor with only tag_value provided."""
        processor = SpanTaggingProcessor(tag_value="backend")

        assert processor.tag_key is None
        assert processor.tag_value == "backend"
        assert processor._span_prefix == "nat"

    def test_custom_span_prefix_only(self):
        """Test processor with only custom span_prefix."""
        processor = SpanTaggingProcessor(span_prefix="myapp")

        assert processor.tag_key is None
        assert processor.tag_value is None
        assert processor._span_prefix == "myapp"

    @patch.dict(os.environ, {"NAT_SPAN_PREFIX": "env_prefix"})
    def test_span_prefix_from_environment_variable(self):
        """Test that span_prefix uses NAT_SPAN_PREFIX environment variable."""
        processor = SpanTaggingProcessor()

        assert processor._span_prefix == "env_prefix"

    @patch.dict(os.environ, {"NAT_SPAN_PREFIX": "env_prefix"})
    def test_explicit_span_prefix_overrides_environment(self):
        """Test that explicit span_prefix overrides environment variable."""
        processor = SpanTaggingProcessor(span_prefix="explicit")

        assert processor._span_prefix == "explicit"

    @patch.dict(os.environ, {"NAT_SPAN_PREFIX": ""})
    def test_empty_environment_variable_fallback(self):
        """Test that empty NAT_SPAN_PREFIX falls back to 'nat'."""
        processor = SpanTaggingProcessor()

        assert processor._span_prefix == "nat"

    @patch.dict(os.environ, {"NAT_SPAN_PREFIX": "  whitespace  "})
    def test_environment_variable_whitespace_trimming(self):
        """Test that NAT_SPAN_PREFIX whitespace is trimmed."""
        processor = SpanTaggingProcessor()

        assert processor._span_prefix == "whitespace"

    @patch.dict(os.environ, {"NAT_SPAN_PREFIX": "   "})
    def test_whitespace_only_environment_variable_fallback(self):
        """Test that whitespace-only NAT_SPAN_PREFIX falls back to 'nat'."""
        processor = SpanTaggingProcessor()

        assert processor._span_prefix == "nat"


class TestSpanTaggingProcessorProcess:
    """Test the process method of SpanTaggingProcessor."""

    async def test_process_with_both_tag_key_and_value(self, sample_span):
        """Test process method when both tag_key and tag_value are provided."""
        processor = SpanTaggingProcessor(tag_key="environment", tag_value="production", span_prefix="myapp")

        result = await processor.process(sample_span)

        # Should return the same span object (modified in place)
        assert result is sample_span

        # Should have added the new attribute
        assert "myapp.environment" in sample_span.attributes
        assert sample_span.attributes["myapp.environment"] == "production"

        # Should preserve existing attributes
        assert sample_span.attributes["existing_key"] == "existing_value"

    async def test_process_with_default_span_prefix(self, sample_span):
        """Test process method with default span prefix."""
        processor = SpanTaggingProcessor(tag_key="service", tag_value="api")

        result = await processor.process(sample_span)

        assert result is sample_span
        assert "nat.service" in sample_span.attributes
        assert sample_span.attributes["nat.service"] == "api"

    async def test_process_with_missing_tag_key(self, sample_span):
        """Test process method when tag_key is None."""
        processor = SpanTaggingProcessor(tag_key=None, tag_value="production")

        original_attributes = sample_span.attributes.copy()
        result = await processor.process(sample_span)

        # Should return the same span object
        assert result is sample_span

        # Should not modify attributes
        assert sample_span.attributes == original_attributes

    async def test_process_with_missing_tag_value(self, sample_span):
        """Test process method when tag_value is None."""
        processor = SpanTaggingProcessor(tag_key="environment", tag_value=None)

        original_attributes = sample_span.attributes.copy()
        result = await processor.process(sample_span)

        # Should return the same span object
        assert result is sample_span

        # Should not modify attributes
        assert sample_span.attributes == original_attributes

    async def test_process_with_both_missing(self, sample_span):
        """Test process method when both tag_key and tag_value are None."""
        processor = SpanTaggingProcessor()

        original_attributes = sample_span.attributes.copy()
        result = await processor.process(sample_span)

        # Should return the same span object
        assert result is sample_span

        # Should not modify attributes
        assert sample_span.attributes == original_attributes

    async def test_process_with_empty_string_tag_key(self, sample_span):
        """Test process method with empty string tag_key."""
        processor = SpanTaggingProcessor(tag_key="", tag_value="production")

        original_attributes = sample_span.attributes.copy()
        result = await processor.process(sample_span)

        # Should return the same span object
        assert result is sample_span

        # Should not modify attributes (empty string is falsy)
        assert sample_span.attributes == original_attributes

    async def test_process_with_empty_string_tag_value(self, sample_span):
        """Test process method with empty string tag_value."""
        processor = SpanTaggingProcessor(tag_key="environment", tag_value="")

        original_attributes = sample_span.attributes.copy()
        result = await processor.process(sample_span)

        # Should return the same span object
        assert result is sample_span

        # Should not modify attributes (empty string is falsy)
        assert sample_span.attributes == original_attributes

    async def test_process_overwrites_existing_attribute(self, sample_span):
        """Test that process method overwrites existing attributes with same key."""
        # Add an attribute that will be overwritten
        sample_span.set_attribute("nat.environment", "development")

        processor = SpanTaggingProcessor(tag_key="environment", tag_value="production")

        result = await processor.process(sample_span)

        assert result is sample_span
        assert sample_span.attributes["nat.environment"] == "production"
        assert sample_span.attributes["existing_key"] == "existing_value"

    async def test_process_multiple_calls_same_processor(self, sample_span):
        """Test multiple calls to process with the same processor."""
        processor = SpanTaggingProcessor(tag_key="call_count", tag_value="multi")

        # Create additional test spans
        span_context2 = SpanContext(span_id=789, trace_id=101)
        span2 = Span(name="operation2", context=span_context2, attributes={})

        # Process both spans
        result1 = await processor.process(sample_span)
        result2 = await processor.process(span2)

        # Both spans should be tagged
        assert result1 is sample_span
        assert result2 is span2
        assert sample_span.attributes["nat.call_count"] == "multi"
        assert span2.attributes["nat.call_count"] == "multi"


class TestSpanTaggingProcessorEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_process_with_special_characters_in_values(self, sample_span):
        """Test process method with special characters in tag values."""
        processor = SpanTaggingProcessor(tag_key="special", tag_value="value with spaces & symbols!@#")

        result = await processor.process(sample_span)

        assert result is sample_span
        assert sample_span.attributes["nat.special"] == "value with spaces & symbols!@#"

    async def test_process_with_unicode_characters(self, sample_span):
        """Test process method with unicode characters."""
        processor = SpanTaggingProcessor(tag_key="unicode", tag_value="héllo wörld 🌍")

        result = await processor.process(sample_span)

        assert result is sample_span
        assert sample_span.attributes["nat.unicode"] == "héllo wörld 🌍"

    async def test_process_with_string_values(self, sample_span):
        """Test process method with string representations of different value types."""
        # Test with numeric value as string
        processor_num = SpanTaggingProcessor(tag_key="count", tag_value="42")

        await processor_num.process(sample_span)
        assert sample_span.attributes["nat.count"] == "42"

        # Test with boolean value as string
        processor_bool = SpanTaggingProcessor(tag_key="enabled", tag_value="true")

        await processor_bool.process(sample_span)
        assert sample_span.attributes["nat.enabled"] == "true"

    async def test_process_with_complex_span_prefix(self, sample_span):
        """Test process method with complex span prefix containing dots."""
        processor = SpanTaggingProcessor(tag_key="service", tag_value="api", span_prefix="my.app.namespace")

        result = await processor.process(sample_span)

        assert result is sample_span
        assert sample_span.attributes["my.app.namespace.service"] == "api"

    async def test_process_preserves_span_properties(self, sample_span):
        """Test that process method preserves all other span properties."""
        processor = SpanTaggingProcessor(tag_key="test", tag_value="value")

        original_name = sample_span.name
        original_context = sample_span.context
        original_parent = sample_span.parent
        original_start_time = sample_span.start_time
        original_end_time = sample_span.end_time
        original_events = sample_span.events
        original_status = sample_span.status

        result = await processor.process(sample_span)

        # All properties should remain unchanged except attributes
        assert result.name == original_name
        assert result.context == original_context
        assert result.parent == original_parent
        assert result.start_time == original_start_time
        assert result.end_time == original_end_time
        assert result.events == original_events
        assert result.status == original_status


class TestSpanTaggingProcessorEnvironmentVariables:
    """Test environment variable handling in SpanTaggingProcessor."""

    @patch.dict(os.environ, {"NAT_SPAN_PREFIX": "test_env"})
    async def test_environment_variable_usage(self, sample_span):
        """Test that NAT_SPAN_PREFIX environment variable is used."""
        processor = SpanTaggingProcessor(tag_key="env_test", tag_value="value")

        await processor.process(sample_span)

        assert "test_env.env_test" in sample_span.attributes
        assert sample_span.attributes["test_env.env_test"] == "value"

    @patch.dict(os.environ, {}, clear=True)
    async def test_missing_environment_variable_fallback(self, sample_span):
        """Test fallback when NAT_SPAN_PREFIX is not set."""
        # Remove NAT_SPAN_PREFIX from environment
        os.environ.pop("NAT_SPAN_PREFIX", None)

        processor = SpanTaggingProcessor(tag_key="fallback_test", tag_value="value")

        await processor.process(sample_span)

        assert "nat.fallback_test" in sample_span.attributes
        assert sample_span.attributes["nat.fallback_test"] == "value"

    @patch.dict(os.environ, {"NAT_SPAN_PREFIX": "  env_with_spaces  "})
    async def test_environment_variable_whitespace_trimming(self, sample_span):
        """Test that environment variable whitespace is properly trimmed."""
        processor = SpanTaggingProcessor(tag_key="trim_test", tag_value="value")

        await processor.process(sample_span)

        assert "env_with_spaces.trim_test" in sample_span.attributes
        assert sample_span.attributes["env_with_spaces.trim_test"] == "value"

    @patch.dict(os.environ, {"NAT_SPAN_PREFIX": "   "})
    async def test_whitespace_only_environment_variable(self, sample_span):
        """Test behavior when NAT_SPAN_PREFIX is only whitespace."""
        processor = SpanTaggingProcessor(tag_key="whitespace_test", tag_value="value")

        await processor.process(sample_span)

        # Should fall back to "nat" when env var is empty after trimming
        assert "nat.whitespace_test" in sample_span.attributes
        assert sample_span.attributes["nat.whitespace_test"] == "value"


class TestSpanTaggingProcessorBehavior:
    """Test behavior and edge cases of SpanTaggingProcessor."""

    async def test_multiple_processors_different_tags(self, sample_span):
        """Test using multiple processors with different tags on the same span."""
        processor1 = SpanTaggingProcessor(tag_key="environment", tag_value="production")
        processor2 = SpanTaggingProcessor(tag_key="service", tag_value="api")

        await processor1.process(sample_span)
        await processor2.process(sample_span)

        assert sample_span.attributes["nat.environment"] == "production"
        assert sample_span.attributes["nat.service"] == "api"
        assert sample_span.attributes["existing_key"] == "existing_value"

    async def test_same_tag_key_different_processors(self, sample_span):
        """Test that same tag_key from different processors overwrites."""
        processor1 = SpanTaggingProcessor(tag_key="stage", tag_value="dev")
        processor2 = SpanTaggingProcessor(tag_key="stage", tag_value="prod")

        await processor1.process(sample_span)
        assert sample_span.attributes["nat.stage"] == "dev"

        await processor2.process(sample_span)
        assert sample_span.attributes["nat.stage"] == "prod"  # Overwritten

    async def test_process_empty_span_attributes(self):
        """Test processing a span with no existing attributes."""
        span = Span(name="test", attributes={})

        processor = SpanTaggingProcessor(tag_key="new", tag_value="tag")

        result = await processor.process(span)

        assert result is span
        assert span.attributes == {"nat.new": "tag"}

    async def test_process_span_without_context(self):
        """Test processing a span without context."""
        span = Span(name="test", context=None, attributes={})

        processor = SpanTaggingProcessor(tag_key="test", tag_value="value")

        result = await processor.process(span)

        assert result is span
        assert span.attributes["nat.test"] == "value"

    async def test_conditional_tagging_logic(self):
        """Test the conditional logic for when tags are applied."""
        test_cases = [
            # (tag_key, tag_value, should_tag, description)
            ("key", "value", True, "both provided"),
            (None, "value", False, "key is None"),
            ("key", None, False, "value is None"),
            (None, None, False, "both are None"),
            ("", "value", False, "key is empty string"),
            ("key", "", False, "value is empty string"),
            ("", "", False, "both are empty strings"),
            ("key", "0", True, "value is string zero"),
            ("key", "false", True, "value is string false"),
        ]

        for tag_key, tag_value, should_tag, description in test_cases:
            # Create a fresh span for each test case
            test_span = Span(name="test", attributes={"original": "data"})

            processor = SpanTaggingProcessor(tag_key=tag_key, tag_value=tag_value)

            result = await processor.process(test_span)

            assert result is test_span

            if should_tag:
                expected_key = f"nat.{tag_key}"
                assert expected_key in test_span.attributes, f"Failed for case: {description}"
                assert test_span.attributes[expected_key] == tag_value, f"Failed for case: {description}"
            else:
                # No new attributes should be added beyond the original
                assert len(test_span.attributes) == 1, f"Failed for case: {description}"
                assert test_span.attributes == {"original": "data"}, f"Failed for case: {description}"


class TestSpanTaggingProcessorTypeIntrospection:
    """Test type introspection capabilities."""

    def test_processor_types(self):
        """Test type introspection for SpanTaggingProcessor."""
        processor = SpanTaggingProcessor()

        # Both input and output should be Span
        assert processor.input_type is Span
        assert processor.output_type is Span
        assert processor.input_class is Span
        assert processor.output_class is Span


class TestSpanTaggingProcessorIntegration:
    """Test integration scenarios with SpanTaggingProcessor."""

    async def test_realistic_usage_scenario(self):
        """Test a realistic usage scenario with multiple spans and processors."""
        # Simulate a realistic scenario with multiple spans
        spans = [
            Span(name="auth_check", attributes={"user_id": "123"}),
            Span(name="database_query", attributes={"table": "users"}),
            Span(name="api_response", attributes={"status_code": 200})
        ]

        # Create processors for different tagging scenarios
        env_processor = SpanTaggingProcessor(tag_key="environment", tag_value="staging")
        service_processor = SpanTaggingProcessor(tag_key="service", tag_value="user-service")
        version_processor = SpanTaggingProcessor(tag_key="version", tag_value="1.2.3")

        # Apply tags to all spans
        for span in spans:
            await env_processor.process(span)
            await service_processor.process(span)
            await version_processor.process(span)

        # Verify all spans have been properly tagged
        for span in spans:
            assert span.attributes["nat.environment"] == "staging"
            assert span.attributes["nat.service"] == "user-service"
            assert span.attributes["nat.version"] == "1.2.3"

            # Original attributes should be preserved
            if span.name == "auth_check":
                assert span.attributes["user_id"] == "123"
            elif span.name == "database_query":
                assert span.attributes["table"] == "users"
            elif span.name == "api_response":
                assert span.attributes["status_code"] == 200

    @patch.dict(os.environ, {"NAT_SPAN_PREFIX": "prod.service"})
    async def test_complex_span_prefix_with_environment(self, sample_span):
        """Test complex span prefix from environment variable."""
        processor = SpanTaggingProcessor(tag_key="region", tag_value="us-east-1")

        result = await processor.process(sample_span)

        assert result is sample_span
        assert "prod.service.region" in sample_span.attributes
        assert sample_span.attributes["prod.service.region"] == "us-east-1"

    async def test_processor_state_isolation(self):
        """Test that different processor instances maintain isolated state."""
        processor1 = SpanTaggingProcessor(tag_key="env", tag_value="dev")
        processor2 = SpanTaggingProcessor(tag_key="env", tag_value="prod")
        processor3 = SpanTaggingProcessor(tag_key="service", tag_value="api")

        span1 = Span(name="test1", attributes={})
        span2 = Span(name="test2", attributes={})
        span3 = Span(name="test3", attributes={})

        await processor1.process(span1)
        await processor2.process(span2)
        await processor3.process(span3)

        # Each processor should have applied its own tags
        assert span1.attributes["nat.env"] == "dev"
        assert span2.attributes["nat.env"] == "prod"
        assert span3.attributes["nat.service"] == "api"

        # Verify no cross-contamination
        assert "nat.service" not in span1.attributes
        assert "nat.service" not in span2.attributes
        assert "nat.env" not in span3.attributes
