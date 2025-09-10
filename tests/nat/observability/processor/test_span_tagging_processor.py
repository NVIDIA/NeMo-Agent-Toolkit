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
from enum import Enum
from typing import cast
from unittest.mock import patch

import pytest

from nat.data_models.span import Span
from nat.data_models.span import SpanContext
from nat.observability.processor.span_tagging_processor import SpanTaggingProcessor


class SampleEnum(str, Enum):
    """Sample enum for testing enum value handling."""
    VALUE1 = "test_value_1"
    VALUE2 = "test_value_2"


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

        assert processor.tags == {}
        assert processor._span_prefix == "nat"  # Default value

    def test_single_tag_initialization(self):
        """Test processor with single tag."""
        processor = SpanTaggingProcessor(tags={"environment": "production"}, span_prefix="custom")

        assert processor.tags == {"environment": "production"}
        assert processor._span_prefix == "custom"

    def test_multiple_tags_initialization(self):
        """Test processor with multiple tags."""
        tags = cast(dict[str, Enum | str], {"environment": "production", "service": "api", "team": "backend"})
        processor = SpanTaggingProcessor(tags=tags)

        assert processor.tags == tags
        assert processor._span_prefix == "nat"

    def test_enum_tag_initialization(self):
        """Test processor with enum tag values."""
        tags = {"status": SampleEnum.VALUE1, "type": "string_value"}
        processor = SpanTaggingProcessor(tags=tags)

        assert processor.tags == tags
        assert processor._span_prefix == "nat"

    def test_empty_tags_initialization(self):
        """Test processor with empty tags dictionary."""
        processor = SpanTaggingProcessor(tags={})

        assert processor.tags == {}
        assert processor._span_prefix == "nat"

    def test_custom_span_prefix_only(self):
        """Test processor with only custom span_prefix."""
        processor = SpanTaggingProcessor(span_prefix="myapp")

        assert processor.tags == {}
        assert processor._span_prefix == "myapp"

    @patch.dict(os.environ, {"NAT_SPAN_PREFIX": "env_prefix"})
    def test_span_prefix_from_environment_variable(self):
        """Test that span_prefix uses NAT_SPAN_PREFIX environment variable."""
        processor = SpanTaggingProcessor(tags={"test": "value"})

        assert processor._span_prefix == "env_prefix"

    @patch.dict(os.environ, {"NAT_SPAN_PREFIX": "env_prefix"})
    def test_explicit_span_prefix_overrides_environment(self):
        """Test that explicit span_prefix overrides environment variable."""
        processor = SpanTaggingProcessor(tags={"test": "value"}, span_prefix="explicit")

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

    async def test_process_with_single_tag(self, sample_span):
        """Test process method with single tag."""
        processor = SpanTaggingProcessor(tags={"environment": "production"}, span_prefix="myapp")

        result = await processor.process(sample_span)

        # Should return the same span object (modified in place)
        assert result is sample_span

        # Should have added the new attribute
        assert "myapp.environment" in sample_span.attributes
        assert sample_span.attributes["myapp.environment"] == "production"

        # Should preserve existing attributes
        assert sample_span.attributes["existing_key"] == "existing_value"

    async def test_process_with_multiple_tags(self, sample_span):
        """Test process method with multiple tags."""
        tags = cast(dict[str, Enum | str], {"environment": "production", "service": "api", "team": "backend"})
        processor = SpanTaggingProcessor(tags=tags, span_prefix="myapp")

        result = await processor.process(sample_span)

        assert result is sample_span
        assert sample_span.attributes["myapp.environment"] == "production"
        assert sample_span.attributes["myapp.service"] == "api"
        assert sample_span.attributes["myapp.team"] == "backend"
        assert sample_span.attributes["existing_key"] == "existing_value"

    async def test_process_with_enum_values(self, sample_span):
        """Test process method with enum tag values."""
        tags = {"status": SampleEnum.VALUE1, "type": "string_value"}
        processor = SpanTaggingProcessor(tags=tags)

        result = await processor.process(sample_span)

        assert result is sample_span
        assert sample_span.attributes["nat.status"] == "test_value_1"  # Enum converted to string
        assert sample_span.attributes["nat.type"] == "string_value"

    async def test_process_with_default_span_prefix(self, sample_span):
        """Test process method with default span prefix."""
        processor = SpanTaggingProcessor(tags={"service": "api"})

        result = await processor.process(sample_span)

        assert result is sample_span
        assert "nat.service" in sample_span.attributes
        assert sample_span.attributes["nat.service"] == "api"

    async def test_process_with_empty_tags(self, sample_span):
        """Test process method with empty tags dictionary."""
        processor = SpanTaggingProcessor(tags={})

        original_attributes = sample_span.attributes.copy()
        result = await processor.process(sample_span)

        # Should return the same span object
        assert result is sample_span

        # Should not modify attributes
        assert sample_span.attributes == original_attributes

    async def test_process_with_no_tags(self, sample_span):
        """Test process method when tags is None."""
        processor = SpanTaggingProcessor()

        original_attributes = sample_span.attributes.copy()
        result = await processor.process(sample_span)

        # Should return the same span object
        assert result is sample_span

        # Should not modify attributes
        assert sample_span.attributes == original_attributes

    async def test_process_with_empty_string_tag_key(self, sample_span):
        """Test process method with empty string tag key."""
        processor = SpanTaggingProcessor(tags={"": "production"})

        original_attributes = sample_span.attributes.copy()
        result = await processor.process(sample_span)

        # Should return the same span object
        assert result is sample_span

        # Should not modify attributes (empty string key is falsy)
        assert sample_span.attributes == original_attributes

    async def test_process_overwrites_existing_attribute(self, sample_span):
        """Test that process method overwrites existing attributes with same key."""
        # Add an attribute that will be overwritten
        sample_span.set_attribute("nat.environment", "development")

        processor = SpanTaggingProcessor(tags={"environment": "production"})

        result = await processor.process(sample_span)

        assert result is sample_span
        assert sample_span.attributes["nat.environment"] == "production"
        assert sample_span.attributes["existing_key"] == "existing_value"

    async def test_process_multiple_calls_same_processor(self, sample_span):
        """Test multiple calls to process with the same processor."""
        processor = SpanTaggingProcessor(tags={"call_count": "multi", "service": "shared"})

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
        assert sample_span.attributes["nat.service"] == "shared"
        assert span2.attributes["nat.call_count"] == "multi"
        assert span2.attributes["nat.service"] == "shared"


class TestSpanTaggingProcessorEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_process_with_special_characters_in_values(self, sample_span):
        """Test process method with special characters in tag values."""
        processor = SpanTaggingProcessor(tags={"special": "value with spaces & symbols!@#"})

        result = await processor.process(sample_span)

        assert result is sample_span
        assert sample_span.attributes["nat.special"] == "value with spaces & symbols!@#"

    async def test_process_with_unicode_characters(self, sample_span):
        """Test process method with unicode characters."""
        processor = SpanTaggingProcessor(tags={"unicode": "héllo wörld 🌍"})

        result = await processor.process(sample_span)

        assert result is sample_span
        assert sample_span.attributes["nat.unicode"] == "héllo wörld 🌍"

    async def test_process_with_string_values(self, sample_span):
        """Test process method with string representations of different value types."""
        # Test with multiple different string value types
        tags = cast(dict[str, Enum | str], {"count": "42", "enabled": "true", "price": "19.99", "empty": ""})
        processor = SpanTaggingProcessor(tags=tags)

        await processor.process(sample_span)

        assert sample_span.attributes["nat.count"] == "42"
        assert sample_span.attributes["nat.enabled"] == "true"
        assert sample_span.attributes["nat.price"] == "19.99"
        # Empty string values are skipped
        assert "nat.empty" not in sample_span.attributes

    async def test_process_with_complex_span_prefix(self, sample_span):
        """Test process method with complex span prefix containing dots."""
        processor = SpanTaggingProcessor(tags={"service": "api", "version": "1.2.3"}, span_prefix="my.app.namespace")

        result = await processor.process(sample_span)

        assert result is sample_span
        assert sample_span.attributes["my.app.namespace.service"] == "api"
        assert sample_span.attributes["my.app.namespace.version"] == "1.2.3"

    async def test_process_preserves_span_properties(self, sample_span):
        """Test that process method preserves all other span properties."""
        processor = SpanTaggingProcessor(tags={"test": "value", "environment": "production"})

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

        # But attributes should be modified
        assert sample_span.attributes["nat.test"] == "value"
        assert sample_span.attributes["nat.environment"] == "production"


class TestSpanTaggingProcessorEnvironmentVariables:
    """Test environment variable handling in SpanTaggingProcessor."""

    @patch.dict(os.environ, {"NAT_SPAN_PREFIX": "test_env"})
    async def test_environment_variable_usage(self, sample_span):
        """Test that NAT_SPAN_PREFIX environment variable is used."""
        processor = SpanTaggingProcessor(tags={"env_test": "value", "service": "api"})

        await processor.process(sample_span)

        assert "test_env.env_test" in sample_span.attributes
        assert sample_span.attributes["test_env.env_test"] == "value"
        assert "test_env.service" in sample_span.attributes
        assert sample_span.attributes["test_env.service"] == "api"

    @patch.dict(os.environ, {}, clear=True)
    async def test_missing_environment_variable_fallback(self, sample_span):
        """Test fallback when NAT_SPAN_PREFIX is not set."""
        # Remove NAT_SPAN_PREFIX from environment
        os.environ.pop("NAT_SPAN_PREFIX", None)

        processor = SpanTaggingProcessor(tags={"fallback_test": "value"})

        await processor.process(sample_span)

        assert "nat.fallback_test" in sample_span.attributes
        assert sample_span.attributes["nat.fallback_test"] == "value"

    @patch.dict(os.environ, {"NAT_SPAN_PREFIX": "  env_with_spaces  "})
    async def test_environment_variable_whitespace_trimming(self, sample_span):
        """Test that environment variable whitespace is properly trimmed."""
        processor = SpanTaggingProcessor(tags={"trim_test": "value"})

        await processor.process(sample_span)

        assert "env_with_spaces.trim_test" in sample_span.attributes
        assert sample_span.attributes["env_with_spaces.trim_test"] == "value"

    @patch.dict(os.environ, {"NAT_SPAN_PREFIX": "   "})
    async def test_whitespace_only_environment_variable(self, sample_span):
        """Test behavior when NAT_SPAN_PREFIX is only whitespace."""
        processor = SpanTaggingProcessor(tags={"whitespace_test": "value"})

        await processor.process(sample_span)

        # Should fall back to "nat" when env var is empty after trimming
        assert "nat.whitespace_test" in sample_span.attributes
        assert sample_span.attributes["nat.whitespace_test"] == "value"


class TestSpanTaggingProcessorBehavior:
    """Test behavior and edge cases of SpanTaggingProcessor."""

    async def test_multiple_processors_different_tags(self, sample_span):
        """Test using multiple processors with different tags on the same span."""
        processor1 = SpanTaggingProcessor(tags={"environment": "production"})
        processor2 = SpanTaggingProcessor(tags={"service": "api", "team": "backend"})

        await processor1.process(sample_span)
        await processor2.process(sample_span)

        assert sample_span.attributes["nat.environment"] == "production"
        assert sample_span.attributes["nat.service"] == "api"
        assert sample_span.attributes["nat.team"] == "backend"
        assert sample_span.attributes["existing_key"] == "existing_value"

    async def test_same_tag_key_different_processors(self, sample_span):
        """Test that same tag key from different processors overwrites."""
        processor1 = SpanTaggingProcessor(tags={"stage": "dev", "version": "1.0"})
        processor2 = SpanTaggingProcessor(tags={"stage": "prod"})  # Same key, different value

        await processor1.process(sample_span)
        assert sample_span.attributes["nat.stage"] == "dev"
        assert sample_span.attributes["nat.version"] == "1.0"

        await processor2.process(sample_span)
        assert sample_span.attributes["nat.stage"] == "prod"  # Overwritten
        assert sample_span.attributes["nat.version"] == "1.0"  # Preserved

    async def test_process_empty_span_attributes(self):
        """Test processing a span with no existing attributes."""
        span = Span(name="test", attributes={})

        processor = SpanTaggingProcessor(tags={"new": "tag", "another": "value"})

        result = await processor.process(span)

        assert result is span
        assert span.attributes == {"nat.new": "tag", "nat.another": "value"}

    async def test_process_span_without_context(self):
        """Test processing a span without context."""
        span = Span(name="test", context=None, attributes={})

        processor = SpanTaggingProcessor(tags={"test": "value", "context_test": "works"})

        result = await processor.process(span)

        assert result is span
        assert span.attributes["nat.test"] == "value"
        assert span.attributes["nat.context_test"] == "works"

    async def test_conditional_tagging_logic(self):
        """Test the conditional logic for when tags are applied."""
        test_cases = [
            # (tags_dict, expected_attributes_count, description)
            ({
                "key": "value"
            }, 2, "normal key-value"),
            ({
                "key": "value", "key2": "value2"
            }, 3, "multiple tags"),
            ({}, 1, "empty tags dict"),
            ({
                "": "value"
            }, 1, "key is empty string"),
            ({
                "key": ""
            }, 1, "value is empty string"),
            ({
                "key": "0"
            }, 2, "value is string zero"),
            ({
                "key": "false"
            }, 2, "value is string false"),
        ]

        for tags_dict, expected_count, description in test_cases:
            # Create a fresh span for each test case
            test_span = Span(name="test", attributes={"original": "data"})

            processor = SpanTaggingProcessor(tags=tags_dict)

            result = await processor.process(test_span)

            assert result is test_span
            assert len(test_span.attributes) == expected_count, f"Failed for case: {description}"

            # Original attribute should always be preserved
            assert test_span.attributes["original"] == "data", f"Failed for case: {description}"


class TestSpanTaggingProcessorTypeIntrospection:
    """Test type introspection capabilities."""

    def test_processor_types(self):
        """Test type introspection for SpanTaggingProcessor."""
        processor = SpanTaggingProcessor()

        # Both input and output should be Span
        assert processor.input_type is Span
        assert processor.output_type is Span

        # Test Pydantic-based validation methods (preferred approach)
        test_span = Span(name="test", span_id="123", trace_id="456")
        assert processor.validate_input_type(test_span)
        assert not processor.validate_input_type("not_a_span")
        assert processor.validate_output_type(test_span)


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

        # Create processors with multiple tags - more realistic
        common_processor = SpanTaggingProcessor(tags={
            "environment": "staging", "service": "user-service", "version": "1.2.3"
        })

        auth_processor = SpanTaggingProcessor(tags={"component": "authentication"})
        db_processor = SpanTaggingProcessor(tags={"component": "database", "db_type": "postgresql"})
        api_processor = SpanTaggingProcessor(tags={"component": "api", "protocol": "http"})

        # Apply common tags to all spans
        for span in spans:
            await common_processor.process(span)

        # Apply specific tags based on span type
        await auth_processor.process(spans[0])  # auth_check
        await db_processor.process(spans[1])  # database_query
        await api_processor.process(spans[2])  # api_response

        # Verify common tags are on all spans
        for span in spans:
            assert span.attributes["nat.environment"] == "staging"
            assert span.attributes["nat.service"] == "user-service"
            assert span.attributes["nat.version"] == "1.2.3"

        # Verify specific component tags
        assert spans[0].attributes["nat.component"] == "authentication"
        assert spans[1].attributes["nat.component"] == "database"
        assert spans[1].attributes["nat.db_type"] == "postgresql"
        assert spans[2].attributes["nat.component"] == "api"
        assert spans[2].attributes["nat.protocol"] == "http"

        # Original attributes should be preserved
        assert spans[0].attributes["user_id"] == "123"
        assert spans[1].attributes["table"] == "users"
        assert spans[2].attributes["status_code"] == 200

    @patch.dict(os.environ, {"NAT_SPAN_PREFIX": "prod.service"})
    async def test_complex_span_prefix_with_environment(self, sample_span):
        """Test complex span prefix from environment variable."""
        processor = SpanTaggingProcessor(tags={"region": "us-east-1", "zone": "1a"})

        result = await processor.process(sample_span)

        assert result is sample_span
        assert "prod.service.region" in sample_span.attributes
        assert sample_span.attributes["prod.service.region"] == "us-east-1"
        assert "prod.service.zone" in sample_span.attributes
        assert sample_span.attributes["prod.service.zone"] == "1a"

    async def test_processor_state_isolation(self):
        """Test that different processor instances maintain isolated state."""
        processor1 = SpanTaggingProcessor(tags={"env": "dev", "team": "alpha"})
        processor2 = SpanTaggingProcessor(tags={"env": "prod", "team": "beta"})
        processor3 = SpanTaggingProcessor(tags={"service": "api", "component": "gateway"})

        span1 = Span(name="test1", attributes={})
        span2 = Span(name="test2", attributes={})
        span3 = Span(name="test3", attributes={})

        await processor1.process(span1)
        await processor2.process(span2)
        await processor3.process(span3)

        # Each processor should have applied its own tags
        assert span1.attributes["nat.env"] == "dev"
        assert span1.attributes["nat.team"] == "alpha"
        assert span2.attributes["nat.env"] == "prod"
        assert span2.attributes["nat.team"] == "beta"
        assert span3.attributes["nat.service"] == "api"
        assert span3.attributes["nat.component"] == "gateway"

        # Verify no cross-contamination
        assert "nat.service" not in span1.attributes
        assert "nat.component" not in span1.attributes
        assert "nat.service" not in span2.attributes
        assert "nat.component" not in span2.attributes
        assert "nat.env" not in span3.attributes
        assert "nat.team" not in span3.attributes

    async def test_enum_integration_with_multiple_tags(self):
        """Test integration scenario with enum values and multiple tags."""
        processor = SpanTaggingProcessor(tags={
            "status": SampleEnum.VALUE1,
            "level": SampleEnum.VALUE2,
            "environment": "production",
            "service_id": "svc-123"
        })

        span = Span(name="complex_operation", attributes={"operation_id": "op-456"})

        result = await processor.process(span)

        assert result is span
        assert span.attributes["nat.status"] == "test_value_1"  # Enum converted to string
        assert span.attributes["nat.level"] == "test_value_2"  # Enum converted to string
        assert span.attributes["nat.environment"] == "production"  # Regular string
        assert span.attributes["nat.service_id"] == "svc-123"  # Regular string
        assert span.attributes["operation_id"] == "op-456"  # Original preserved
