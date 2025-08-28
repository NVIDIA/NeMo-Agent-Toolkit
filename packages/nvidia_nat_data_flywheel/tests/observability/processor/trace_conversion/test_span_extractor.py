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

from unittest.mock import patch

from pydantic import ValidationError

from nat.data_models.intermediate_step import TokenUsageBaseModel
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.span import Span
from nat.data_models.span import SpanContext
from nat.plugins.data_flywheel.observability.processor.trace_conversion.span_extractor import extract_timestamp
from nat.plugins.data_flywheel.observability.processor.trace_conversion.span_extractor import extract_token_usage
from nat.plugins.data_flywheel.observability.processor.trace_conversion.span_extractor import extract_usage_info


class TestExtractTokenUsage:
    """Test suite for extract_token_usage function."""

    def test_extract_token_usage_with_all_attributes(self):
        """Test extracting token usage when all attributes are present."""
        span = Span(name="test_span",
                    context=SpanContext(),
                    attributes={
                        "llm.token_count.prompt": 100, "llm.token_count.completion": 50, "llm.token_count.total": 150
                    })

        result = extract_token_usage(span)

        assert isinstance(result, TokenUsageBaseModel)
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150

    def test_extract_token_usage_with_missing_attributes(self):
        """Test extracting token usage when some attributes are missing."""
        span = Span(
            name="test_span",
            context=SpanContext(),
            attributes={"llm.token_count.prompt": 75
                        # Missing completion and total tokens
                        })

        result = extract_token_usage(span)

        assert isinstance(result, TokenUsageBaseModel)
        assert result.prompt_tokens == 75
        assert result.completion_tokens == 0  # Default value
        assert result.total_tokens == 0  # Default value

    def test_extract_token_usage_with_no_attributes(self):
        """Test extracting token usage when no token attributes are present."""
        span = Span(name="test_span", context=SpanContext(), attributes={})

        result = extract_token_usage(span)

        assert isinstance(result, TokenUsageBaseModel)
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0

    def test_extract_token_usage_with_string_values(self):
        """Test extracting token usage when attributes are string values."""
        span = Span(name="test_span",
                    context=SpanContext(),
                    attributes={
                        "llm.token_count.prompt": "200",
                        "llm.token_count.completion": "100",
                        "llm.token_count.total": "300"
                    })

        result = extract_token_usage(span)

        assert isinstance(result, TokenUsageBaseModel)
        # TokenUsageBaseModel converts string values to integers
        assert result.prompt_tokens == 200
        assert result.completion_tokens == 100
        assert result.total_tokens == 300

    def test_extract_token_usage_with_zero_values(self):
        """Test extracting token usage with explicit zero values."""
        span = Span(name="test_span",
                    context=SpanContext(),
                    attributes={
                        "llm.token_count.prompt": 0, "llm.token_count.completion": 0, "llm.token_count.total": 0
                    })

        result = extract_token_usage(span)

        assert isinstance(result, TokenUsageBaseModel)
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0

    def test_extract_token_usage_with_large_values(self):
        """Test extracting token usage with large token values."""
        span = Span(name="test_span",
                    context=SpanContext(),
                    attributes={
                        "llm.token_count.prompt": 10000,
                        "llm.token_count.completion": 5000,
                        "llm.token_count.total": 15000
                    })

        result = extract_token_usage(span)

        assert isinstance(result, TokenUsageBaseModel)
        assert result.prompt_tokens == 10000
        assert result.completion_tokens == 5000
        assert result.total_tokens == 15000

    def test_extract_token_usage_with_mixed_attributes(self):
        """Test extracting token usage with a mix of present and missing attributes."""
        span = Span(
            name="test_span",
            context=SpanContext(),
            attributes={
                "llm.token_count.prompt": 250,
                "llm.token_count.total": 400,  # Missing completion tokens
                "other.attribute": "ignored"
            })

        result = extract_token_usage(span)

        assert isinstance(result, TokenUsageBaseModel)
        assert result.prompt_tokens == 250
        assert result.completion_tokens == 0  # Default value
        assert result.total_tokens == 400


class TestExtractUsageInfo:
    """Test suite for extract_usage_info function."""

    def test_extract_usage_info_with_all_attributes(self):
        """Test extracting usage info when all attributes are present."""
        span = Span(name="test_span",
                    context=SpanContext(),
                    attributes={
                        "llm.token_count.prompt": 100,
                        "llm.token_count.completion": 50,
                        "llm.token_count.total": 150,
                        "nat.usage.num_llm_calls": 3,
                        "nat.usage.seconds_between_calls": 2
                    })

        result = extract_usage_info(span)

        assert isinstance(result, UsageInfo)
        assert isinstance(result.token_usage, TokenUsageBaseModel)
        assert result.token_usage.prompt_tokens == 100
        assert result.token_usage.completion_tokens == 50
        assert result.token_usage.total_tokens == 150
        assert result.num_llm_calls == 3
        assert result.seconds_between_calls == 2

    def test_extract_usage_info_with_missing_usage_attributes(self):
        """Test extracting usage info when usage-specific attributes are missing."""
        span = Span(
            name="test_span",
            context=SpanContext(),
            attributes={
                "llm.token_count.prompt": 80,
                "llm.token_count.completion": 40,
                "llm.token_count.total": 120
                # Missing nat.usage attributes
            })

        result = extract_usage_info(span)

        assert isinstance(result, UsageInfo)
        assert isinstance(result.token_usage, TokenUsageBaseModel)
        assert result.token_usage.prompt_tokens == 80
        assert result.token_usage.completion_tokens == 40
        assert result.token_usage.total_tokens == 120
        assert result.num_llm_calls == 0  # Default value
        assert result.seconds_between_calls == 0  # Default value

    def test_extract_usage_info_with_no_attributes(self):
        """Test extracting usage info when no relevant attributes are present."""
        span = Span(name="test_span", context=SpanContext(), attributes={})

        result = extract_usage_info(span)

        assert isinstance(result, UsageInfo)
        assert isinstance(result.token_usage, TokenUsageBaseModel)
        assert result.token_usage.prompt_tokens == 0
        assert result.token_usage.completion_tokens == 0
        assert result.token_usage.total_tokens == 0
        assert result.num_llm_calls == 0
        assert result.seconds_between_calls == 0

    def test_extract_usage_info_with_partial_attributes(self):
        """Test extracting usage info with only some usage attributes."""
        span = Span(
            name="test_span",
            context=SpanContext(),
            attributes={
                "llm.token_count.prompt": 60, "nat.usage.num_llm_calls": 2
                # Missing other attributes
            })

        result = extract_usage_info(span)

        assert isinstance(result, UsageInfo)
        assert result.token_usage.prompt_tokens == 60
        assert result.token_usage.completion_tokens == 0
        assert result.token_usage.total_tokens == 0
        assert result.num_llm_calls == 2
        assert result.seconds_between_calls == 0

    @patch('nat.plugins.data_flywheel.observability.processor.trace_conversion.span_extractor.extract_token_usage')
    def test_extract_usage_info_calls_extract_token_usage(self, mock_extract_token_usage):
        """Test that extract_usage_info calls extract_token_usage."""
        mock_token_usage = TokenUsageBaseModel(prompt_tokens=50, completion_tokens=25, total_tokens=75)
        mock_extract_token_usage.return_value = mock_token_usage

        span = Span(name="test_span",
                    context=SpanContext(),
                    attributes={
                        "nat.usage.num_llm_calls": 1, "nat.usage.seconds_between_calls": 1
                    })

        result = extract_usage_info(span)

        mock_extract_token_usage.assert_called_once_with(span)
        assert result.token_usage == mock_token_usage
        assert result.num_llm_calls == 1
        assert result.seconds_between_calls == 1

    def test_extract_usage_info_with_different_data_types(self):
        """Test extracting usage info with different data types for attributes."""
        span = Span(name="test_span",
                    context=SpanContext(),
                    attributes={
                        "llm.token_count.prompt": 90,
                        "llm.token_count.completion": 45,
                        "llm.token_count.total": 135,
                        "nat.usage.num_llm_calls": 5,
                        "nat.usage.seconds_between_calls": 2
                    })

        result = extract_usage_info(span)

        assert isinstance(result, UsageInfo)
        assert result.num_llm_calls == 5
        assert result.seconds_between_calls == 2


class TestExtractTimestamp:
    """Test suite for extract_timestamp function."""

    def test_extract_timestamp_with_valid_integer(self):
        """Test extracting timestamp with valid integer value."""
        span = Span(name="test_span", context=SpanContext(), attributes={"nat.event_timestamp": 1642780800})

        result = extract_timestamp(span)

        assert isinstance(result, int)
        assert result == 1642780800

    def test_extract_timestamp_with_valid_float(self):
        """Test extracting timestamp with valid float value."""
        span = Span(name="test_span", context=SpanContext(), attributes={"nat.event_timestamp": 1642780800.5})

        result = extract_timestamp(span)

        assert isinstance(result, int)
        assert result == 1642780800  # Truncated to int

    def test_extract_timestamp_with_valid_string_number(self):
        """Test extracting timestamp with valid string number."""
        span = Span(name="test_span", context=SpanContext(), attributes={"nat.event_timestamp": "1642780800"})

        result = extract_timestamp(span)

        assert isinstance(result, int)
        assert result == 1642780800

    def test_extract_timestamp_with_valid_string_float(self):
        """Test extracting timestamp with valid string float."""
        span = Span(name="test_span", context=SpanContext(), attributes={"nat.event_timestamp": "1642780800.9"})

        result = extract_timestamp(span)

        assert isinstance(result, int)
        assert result == 1642780800  # Truncated to int

    def test_extract_timestamp_with_missing_attribute(self):
        """Test extracting timestamp when attribute is missing."""
        span = Span(name="test_span", context=SpanContext(), attributes={})

        result = extract_timestamp(span)

        assert isinstance(result, int)
        assert result == 0  # Default value

    def test_extract_timestamp_with_zero_value(self):
        """Test extracting timestamp with explicit zero value."""
        span = Span(name="test_span", context=SpanContext(), attributes={"nat.event_timestamp": 0})

        result = extract_timestamp(span)

        assert isinstance(result, int)
        assert result == 0

    @patch('nat.plugins.data_flywheel.observability.processor.trace_conversion.span_extractor.logger')
    def test_extract_timestamp_with_invalid_string(self, mock_logger):
        """Test extracting timestamp with invalid string value."""
        span = Span(name="test_span", context=SpanContext(), attributes={"nat.event_timestamp": "invalid_timestamp"})

        result = extract_timestamp(span)

        assert isinstance(result, int)
        assert result == 0  # Default value for invalid input
        mock_logger.warning.assert_called_once_with("Invalid timestamp in span '%s', using 0", "test_span")

    @patch('nat.plugins.data_flywheel.observability.processor.trace_conversion.span_extractor.logger')
    def test_extract_timestamp_with_none_value(self, mock_logger):
        """Test extracting timestamp with None value."""
        span = Span(name="test_span", context=SpanContext(), attributes={"nat.event_timestamp": None})

        result = extract_timestamp(span)

        assert isinstance(result, int)
        assert result == 0  # Default value for None
        mock_logger.warning.assert_called_once_with("Invalid timestamp in span '%s', using 0", "test_span")

    @patch('nat.plugins.data_flywheel.observability.processor.trace_conversion.span_extractor.logger')
    def test_extract_timestamp_with_complex_object(self, mock_logger):
        """Test extracting timestamp with complex object that can't be converted."""
        span = Span(name="test_span", context=SpanContext(), attributes={"nat.event_timestamp": {"complex": "object"}})

        result = extract_timestamp(span)

        assert isinstance(result, int)
        assert result == 0  # Default value for invalid input
        mock_logger.warning.assert_called_once_with("Invalid timestamp in span '%s', using 0", "test_span")

    def test_extract_timestamp_with_negative_value(self):
        """Test extracting timestamp with negative value."""
        span = Span(name="test_span", context=SpanContext(), attributes={"nat.event_timestamp": -1642780800})

        result = extract_timestamp(span)

        assert isinstance(result, int)
        assert result == -1642780800  # Negative timestamps are valid

    def test_extract_timestamp_with_large_value(self):
        """Test extracting timestamp with large value."""
        large_timestamp = 9999999999
        span = Span(name="test_span", context=SpanContext(), attributes={"nat.event_timestamp": large_timestamp})

        result = extract_timestamp(span)

        assert isinstance(result, int)
        assert result == large_timestamp

    @patch('nat.plugins.data_flywheel.observability.processor.trace_conversion.span_extractor.logger')
    def test_extract_timestamp_with_empty_string(self, mock_logger):
        """Test extracting timestamp with empty string."""
        span = Span(name="test_span", context=SpanContext(), attributes={"nat.event_timestamp": ""})

        result = extract_timestamp(span)

        assert isinstance(result, int)
        assert result == 0  # Default value for empty string
        mock_logger.warning.assert_called_once_with("Invalid timestamp in span '%s', using 0", "test_span")


class TestIntegrationScenarios:
    """Integration test scenarios combining multiple functions."""

    def test_complete_span_data_extraction(self):
        """Test extracting all data types from a complete span."""
        span = Span(name="complete_test_span",
                    context=SpanContext(),
                    attributes={
                        "llm.token_count.prompt": 150,
                        "llm.token_count.completion": 75,
                        "llm.token_count.total": 225,
                        "nat.usage.num_llm_calls": 2,
                        "nat.usage.seconds_between_calls": 2,
                        "nat.event_timestamp": 1642780800
                    })

        token_usage = extract_token_usage(span)
        usage_info = extract_usage_info(span)
        timestamp = extract_timestamp(span)

        # Verify token usage
        assert token_usage.prompt_tokens == 150
        assert token_usage.completion_tokens == 75
        assert token_usage.total_tokens == 225

        # Verify usage info (includes token usage)
        assert usage_info.token_usage.prompt_tokens == 150
        assert usage_info.token_usage.completion_tokens == 75
        assert usage_info.token_usage.total_tokens == 225
        assert usage_info.num_llm_calls == 2
        assert usage_info.seconds_between_calls == 2

        # Verify timestamp
        assert timestamp == 1642780800

    def test_minimal_span_data_extraction(self):
        """Test extracting data from a minimal span with no attributes."""
        span = Span(name="minimal_test_span", context=SpanContext(), attributes={})

        token_usage = extract_token_usage(span)
        usage_info = extract_usage_info(span)
        timestamp = extract_timestamp(span)

        # All should return default values
        assert token_usage.prompt_tokens == 0
        assert token_usage.completion_tokens == 0
        assert token_usage.total_tokens == 0

        assert usage_info.token_usage.prompt_tokens == 0
        assert usage_info.num_llm_calls == 0
        assert usage_info.seconds_between_calls == 0

        assert timestamp == 0

    def test_partial_span_data_extraction(self):
        """Test extracting data from a span with only some attributes."""
        span = Span(
            name="partial_test_span",
            context=SpanContext(),
            attributes={
                "llm.token_count.prompt": 100,
                "nat.usage.num_llm_calls": 1,
                "nat.event_timestamp": "1642780800"
                # Missing completion tokens, total tokens, and seconds_between_calls
            })

        token_usage = extract_token_usage(span)
        usage_info = extract_usage_info(span)
        timestamp = extract_timestamp(span)

        # Verify mixed results
        assert token_usage.prompt_tokens == 100
        assert token_usage.completion_tokens == 0  # Default
        assert token_usage.total_tokens == 0  # Default

        assert usage_info.token_usage.prompt_tokens == 100
        assert usage_info.num_llm_calls == 1
        assert usage_info.seconds_between_calls == 0  # Default

        assert timestamp == 1642780800


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def test_function_signatures_and_return_types(self):
        """Test that functions have expected signatures and return correct types."""
        span = Span(name="test", context=SpanContext(), attributes={})

        # Test function callability
        assert callable(extract_token_usage)
        assert callable(extract_usage_info)
        assert callable(extract_timestamp)

        # Test return types
        token_result = extract_token_usage(span)
        usage_result = extract_usage_info(span)
        timestamp_result = extract_timestamp(span)

        assert isinstance(token_result, TokenUsageBaseModel)
        assert isinstance(usage_result, UsageInfo)
        assert isinstance(timestamp_result, int)

    def test_functions_with_span_containing_unexpected_attributes(self):
        """Test functions handle spans with unexpected attribute types gracefully."""
        span = Span(
            name="unexpected_test_span",
            context=SpanContext(),
            attributes={
                "llm.token_count.prompt": [1, 2, 3],  # List instead of int
                "nat.usage.num_llm_calls": {
                    "nested": "dict"
                },  # Dict instead of int
                "nat.event_timestamp": True,  # Boolean instead of number
                "unrelated.attribute": "should_be_ignored"
            })

        # Functions should raise ValidationError for invalid types (this is expected behavior)
        try:
            extract_token_usage(span)
            assert False, "Expected ValidationError for invalid token type"
        except ValidationError:
            pass  # Expected behavior

        # Test with valid token data but invalid usage data
        span_with_valid_tokens = Span(
            name="valid_tokens_span",
            context=SpanContext(),
            attributes={
                "llm.token_count.prompt": 100,
                "llm.token_count.completion": 50,
                "llm.token_count.total": 150,
                "nat.usage.num_llm_calls": {
                    "nested": "dict"
                },  # Invalid type
                "nat.usage.seconds_between_calls": 1
            })

        try:
            extract_usage_info(span_with_valid_tokens)
            assert False, "Expected ValidationError for invalid usage type"
        except ValidationError:
            pass  # Expected behavior

        # timestamp function should log warning and return 0
        span_with_invalid_timestamp = Span(
            name="timestamp_span",
            context=SpanContext(),
            attributes={"nat.event_timestamp": True}  # Boolean instead of number
        )

        with patch('nat.plugins.data_flywheel.observability.processor.trace_conversion.span_extractor.logger'
                   ) as mock_logger:
            timestamp = extract_timestamp(span_with_invalid_timestamp)
            assert timestamp == 0
            mock_logger.warning.assert_called_once()
