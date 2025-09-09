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

import pytest

from nat.plugins.data_flywheel.observability.utils.deserialize import deserialize_span_attribute


class TestDeserializeSpanAttribute:
    """Test cases for deserialize_span_attribute function."""

    def test_dict_input_returns_unchanged(self):
        """Test that dict input is returned unchanged."""
        # Test simple dict
        input_dict = {"key1": "value1", "key2": "value2"}
        result = deserialize_span_attribute(input_dict)
        assert result == input_dict
        assert isinstance(result, dict)

        # Test nested dict
        nested_dict = {"outer": {"inner": "value"}, "list": [1, 2, 3]}
        result = deserialize_span_attribute(nested_dict)
        assert result == nested_dict
        assert isinstance(result, dict)

        # Test empty dict
        empty_dict = {}
        result = deserialize_span_attribute(empty_dict)
        assert result == empty_dict
        assert isinstance(result, dict)

    def test_list_input_returns_unchanged(self):
        """Test that list input is returned unchanged."""
        # Test simple list
        input_list = [1, 2, 3]
        result = deserialize_span_attribute(input_list)
        assert result == input_list
        assert isinstance(result, list)

        # Test list with mixed types
        mixed_list = [1, "string", {"key": "value"}, [1, 2]]
        result = deserialize_span_attribute(mixed_list)
        assert result == mixed_list
        assert isinstance(result, list)

        # Test empty list
        empty_list = []
        result = deserialize_span_attribute(empty_list)
        assert result == empty_list
        assert isinstance(result, list)

    def test_valid_json_dict_string(self):
        """Test deserializing valid JSON dict strings."""
        # Test simple JSON dict
        json_str = '{"key": "value", "number": 42}'
        result = deserialize_span_attribute(json_str)
        expected = {"key": "value", "number": 42}
        assert result == expected
        assert isinstance(result, dict)

        # Test nested JSON dict
        nested_json = '{"outer": {"inner": "value"}, "array": [1, 2, 3]}'
        result = deserialize_span_attribute(nested_json)
        expected = {"outer": {"inner": "value"}, "array": [1, 2, 3]}
        assert result == expected
        assert isinstance(result, dict)

        # Test empty JSON object
        empty_json = "{}"
        result = deserialize_span_attribute(empty_json)
        assert result == {}
        assert isinstance(result, dict)

    def test_valid_json_list_string(self):
        """Test deserializing valid JSON list strings."""
        # Test simple JSON array
        json_array = '[1, 2, 3]'
        result = deserialize_span_attribute(json_array)
        expected = [1, 2, 3]
        assert result == expected
        assert isinstance(result, list)

        # Test JSON array with mixed types
        mixed_array = '["string", 42, {"key": "value"}, [1, 2]]'
        result = deserialize_span_attribute(mixed_array)
        expected = ["string", 42, {"key": "value"}, [1, 2]]
        assert result == expected
        assert isinstance(result, list)

        # Test empty JSON array
        empty_array = "[]"
        result = deserialize_span_attribute(empty_array)
        assert result == []
        assert isinstance(result, list)

    def test_valid_json_primitive_values(self):
        """Test deserializing valid JSON primitive values."""
        # Test JSON string
        result = deserialize_span_attribute('"hello"')
        assert result == "hello"
        assert isinstance(result, str)

        # Test JSON number
        result = deserialize_span_attribute('42')
        assert result == 42
        assert isinstance(result, int)

        # Test JSON float
        result = deserialize_span_attribute('3.14')
        assert result == 3.14
        assert isinstance(result, float)

        # Test JSON boolean
        result = deserialize_span_attribute('true')
        assert result is True
        assert isinstance(result, bool)

        result = deserialize_span_attribute('false')
        assert result is False
        assert isinstance(result, bool)

        # Test JSON null
        result = deserialize_span_attribute('null')
        assert result is None

    def test_invalid_json_raises_value_error(self):
        """Test that invalid JSON strings raise ValueError."""
        # Test malformed JSON
        with pytest.raises(ValueError, match="Failed to parse input_value"):
            deserialize_span_attribute('{"key": invalid}')

        # Test incomplete JSON
        with pytest.raises(ValueError, match="Failed to parse input_value"):
            deserialize_span_attribute('{"key":')

        # Test unquoted strings
        with pytest.raises(ValueError, match="Failed to parse input_value"):
            deserialize_span_attribute('hello world')

        # Test single quotes instead of double
        with pytest.raises(ValueError, match="Failed to parse input_value"):
            deserialize_span_attribute("{'key': 'value'}")

        # Test trailing comma
        with pytest.raises(ValueError, match="Failed to parse input_value"):
            deserialize_span_attribute('{"key": "value",}')

    def test_edge_cases_with_type_error(self):
        """Test edge cases that should raise ValueError due to TypeError."""
        # Test None input
        with pytest.raises(ValueError, match="Failed to parse input_value"):
            deserialize_span_attribute(None)  # type: ignore[arg-type]

        # Test numeric input
        with pytest.raises(ValueError, match="Failed to parse input_value"):
            deserialize_span_attribute(42)  # type: ignore[arg-type]

        # Test boolean input
        with pytest.raises(ValueError, match="Failed to parse input_value"):
            deserialize_span_attribute(True)  # type: ignore[arg-type]

    def test_empty_string_raises_value_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Failed to parse input_value"):
            deserialize_span_attribute("")

    def test_whitespace_only_string_raises_value_error(self):
        """Test that whitespace-only strings raise ValueError."""
        with pytest.raises(ValueError, match="Failed to parse input_value"):
            deserialize_span_attribute("   ")

        with pytest.raises(ValueError, match="Failed to parse input_value"):
            deserialize_span_attribute("\t\n")

    def test_error_message_contains_original_value_and_error(self):
        """Test that ValueError contains original value and underlying error."""
        invalid_json = '{"invalid": json}'

        with pytest.raises(ValueError) as exc_info:
            deserialize_span_attribute(invalid_json)

        error_message = str(exc_info.value)
        assert "Failed to parse input_value" in error_message
        assert invalid_json in error_message
        assert "error:" in error_message

    def test_complex_nested_structures(self):
        """Test complex nested JSON structures."""
        complex_json = """
        {
            "metadata": {
                "version": "1.0",
                "timestamp": "2024-01-01T00:00:00Z"
            },
            "data": [
                {
                    "id": 1,
                    "values": [10, 20, 30],
                    "config": {
                        "enabled": true,
                        "threshold": 0.95
                    }
                },
                {
                    "id": 2,
                    "values": [],
                    "config": null
                }
            ]
        }
        """

        result = deserialize_span_attribute(complex_json)

        assert isinstance(result, dict)
        assert result["metadata"]["version"] == "1.0"
        assert len(result["data"]) == 2
        assert result["data"][0]["values"] == [10, 20, 30]
        assert result["data"][1]["config"] is None

    def test_unicode_and_special_characters(self):
        """Test JSON strings with unicode and special characters."""
        # Test unicode characters
        unicode_json = '{"message": "Hello 世界", "emoji": "🚀"}'
        result = deserialize_span_attribute(unicode_json)
        assert isinstance(result, dict)
        assert result["message"] == "Hello 世界"
        assert result["emoji"] == "🚀"

        # Test escaped characters
        escaped_json = '{"path": "/home/user\\nfile.txt", "quote": "He said \\"Hello\\""}'
        result = deserialize_span_attribute(escaped_json)
        assert isinstance(result, dict)
        assert result["path"] == "/home/user\nfile.txt"
        assert result["quote"] == 'He said "Hello"'
