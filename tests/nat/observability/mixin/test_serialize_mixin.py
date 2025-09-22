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

import json
from unittest.mock import patch

from pydantic import BaseModel

from nat.observability.mixin.serialize_mixin import SerializeMixin


class SampleModel(BaseModel):
    """Sample model for testing serialization."""
    name: str
    value: int


class TestSerializeMixin:
    """Test cases for SerializeMixin class."""

    def setup_method(self):
        """Set up test instance."""
        self.mixin = SerializeMixin()

    def test_process_streaming_output_with_basemodel(self):
        """Test _process_streaming_output with BaseModel input."""
        test_model = SampleModel(name="test", value=42)
        result = self.mixin._process_streaming_output(test_model)

        assert isinstance(result, dict)
        assert result == {"name": "test", "value": 42}

    def test_process_streaming_output_with_dict(self):
        """Test _process_streaming_output with dict input."""
        test_dict = {"key": "value", "number": 123}
        result = self.mixin._process_streaming_output(test_dict)

        assert result == test_dict
        assert result is test_dict  # Should return the same object

    def test_process_streaming_output_with_other_types(self):
        """Test _process_streaming_output with various other types."""
        # String
        assert self.mixin._process_streaming_output("test") == "test"

        # Integer
        assert self.mixin._process_streaming_output(42) == 42

        # Float
        assert self.mixin._process_streaming_output(3.14) == 3.14

        # Boolean
        assert self.mixin._process_streaming_output(True) is True

        # None
        assert self.mixin._process_streaming_output(None) is None

        # List
        test_list = [1, 2, 3]
        assert self.mixin._process_streaming_output(test_list) == test_list

    def test_serialize_payload_with_basemodel(self):
        """Test _serialize_payload with BaseModel input."""
        test_model = SampleModel(name="test", value=42)
        result, is_json = self.mixin._serialize_payload(test_model)

        assert is_json is True
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == {"name": "test", "value": 42}

    def test_serialize_payload_with_dict(self):
        """Test _serialize_payload with dict input."""
        test_dict = {"key": "value", "number": 123}
        result, is_json = self.mixin._serialize_payload(test_dict)

        assert is_json is True
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == test_dict

    def test_serialize_payload_with_list_of_basemodels(self):
        """Test _serialize_payload with list containing BaseModels."""
        test_models = [SampleModel(name="first", value=1), SampleModel(name="second", value=2)]
        result, is_json = self.mixin._serialize_payload(test_models)

        # Lists are now properly converted to JSON after processing BaseModels
        assert is_json is True
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == [{"name": "first", "value": 1}, {"name": "second", "value": 2}]

    def test_serialize_payload_with_list_of_dicts(self):
        """Test _serialize_payload with list containing dicts."""
        test_list = [{"name": "first", "value": 1}, {"name": "second", "value": 2}]
        result, is_json = self.mixin._serialize_payload(test_list)

        # Lists are now properly converted to JSON
        assert is_json is True
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == test_list

    def test_serialize_payload_with_mixed_list(self):
        """Test _serialize_payload with list containing mixed types."""
        test_model = SampleModel(name="model", value=1)
        test_dict = {"name": "dict", "value": 2}
        test_list = [test_model, test_dict, "string", 42]

        result, is_json = self.mixin._serialize_payload(test_list)

        # Lists are now properly converted to JSON after processing all items
        assert is_json is True
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == [{"name": "model", "value": 1}, {"name": "dict", "value": 2}, "string", 42]

    def test_serialize_payload_with_nested_list(self):
        """Test _serialize_payload with nested list structure."""
        test_model = SampleModel(name="nested", value=1)
        nested_list = [test_model, {"key": "value"}, [1, 2, 3]]

        result, is_json = self.mixin._serialize_payload(nested_list)

        # Lists are now properly converted to JSON after processing all items
        assert is_json is True
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == [{"name": "nested", "value": 1}, {"key": "value"}, [1, 2, 3]]

    def test_serialize_payload_with_string(self):
        """Test _serialize_payload with string input."""
        result, is_json = self.mixin._serialize_payload("test string")

        assert is_json is False
        assert result == "test string"

    def test_serialize_payload_with_number(self):
        """Test _serialize_payload with numeric input."""
        # Integer
        result, is_json = self.mixin._serialize_payload(42)
        assert is_json is False
        assert result == "42"

        # Float
        result, is_json = self.mixin._serialize_payload(3.14)
        assert is_json is False
        assert result == "3.14"

    def test_serialize_payload_with_boolean(self):
        """Test _serialize_payload with boolean input."""
        result, is_json = self.mixin._serialize_payload(True)
        assert is_json is False
        assert result == "True"

        result, is_json = self.mixin._serialize_payload(False)
        assert is_json is False
        assert result == "False"

    def test_serialize_payload_with_none(self):
        """Test _serialize_payload with None input."""
        result, is_json = self.mixin._serialize_payload(None)

        assert is_json is False
        assert result == "None"

    def test_serialize_payload_exception_handling_basemodel(self):
        """Test _serialize_payload exception handling for BaseModel serialization."""
        test_model = SampleModel(name="test", value=42)

        # Mock TypeAdapter to raise an exception
        with patch('nat.observability.mixin.serialize_mixin.TypeAdapter') as mock_adapter:
            mock_adapter.return_value.dump_json.side_effect = Exception("Serialization error")

            result, is_json = self.mixin._serialize_payload(test_model)

            assert is_json is False
            assert isinstance(result, str)
            # Should fallback to string representation
            assert "name='test'" in result
            assert "value=42" in result

    def test_serialize_payload_exception_handling_dict(self):
        """Test _serialize_payload exception handling for dict serialization."""
        # Create a dict that can't be JSON serialized (contains a set)
        problematic_dict = {"set": {1, 2, 3}}

        with patch('json.dumps', side_effect=TypeError("Object of type set is not JSON serializable")):
            result, is_json = self.mixin._serialize_payload(problematic_dict)

            assert is_json is False
            assert isinstance(result, str)

    def test_serialize_payload_exception_handling_list(self):
        """Test _serialize_payload exception handling for list processing."""
        test_list = [1, 2, 3]

        # Mock _process_streaming_output to raise an exception
        with patch.object(self.mixin, '_process_streaming_output', side_effect=Exception("Processing error")):
            result, is_json = self.mixin._serialize_payload(test_list)

            assert is_json is False
            assert isinstance(result, str)

    def test_serialize_payload_empty_list(self):
        """Test _serialize_payload with empty list."""
        result, is_json = self.mixin._serialize_payload([])

        # Empty lists are now properly converted to JSON
        assert is_json is True
        assert result == "[]"

    def test_serialize_payload_empty_dict(self):
        """Test _serialize_payload with empty dict."""
        result, is_json = self.mixin._serialize_payload({})

        assert is_json is True
        assert result == "{}"

    def test_serialize_payload_complex_nested_structure_with_basemodel(self):
        """Test _serialize_payload with complex nested data structure containing BaseModel."""
        test_model = SampleModel(name="complex", value=100)
        complex_data = {
            "models": [test_model],
            "metadata": {
                "version": "1.0", "items": [{
                    "id": 1, "active": True
                }, {
                    "id": 2, "active": False
                }]
            },
            "simple": "string"
        }

        result, is_json = self.mixin._serialize_payload(complex_data)

        # This fails because BaseModel inside the dict's list can't be JSON serialized
        # (dict serialization doesn't process nested BaseModels)
        assert is_json is False
        assert isinstance(result, str)
        # Should contain the string representation of the dict
        assert "SampleModel(name='complex', value=100)" in result
        assert "'simple': 'string'" in result

    def test_serialize_payload_complex_nested_structure_with_dicts_only(self):
        """Test _serialize_payload with complex nested data structure containing only serializable types."""
        complex_data = {
            "models": [{
                "name": "complex", "value": 100
            }],  # Already dict, not BaseModel
            "metadata": {
                "version": "1.0", "items": [{
                    "id": 1, "active": True
                }, {
                    "id": 2, "active": False
                }]
            },
            "simple": "string"
        }

        result, is_json = self.mixin._serialize_payload(complex_data)

        # This works because all nested objects are JSON serializable
        assert is_json is True
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == complex_data


class TestSerializeMixinIntegration:
    """Integration tests for SerializeMixin."""

    def test_mixin_inheritance(self):
        """Test that SerializeMixin can be properly inherited."""

        class TestClass(SerializeMixin):

            def process_data(self, data):
                return self._serialize_payload(data)

        test_instance = TestClass()
        test_model = SampleModel(name="inheritance", value=999)

        result, is_json = test_instance.process_data(test_model)
        assert is_json is True
        parsed = json.loads(result)
        assert parsed == {"name": "inheritance", "value": 999}
