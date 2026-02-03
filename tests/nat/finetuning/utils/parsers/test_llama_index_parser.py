# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import MagicMock

from nat.finetuning.utils.parsers.llama_index_parser import _extract_content
from nat.finetuning.utils.parsers.llama_index_parser import _parse_dict_message


class TestParseDictMessage:
    """Tests for _parse_dict_message function."""

    def test_parse_basic_dict(self):
        """Test parsing basic dictionary message."""
        msg_dict = {"role": "user", "content": "Test message"}
        result = _parse_dict_message(msg_dict)
        assert result["role"] == "user"
        assert result["content"] == "Test message"

    def test_parse_with_text_field(self):
        """Test parsing dict with 'text' instead of 'content'."""
        msg_dict = {"role": "assistant", "text": "Response"}
        result = _parse_dict_message(msg_dict)
        assert result["role"] == "assistant"
        assert result["content"] == "Response"

    def test_parse_with_blocks(self):
        """Test parsing dict with LlamaIndex blocks format."""
        msg_dict = {"role": "assistant", "blocks": [{"text": "First block "}, {"text": "Second block"}]}
        result = _parse_dict_message(msg_dict)
        assert result["role"] == "assistant"
        assert result["content"] == "First block Second block"

    def test_parse_with_blocks_objects(self):
        """Test parsing dict with block objects (not dicts)."""
        block1 = MagicMock()
        block1.text = "Block 1 "
        block2 = MagicMock()
        block2.text = "Block 2"

        msg_dict = {"role": "assistant", "blocks": [block1, block2]}
        result = _parse_dict_message(msg_dict)
        assert result["role"] == "assistant"
        assert result["content"] == "Block 1 Block 2"

    def test_parse_default_role(self):
        """Test that default role is 'user'."""
        msg_dict = {"content": "No role specified"}
        result = _parse_dict_message(msg_dict)
        assert result["role"] == "user"

    def test_parse_with_optional_fields(self):
        """Test parsing with optional fields."""
        msg_dict = {
            "role": "assistant",
            "content": "Test",
            "tool_calls": [{
                "id": "1"
            }],
            "function_call": {
                "name": "test"
            },
            "logprobs": {
                "tokens": []
            }
        }
        result = _parse_dict_message(msg_dict)
        assert result["tool_calls"] == [{"id": "1"}]
        assert "function_call" in result
        assert "logprobs" in result

    def test_parse_empty_content(self):
        """Test parsing dict with no content field."""
        msg_dict = {"role": "user"}
        result = _parse_dict_message(msg_dict)
        assert result["content"] == ""


class TestExtractContent:
    """Tests for _extract_content function."""

    def test_extract_string(self):
        """Test extracting content from string."""
        assert _extract_content("Simple string") == "Simple string"

    def test_extract_from_dict_with_content(self):
        """Test extracting content from dict with 'content' key."""
        data = {"content": "Message content"}
        assert _extract_content(data) == "Message content"

    def test_extract_from_dict_with_text(self):
        """Test extracting content from dict with 'text' key."""
        data = {"text": "Text content"}
        assert _extract_content(data) == "Text content"

    def test_extract_from_dict_with_blocks(self):
        """Test extracting content from dict with blocks."""
        data = {"blocks": [{"text": "First "}, {"text": "Second"}]}
        assert _extract_content(data) == "First Second"

    def test_extract_from_dict_fallback_to_json(self):
        """Test fallback to JSON for dict without known keys."""
        data = {"unknown_key": "value"}
        result = _extract_content(data)
        assert "unknown_key" in result
        assert "value" in result

    def test_extract_from_string_list(self):
        """Test extracting from list of strings."""
        data = ["First line", "Second line", "Third line"]
        result = _extract_content(data)
        assert result == "First line\nSecond line\nThird line"

    def test_extract_from_mixed_list(self):
        """Test extracting from list with non-strings."""
        data = ["String", 123, {"key": "value"}]
        result = _extract_content(data)
        # Should convert to JSON
        assert isinstance(result, str)

    def test_extract_from_object_with_content_attr(self):
        """Test extracting from object with content attribute."""
        mock_obj = MagicMock()
        mock_obj.content = "Object content"
        assert _extract_content(mock_obj) == "Object content"

    def test_extract_from_object_with_text_attr(self):
        """Test extracting from object with text attribute."""
        mock_obj = MagicMock()
        del mock_obj.content  # Remove content attribute
        mock_obj.text = "Object text"
        assert _extract_content(mock_obj) == "Object text"

    def test_extract_fallback_to_str(self):
        """Test fallback to str() for unknown types."""
        data = 12345
        assert _extract_content(data) == "12345"

    def test_extract_empty_string(self):
        """Test extracting empty string."""
        assert _extract_content("") == ""

    def test_extract_none(self):
        """Test extracting None."""
        assert _extract_content(None) == "None"
