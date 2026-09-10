# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for FastMCP tool_converter parameter name sanitization."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from pydantic import create_model

from nat.plugins.fastmcp.server.tool_converter import _build_name_mapping
from nat.plugins.fastmcp.server.tool_converter import _sanitize_parameter_name
from nat.plugins.fastmcp.server.tool_converter import create_function_wrapper
from nat.runtime.session import SessionManager


def _mock_session_manager(result_value="result"):
    """Create a mock SessionManager for testing."""
    mock_sm = MagicMock(spec=SessionManager)
    mock_runner = MagicMock()
    mock_runner.__aenter__ = AsyncMock(return_value=mock_runner)
    mock_runner.__aexit__ = AsyncMock(return_value=None)
    mock_runner.result = AsyncMock(return_value=result_value)
    mock_sm.run = MagicMock(return_value=mock_runner)
    return mock_sm


class TestSanitizeParameterName:
    """Test cases for _sanitize_parameter_name utility function."""

    def test_valid_identifier_unchanged(self):
        """Test that a valid Python identifier is returned unchanged."""
        assert _sanitize_parameter_name("query") == "query"

    def test_python_keyword_gets_trailing_underscore(self):
        """Test that Python keywords get a trailing underscore appended."""
        assert _sanitize_parameter_name("from") == "from_"
        assert _sanitize_parameter_name("class") == "class_"
        assert _sanitize_parameter_name("import") == "import_"
        assert _sanitize_parameter_name("return") == "return_"

    def test_hyphenated_name_replaced(self):
        """Test that hyphens are replaced with underscores."""
        assert _sanitize_parameter_name("cik-A") == "cik_A"
        assert _sanitize_parameter_name("start-date") == "start_date"

    def test_digit_prefix_gets_underscore(self):
        """Test that names starting with a digit get a leading underscore."""
        assert _sanitize_parameter_name("1field") == "_1field"

    def test_special_characters_replaced(self):
        """Test that special characters are replaced with underscores."""
        assert _sanitize_parameter_name("field.name") == "field_name"
        assert _sanitize_parameter_name("field name") == "field_name"

    def test_empty_string_gets_underscore(self):
        """Test that an empty string gets a leading underscore."""
        assert _sanitize_parameter_name("") == "_"


class TestBuildNameMapping:
    """Test cases for _build_name_mapping utility function."""

    def test_no_changes_when_names_are_valid(self):
        """Test that valid identifiers map to themselves."""
        result = _build_name_mapping(["name", "age"])
        assert result == {"name": "name", "age": "age"}

    def test_mapping_for_keywords(self):
        """Test mapping for Python keywords."""
        result = _build_name_mapping(["from", "class", "query"])
        assert result == {"from": "from_", "class": "class_", "query": "query"}

    def test_mapping_for_hyphenated_names(self):
        """Test mapping for hyphenated names."""
        result = _build_name_mapping(["cik-A", "name"])
        assert result == {"cik-A": "cik_A", "name": "name"}

    def test_collision_keyword_and_suffixed_name(self):
        """Test collision when schema has both 'from' and 'from_'."""
        result = _build_name_mapping(["from", "from_"])
        assert result["from_"] == "from_"
        assert result["from"] == "from_2"

    def test_collision_hyphenated_and_underscored(self):
        """Test collision when schema has both 'cik-A' and 'cik_A'."""
        result = _build_name_mapping(["cik-A", "cik_A"])
        assert result["cik_A"] == "cik_A"
        assert result["cik-A"] == "cik_A2"


class TestParameterNameSanitization:
    """Test that schemas with non-identifier field names produce valid wrappers."""

    def test_wrapper_created_for_schema_with_keyword_fields(self):
        """Test that create_function_wrapper succeeds with keyword field names."""
        schema = create_model("DateRangeSchema", **{
            "from": (str, ...),
            "class": (str, ...),
            "query": (str, ...),
        })  # type: ignore[call-overload]

        wrapper = create_function_wrapper("date_tool", _mock_session_manager(), schema)

        assert callable(wrapper)
        sig = getattr(wrapper, '__signature__', None)
        assert sig is not None
        assert "from_" in sig.parameters
        assert "class_" in sig.parameters
        assert "query" in sig.parameters

    def test_wrapper_created_for_schema_with_hyphenated_fields(self):
        """Test that create_function_wrapper succeeds with hyphenated field names."""
        schema = create_model("SecSchema", **{
            "cik-A": (str, ...),
            "name": (str, ...),
        })  # type: ignore[call-overload]

        wrapper = create_function_wrapper("sec_tool", _mock_session_manager(), schema)

        assert callable(wrapper)
        sig = getattr(wrapper, '__signature__', None)
        assert sig is not None
        assert "cik_A" in sig.parameters
        assert "name" in sig.parameters

    async def test_wrapper_reverse_maps_kwargs_for_validation(self):
        """Test that sanitized kwargs are reverse-mapped before Pydantic validation."""
        schema = create_model("ToolSchema", **{
            "from": (str, ...),
            "cik-A": (str, "default"),
        })  # type: ignore[call-overload]

        mock_sm = _mock_session_manager(result_value="ok")
        wrapper = create_function_wrapper("tool", mock_sm, schema)

        result = await wrapper(**{"from_": "2024-01-01", "cik_A": "ABC"})

        assert result == "ok"
        mock_sm.run.assert_called_once()
        payload = mock_sm.run.call_args[0][0]
        assert getattr(payload, "from") == "2024-01-01"
        assert getattr(payload, "cik-A") == "ABC"

    def test_annotations_use_sanitized_names(self):
        """Test that __annotations__ on the wrapper use sanitized names."""
        schema = create_model("Schema", **{"from": (str, ...)})  # type: ignore[call-overload]

        wrapper = create_function_wrapper("tool", _mock_session_manager(), schema)

        assert "from_" in wrapper.__annotations__
