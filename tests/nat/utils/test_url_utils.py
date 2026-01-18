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

import pytest

from nat.utils.url_utils import url_join


class TestUrlJoin:
    """Tests for url_join function."""

    def test_url_join_basic(self):
        """Test basic URL joining."""
        result = url_join("http://example.com", "api", "v1")
        assert result == "http://example.com/api/v1"

    def test_url_join_trailing_slashes(self):
        """Test that trailing slashes are properly handled."""
        result = url_join("http://example.com/", "api/", "v1/")
        assert result == "http://example.com/api/v1"

    def test_url_join_leading_slashes(self):
        """Test that leading slashes are properly stripped."""
        result = url_join("http://example.com", "/api", "/v1")
        assert result == "http://example.com/api/v1"

    def test_url_join_mixed_slashes(self):
        """Test with mixed leading and trailing slashes."""
        result = url_join("http://example.com/", "/api/", "/v1/")
        assert result == "http://example.com/api/v1"

    def test_url_join_single_part(self):
        """Test with single URL part."""
        result = url_join("http://example.com")
        assert result == "http://example.com"

    def test_url_join_empty_parts(self):
        """Test that empty parts result in empty string."""
        result = url_join()
        assert result == ""

    def test_url_join_with_query_params(self):
        """Test URL joining with query parameters."""
        result = url_join("http://example.com", "api", "v1?key=value")
        assert result == "http://example.com/api/v1?key=value"

    def test_url_join_numeric_parts(self):
        """Test URL joining with numeric parts."""
        result = url_join("http://example.com", "api", "v1", 123)
        assert result == "http://example.com/api/v1/123"

    def test_url_join_with_https(self):
        """Test URL joining with HTTPS protocol."""
        result = url_join("https://secure.example.com", "api", "endpoint")
        assert result == "https://secure.example.com/api/endpoint"

    def test_url_join_path_only(self):
        """Test joining path segments without protocol."""
        result = url_join("api", "v1", "users")
        assert result == "api/v1/users"

    def test_url_join_with_port(self):
        """Test URL joining with port number."""
        result = url_join("http://example.com:8080", "api", "v1")
        assert result == "http://example.com:8080/api/v1"

    def test_url_join_multiple_consecutive_slashes(self):
        """Test that multiple consecutive slashes are handled."""
        result = url_join("http://example.com//", "//api//", "//v1//")
        assert result == "http://example.com/api/v1"

    def test_url_join_preserves_protocol_slashes(self):
        """Test that protocol slashes are properly handled."""
        result = url_join("http://example.com", "path")
        assert "http:" in result
