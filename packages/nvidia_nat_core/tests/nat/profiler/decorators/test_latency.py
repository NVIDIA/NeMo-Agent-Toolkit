# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nat.profiler.decorators.latency import LatencySensitivity


class TestLatencySensitivity:
    """Tests for LatencySensitivity enum."""

    def test_enum_values_exist(self):
        """Test that all three sensitivity levels exist."""
        assert LatencySensitivity.LOW
        assert LatencySensitivity.MEDIUM
        assert LatencySensitivity.HIGH

    def test_enum_string_values(self):
        """Test that enum values are correct strings."""
        assert LatencySensitivity.LOW.value == "LOW"
        assert LatencySensitivity.MEDIUM.value == "MEDIUM"
        assert LatencySensitivity.HIGH.value == "HIGH"

    def test_priority_values(self):
        """Test that priority property returns correct numeric values."""
        assert LatencySensitivity.LOW.priority == 1
        assert LatencySensitivity.MEDIUM.priority == 2
        assert LatencySensitivity.HIGH.priority == 3

    def test_parse_enum_value(self):
        """Test that parse accepts enum values directly."""
        result = LatencySensitivity.parse(LatencySensitivity.HIGH)
        assert result == LatencySensitivity.HIGH

    def test_parse_uppercase_string(self):
        """Test that parse accepts uppercase strings."""
        assert LatencySensitivity.parse("LOW") == LatencySensitivity.LOW
        assert LatencySensitivity.parse("MEDIUM") == LatencySensitivity.MEDIUM
        assert LatencySensitivity.parse("HIGH") == LatencySensitivity.HIGH

    def test_parse_lowercase_string(self):
        """Test that parse accepts lowercase strings (case-insensitive)."""
        assert LatencySensitivity.parse("low") == LatencySensitivity.LOW
        assert LatencySensitivity.parse("medium") == LatencySensitivity.MEDIUM
        assert LatencySensitivity.parse("high") == LatencySensitivity.HIGH

    def test_parse_mixed_case_string(self):
        """Test that parse accepts mixed-case strings."""
        assert LatencySensitivity.parse("Low") == LatencySensitivity.LOW
        assert LatencySensitivity.parse("MeDiUm") == LatencySensitivity.MEDIUM
        assert LatencySensitivity.parse("HiGh") == LatencySensitivity.HIGH

    def test_parse_invalid_string(self):
        """Test that parse raises ValueError for invalid strings."""
        with pytest.raises(ValueError, match="Invalid latency sensitivity"):
            LatencySensitivity.parse("INVALID")

    def test_parse_invalid_type(self):
        """Test that parse raises ValueError for invalid types."""
        with pytest.raises(ValueError, match="Invalid latency sensitivity"):
            LatencySensitivity.parse(123)

    def test_parse_empty_string(self):
        """Test that parse raises ValueError for empty string."""
        with pytest.raises(ValueError, match="Invalid latency sensitivity"):
            LatencySensitivity.parse("")
