# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from nat.builder.context import Context
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


class TestContextIntegration:
    """Tests for Context integration with latency sensitivity."""

    def test_default_sensitivity_is_medium(self):
        """Test that default latency sensitivity is MEDIUM."""
        ctx = Context.get()
        sensitivity = ctx.latency_sensitivity
        assert sensitivity == LatencySensitivity.MEDIUM

    def test_push_higher_priority_changes_sensitivity(self):
        """Test that pushing higher priority changes current sensitivity."""
        ctx = Context.get()

        # Default is MEDIUM
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

        # Push HIGH (higher priority)
        with ctx.push_latency_sensitivity(LatencySensitivity.HIGH):
            assert ctx.latency_sensitivity == LatencySensitivity.HIGH

        # Reverts to MEDIUM
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

    def test_push_lower_priority_keeps_current(self):
        """Test that pushing lower priority keeps current sensitivity."""
        ctx = Context.get()

        # Push HIGH first
        with ctx.push_latency_sensitivity(LatencySensitivity.HIGH):
            assert ctx.latency_sensitivity == LatencySensitivity.HIGH

            # Try to push LOW (lower priority) - should stay HIGH
            with ctx.push_latency_sensitivity(LatencySensitivity.LOW):
                assert ctx.latency_sensitivity == LatencySensitivity.HIGH

            # Still HIGH after inner context exits
            assert ctx.latency_sensitivity == LatencySensitivity.HIGH

        # Reverts to MEDIUM
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

    def test_deep_nesting_maintains_priority(self):
        """Test that deep nesting correctly maintains highest priority."""
        ctx = Context.get()

        # MEDIUM (default)
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

        with ctx.push_latency_sensitivity(LatencySensitivity.LOW):
            # LOW < MEDIUM, stays MEDIUM
            assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

            with ctx.push_latency_sensitivity(LatencySensitivity.HIGH):
                # HIGH > MEDIUM, becomes HIGH
                assert ctx.latency_sensitivity == LatencySensitivity.HIGH

                with ctx.push_latency_sensitivity(LatencySensitivity.MEDIUM):
                    # MEDIUM < HIGH, stays HIGH
                    assert ctx.latency_sensitivity == LatencySensitivity.HIGH

                    with ctx.push_latency_sensitivity(LatencySensitivity.LOW):
                        # LOW < HIGH, stays HIGH
                        assert ctx.latency_sensitivity == LatencySensitivity.HIGH

                    # Still HIGH
                    assert ctx.latency_sensitivity == LatencySensitivity.HIGH

                # Still HIGH
                assert ctx.latency_sensitivity == LatencySensitivity.HIGH

            # Back to MEDIUM
            assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

        # Back to MEDIUM
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

    def test_exception_in_context_still_pops(self):
        """Test that exceptions don't break stack management."""
        ctx = Context.get()

        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

        try:
            with ctx.push_latency_sensitivity(LatencySensitivity.HIGH):
                assert ctx.latency_sensitivity == LatencySensitivity.HIGH
                raise ValueError("test error")
        except ValueError:
            pass

        # Should revert to MEDIUM despite exception
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM
