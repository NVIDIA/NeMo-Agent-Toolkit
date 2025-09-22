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

from nat.observability.utils.time_utils import ns_timestamp


def test_ns_timestamp_basic():
    """Test basic timestamp conversion from seconds to nanoseconds."""
    seconds = 1.0
    result = ns_timestamp(seconds)
    assert result == 1_000_000_000
    assert isinstance(result, int)


def test_ns_timestamp_zero():
    """Test timestamp conversion with zero seconds."""
    seconds = 0.0
    result = ns_timestamp(seconds)
    assert result == 0
    assert isinstance(result, int)


def test_ns_timestamp_fractional_seconds():
    """Test timestamp conversion with fractional seconds."""
    seconds = 1.5
    result = ns_timestamp(seconds)
    assert result == 1_500_000_000
    assert isinstance(result, int)


def test_ns_timestamp_small_fractional():
    """Test timestamp conversion with small fractional seconds."""
    seconds = 0.001  # 1 millisecond
    result = ns_timestamp(seconds)
    assert result == 1_000_000  # 1 million nanoseconds
    assert isinstance(result, int)


def test_ns_timestamp_microseconds():
    """Test timestamp conversion with microsecond precision."""
    seconds = 0.000001  # 1 microsecond
    result = ns_timestamp(seconds)
    assert result == 1_000  # 1000 nanoseconds
    assert isinstance(result, int)


def test_ns_timestamp_nanoseconds():
    """Test timestamp conversion with nanosecond precision."""
    seconds = 0.000000001  # 1 nanosecond
    result = ns_timestamp(seconds)
    assert result == 1
    assert isinstance(result, int)


def test_ns_timestamp_large_value():
    """Test timestamp conversion with large values."""
    seconds = 1234567890.123456789
    result = ns_timestamp(seconds)
    expected = int(1234567890.123456789 * 1e9)
    assert result == expected
    assert isinstance(result, int)


def test_ns_timestamp_negative_value():
    """Test timestamp conversion with negative values."""
    seconds = -1.5
    result = ns_timestamp(seconds)
    assert result == -1_500_000_000
    assert isinstance(result, int)


def test_ns_timestamp_precision_loss():
    """Test that conversion handles floating point precision correctly."""
    # Test with a value that might have floating point precision issues
    seconds = 1.0000000001
    result = ns_timestamp(seconds)
    # Due to floating point precision, this should be close to but not exactly 1000000000.1
    expected = int(1.0000000001 * 1e9)
    assert result == expected
    assert isinstance(result, int)


def test_ns_timestamp_unix_epoch():
    """Test timestamp conversion with typical Unix epoch timestamps."""
    # January 1, 2024 00:00:00 UTC (approximate)
    seconds = 1704067200.0
    result = ns_timestamp(seconds)
    assert result == 1704067200_000_000_000
    assert isinstance(result, int)


def test_ns_timestamp_high_precision():
    """Test timestamp conversion with high precision fractional seconds."""
    seconds = 1.123456789
    result = ns_timestamp(seconds)
    expected = int(1.123456789 * 1e9)
    assert result == expected
    assert isinstance(result, int)


def test_ns_timestamp_edge_cases():
    """Test timestamp conversion with various edge cases."""
    # Very small positive value
    result = ns_timestamp(1e-10)
    assert result == 0  # Should round down to 0

    # Very small negative value
    result = ns_timestamp(-1e-10)
    assert result == 0  # Should round up to 0

    # Test with integer input (should work fine)
    result = ns_timestamp(5)
    assert result == 5_000_000_000
    assert isinstance(result, int)


@pytest.mark.parametrize("seconds,expected",
                         [
                             (0.0, 0),
                             (1.0, 1_000_000_000),
                             (0.5, 500_000_000),
                             (2.5, 2_500_000_000),
                             (0.001, 1_000_000),
                             (0.000001, 1_000),
                             (0.000000001, 1),
                             (-1.0, -1_000_000_000),
                             (-0.5, -500_000_000),
                         ])
def test_ns_timestamp_parametrized(seconds, expected):
    """Parametrized test for various timestamp conversion scenarios."""
    result = ns_timestamp(seconds)
    assert result == expected
    assert isinstance(result, int)


def test_ns_timestamp_extreme_edge_cases():
    """Test timestamp conversion with extreme edge cases."""
    # Test with infinity - should raise an exception or handle gracefully
    with pytest.raises((ValueError, OverflowError)):
        ns_timestamp(float('inf'))

    with pytest.raises((ValueError, OverflowError)):
        ns_timestamp(float('-inf'))

    # Test with NaN - should raise an exception or handle gracefully
    with pytest.raises((ValueError, TypeError)):
        ns_timestamp(float('nan'))


def test_ns_timestamp_very_large_numbers():
    """Test timestamp conversion with very large numbers that might cause overflow."""
    # Test with a very large number that should still work
    large_seconds = 1e15  # 1 quadrillion seconds
    result = ns_timestamp(large_seconds)
    expected = int(1e15 * 1e9)  # 1e24
    assert result == expected
    assert isinstance(result, int)


def test_ns_timestamp_type_validation():
    """Test that function works with different numeric types."""
    # Test with int (should work)
    result = ns_timestamp(5)
    assert result == 5_000_000_000
    assert isinstance(result, int)

    # Test with numpy types if available
    try:
        import numpy as np
        result = ns_timestamp(np.float64(1.5))
        assert result == 1_500_000_000
        assert isinstance(result, int)
    except ImportError:
        # Skip numpy test if not available
        pass
