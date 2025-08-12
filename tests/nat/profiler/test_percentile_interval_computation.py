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

import math
import statistics
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nat.profiler.inference_metrics_model import InferenceMetricsModel
from nat.profiler.profile_runner import ProfilerRunner

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _percentile_reference(arr: list[float], pct: float) -> float:
    """
    Reference percentile implementation mirroring the one in
    _compute_confidence_intervals for cross-checking.
    * pct is in the range [0, 1] – e.g. 0.90 for p90.
    """
    if not arr:
        return 0.0
    k = (len(arr) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return arr[int(k)]
    return arr[f] + (arr[c] - arr[f]) * (k - f)


@pytest.fixture
def runner(tmp_path) -> ProfilerRunner:
    """A ProfilerRunner pointing at a temp directory."""
    return ProfilerRunner(MagicMock(), Path(tmp_path))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_empty_input_returns_defaults(runner: ProfilerRunner):
    """Empty data → model with default values."""
    result = runner._compute_confidence_intervals([], "dummy")
    assert isinstance(result, InferenceMetricsModel)
    assert result.n == 0
    assert result.mean == 0
    assert result.ninetieth_interval == (0, 0)
    assert result.ninety_fifth_interval == (0, 0)
    assert result.ninety_ninth_interval == (0, 0)
    assert result.p90 == 0
    assert result.p95 == 0
    assert result.p99 == 0


def test_single_value_collapses_intervals_and_percentiles(runner: ProfilerRunner):
    """Single sample: all intervals collapse to the mean."""
    value = 5.0
    res = runner._compute_confidence_intervals([value], "single-point")
    assert res.n == 1
    assert res.mean == pytest.approx(value)
    assert res.ninetieth_interval == (value, value)
    assert res.ninety_fifth_interval == (value, value)
    assert res.ninety_ninth_interval == (value, value)
    assert res.p90 == value
    assert res.p95 == value
    assert res.p99 == value


def test_multiple_values_compute_correct_stats(runner: ProfilerRunner):
    """Validate mean, CI bounds, and percentiles for a small dataset."""
    data = [1, 2, 3, 4, 5]
    res = runner._compute_confidence_intervals(data, "multi-point")

    # mean
    expected_mean = statistics.mean(data)
    assert res.mean == pytest.approx(expected_mean)

    # percentiles
    sorted_data = sorted(data)
    assert res.p90 == pytest.approx(_percentile_reference(sorted_data, 0.90))
    assert res.p95 == pytest.approx(_percentile_reference(sorted_data, 0.95))
    assert res.p99 == pytest.approx(_percentile_reference(sorted_data, 0.99))

    # 90 % confidence interval bounds
    stdev_val = statistics.pstdev(data)
    se = stdev_val / math.sqrt(len(data))
    z_90 = 1.645
    lower_90 = expected_mean - z_90 * se
    upper_90 = expected_mean + z_90 * se
    assert res.ninetieth_interval[0] == pytest.approx(lower_90)
    assert res.ninetieth_interval[1] == pytest.approx(upper_90)
