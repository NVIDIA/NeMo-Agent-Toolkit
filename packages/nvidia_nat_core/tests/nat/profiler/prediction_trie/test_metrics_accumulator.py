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

from nat.profiler.prediction_trie.metrics_accumulator import MetricsAccumulator


def test_accumulator_add_single_sample():
    acc = MetricsAccumulator()
    acc.add_sample(10.0)
    metrics = acc.compute_metrics()
    assert metrics.sample_count == 1
    assert metrics.mean == 10.0
    assert metrics.p50 == 10.0
    assert metrics.p90 == 10.0
    assert metrics.p95 == 10.0


def test_accumulator_add_multiple_samples():
    acc = MetricsAccumulator()
    for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        acc.add_sample(v)
    metrics = acc.compute_metrics()
    assert metrics.sample_count == 10
    assert metrics.mean == 5.5
    assert metrics.p50 == 5.5  # median of 1-10
    assert metrics.p90 == 9.1  # 90th percentile
    assert metrics.p95 == pytest.approx(9.55)  # 95th percentile


def test_accumulator_empty():
    acc = MetricsAccumulator()
    metrics = acc.compute_metrics()
    assert metrics.sample_count == 0
    assert metrics.mean == 0.0
