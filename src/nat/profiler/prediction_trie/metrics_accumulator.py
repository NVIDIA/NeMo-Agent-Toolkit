# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

from nat.profiler.prediction_trie.data_models import PredictionMetrics


class MetricsAccumulator:
    """Accumulates samples and computes aggregated statistics."""

    def __init__(self) -> None:
        self._samples: list[float] = []

    def add_sample(self, value: float) -> None:
        """Add a sample value to the accumulator."""
        self._samples.append(value)

    def has_samples(self) -> bool:
        """Return True if any samples have been added."""
        return len(self._samples) > 0

    def compute_metrics(self) -> PredictionMetrics:
        """Compute aggregated metrics from accumulated samples."""
        if not self._samples:
            return PredictionMetrics()

        n = len(self._samples)
        mean_val = sum(self._samples) / n
        sorted_samples = sorted(self._samples)

        return PredictionMetrics(
            sample_count=n,
            mean=mean_val,
            p50=self._percentile(sorted_samples, 50),
            p90=self._percentile(sorted_samples, 90),
            p95=self._percentile(sorted_samples, 95),
        )

    @staticmethod
    def _percentile(sorted_data: list[float], pct: float) -> float:
        """Compute percentile using linear interpolation."""
        if not sorted_data:
            return 0.0
        if len(sorted_data) == 1:
            return sorted_data[0]
        k = (len(sorted_data) - 1) * (pct / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)
