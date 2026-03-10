# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from ragas.metrics.base import SimpleBaseMetric
from ragas.metrics.result import MetricResult


def nan_to_zero(v: float | None) -> float:
    """Convert NaN or None to 0.0 for safe arithmetic/serialization."""
    return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else v


def extract_metric_score(metric_result: MetricResult) -> float | None:
    """Extract scalar score from a ragas metric result object."""
    if not isinstance(metric_result, MetricResult):
        raise TypeError(f"Expected ragas MetricResult, got {type(metric_result).__name__}.")

    value = metric_result.value
    if value is None:
        return None
    if isinstance(value, int | float):
        return value
    raise TypeError(f"MetricResult.value must be numeric or None, got {type(value).__name__}.")


def build_metric_kwargs(sample: object) -> dict[str, str | list[str]]:
    """Build kwargs payload for `metric.ascore(**kwargs)` from a ragas sample."""
    kwargs = {
        "user_input": getattr(sample, "user_input", None),
        "reference": getattr(sample, "reference", None),
        "response": getattr(sample, "response", None),
        "reference_contexts": getattr(sample, "reference_contexts", None),
        "retrieved_contexts": getattr(sample, "retrieved_contexts", None),
    }
    # Avoid passing unsupported optional fields if absent.
    return {k: v for k, v in kwargs.items() if v is not None}


async def score_metric_result(metric: SimpleBaseMetric, sample: object) -> MetricResult:
    """Run one metric and return raw ragas `MetricResult`."""
    return await metric.ascore(**build_metric_kwargs(sample))
