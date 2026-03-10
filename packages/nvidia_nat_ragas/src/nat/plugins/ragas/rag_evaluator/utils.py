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

import inspect
import math
from typing import Any


def nan_to_zero(v: float | None) -> float:
    """Convert NaN or None to 0.0 for safe arithmetic/serialization."""
    return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else v


def extract_metric_score(metric_result: Any) -> float | None:
    """Extract scalar score from a ragas metric result object."""
    # v0.4 collections metrics return a result object with `value`.
    if hasattr(metric_result, "value"):
        return getattr(metric_result, "value")
    # Legacy-style or fallback score outputs.
    if isinstance(metric_result, int | float):
        return metric_result
    if isinstance(metric_result, dict):
        value = metric_result.get("value")
        if isinstance(value, int | float):
            return value
    return None


def build_metric_kwargs(sample: Any) -> dict[str, Any]:
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


async def score_metric(metric: Any, sample: Any) -> float | None:
    """Score a single sample with one metric via v0.4-style async API."""
    if not hasattr(metric, "ascore"):
        raise TypeError(f"Metric '{getattr(metric, 'name', type(metric).__name__)}' does not implement ascore().")

    score_result = metric.ascore(**build_metric_kwargs(sample))
    if inspect.isawaitable(score_result):
        score_result = await score_result

    return extract_metric_score(score_result)
