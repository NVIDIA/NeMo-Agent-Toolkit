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
from inspect import Parameter
from inspect import signature

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
    keys = {"user_input", "reference", "response", "reference_contexts", "retrieved_contexts"}
    # Avoid passing unsupported optional fields if absent.
    return {k: getattr(sample, k) for k in keys if hasattr(sample, k)}


async def score_metric_result(metric: SimpleBaseMetric, sample: object) -> MetricResult:
    """Run one metric and return raw ragas ``MetricResult``.

    We first build a superset of possible sample fields, then filter kwargs by the
    concrete ``metric.ascore(...)`` signature so each metric only receives supported args.

    Examples:

    - ``AnswerAccuracy(self, user_input, response, reference)`` forwards ``user_input, response, reference``.
    - ``AnswerCorrectness(self, user_input, response, reference)`` forwards ``user_input, response, reference``.
    - ``AnswerRelevancy(self, user_input, response)`` forwards ``user_input, response``.
    - ``BleuScore(self, reference, response)`` forwards ``reference, response``.
    - ``ResponseGroundedness(self, response, retrieved_contexts)`` forwards ``response, retrieved_contexts``.

    """
    metric_kwargs = build_metric_kwargs(sample)
    params = signature(metric.ascore).parameters
    has_var_kwargs = any(p.kind is Parameter.VAR_KEYWORD for p in params.values())
    if not has_var_kwargs:
        metric_kwargs = {k: v for k, v in metric_kwargs.items() if k in params}
    return await metric.ascore(**metric_kwargs)
