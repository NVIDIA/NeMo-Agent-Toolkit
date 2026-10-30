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

import logging
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_evaluation_run() -> type:
    """Lazily load eval runtime class required by `nat optimize`."""
    try:
        from nat.plugins.eval.runtime.evaluate import EvaluationRun
        return EvaluationRun
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The `nat optimize` command requires evaluation support from `nvidia-nat-eval`. "
            "Install it with `uv pip install nvidia-nat-eval` (or `pip install nvidia-nat-eval`).") from exc


def warn_on_evaluator_errors(evaluation_results: Iterable[tuple[str, Any]], *, context: str) -> None:
    """Log a warning for every evaluator whose output contains failed items.

    Failed items are scored 0.0 and counted in `average_score` (the optimization objective) as if they had been
    judged, so they are indistinguishable from a poor answer. The warning names the evaluator and the failure
    count so a transient judge outage is not mistaken for a bad configuration. Outputs without
    `eval_output_items` are ignored.
    """
    for evaluator_name, eval_output in evaluation_results:
        items = getattr(eval_output, "eval_output_items", None) or []
        num_errors = sum(1 for item in items if getattr(item, "error", None) is not None)
        if num_errors:
            logger.warning(
                "%s: evaluator '%s' failed on %d/%d items; the failed items are counted in its average_score",
                context,
                evaluator_name,
                num_errors,
                len(items))
