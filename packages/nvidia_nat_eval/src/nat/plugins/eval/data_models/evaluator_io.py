# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Evaluation output models owned by the eval plugin."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SerializeAsAny


class EvalOutputItem(BaseModel):
    """A single output item from evaluation."""

    model_config = ConfigDict(exclude_none=True)  # pyright: ignore[reportCallIssue]

    id: Any = Field(description="Identifier matching the corresponding input item.")
    score: Any = Field(description="Evaluation score (typically float, may be NaN on failure).")
    reasoning: Any = Field(description="Evaluation context and LLM judge explanation.")
    error: str | None = Field(default=None, description="Evaluation error message if this item failed.")


class EvalOutput(BaseModel):
    """Container for evaluation output items."""

    average_score: Any = Field(description="Average score across all evaluated items.")
    eval_output_items: list[SerializeAsAny[EvalOutputItem]] = Field(description="List of evaluation results.")

    @property
    def num_items(self) -> int:
        """Number of evaluated items."""
        return len(self.eval_output_items)

    @property
    def num_errors(self) -> int:
        """Number of items whose evaluation failed (`error` is set).

        Failed items are scored 0.0 and still count towards `average_score`, so a non-zero value means the
        average is lower than the workflow's answers alone would warrant.
        """
        return count_evaluator_errors(self)


def count_evaluator_errors(eval_output: Any) -> int:
    """Count the items of an evaluator output whose `error` is set.

    Works on `EvalOutput` and on any object exposing `eval_output_items` with item-level `error` attributes;
    anything else counts as having no errors.
    """
    items = getattr(eval_output, "eval_output_items", None) or []
    return sum(1 for item in items if getattr(item, "error", None) is not None)
