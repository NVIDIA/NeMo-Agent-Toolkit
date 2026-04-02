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
"""ATIF-native evaluator protocol definitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from pydantic import BaseModel
from pydantic import Field

from nat.atif import ATIFTrajectory
from nat.plugins.eval.data_models.evaluator_io import EvalOutput


class AtifEvalSample(BaseModel):
    """ATIF-native evaluation sample used by ATIF-backed evaluators."""

    item_id: Any = Field(description="Identifier matching the source EvalInputItem.")
    trajectory: ATIFTrajectory = Field(description="Canonical ATIF trajectory.")
    expected_output_obj: Any = Field(default=None, description="Optional expected output reference.")
    output_obj: Any = Field(default=None, description="Optional workflow output reference.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional evaluator metadata.")


AtifEvalSampleList = Sequence[AtifEvalSample]


@runtime_checkable
class AtifEvaluator(Protocol):
    """Protocol for evaluators that consume ATIF-native samples."""

    async def evaluate_atif_fn(self, atif_samples: AtifEvalSampleList) -> EvalOutput:
        """Evaluate using ATIF-native sample payloads."""
        ...


@runtime_checkable
class LegacyEvaluator(Protocol):
    """Protocol for evaluators that consume legacy `EvalInput` payloads."""

    async def evaluate_fn(self, eval_input) -> EvalOutput:
        """Evaluate using legacy eval input payloads."""
        ...
