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

"""Standalone callback contracts for prompt optimization."""

from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Mapping
from typing import Any
from typing import Protocol
from typing import TypeAlias

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

PromptSet: TypeAlias = Mapping[str, str]


class _StrictModel(BaseModel):
    """Base model for standalone prompt optimization contracts."""

    model_config = ConfigDict(extra="forbid")


class PromptMutationInput(_StrictModel):
    """Input payload for prompt mutation callbacks."""

    original_prompt: str
    objective: str
    oracle_feedback: str | None = None


class PromptRecombinationInput(PromptMutationInput):
    """Input payload for prompt recombination callbacks."""

    parent_b: str | None = None


class PromptEvaluationItem(_StrictModel):
    """Item-level feedback emitted by an evaluation callback."""

    evaluator_name: str | None = None
    score: float
    reasoning: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptEvaluationResult(_StrictModel):
    """Evaluation callback output for a single prompt candidate."""

    metric_scores: dict[str, float] = Field(default_factory=dict)
    item_feedback: list[PromptEvaluationItem] = Field(default_factory=list)


class PromptOptimizationHistoryEntry(_StrictModel):
    """History row captured for one candidate in one generation."""

    generation: int
    candidate_index: int
    scalar_fitness: float | None = None
    metric_scores: dict[str, float] = Field(default_factory=dict)


class PromptOptimizationResult(_StrictModel):
    """Final output shape for prompt optimization runs."""

    best_prompts: dict[str, str] = Field(default_factory=dict)
    history: list[PromptOptimizationHistoryEntry] = Field(default_factory=list)
    checkpoints: dict[int, dict[str, str]] = Field(default_factory=dict)


class PromptMutator(Protocol):
    """Async callback interface for generating mutated prompts."""

    def __call__(self, value: PromptMutationInput, /) -> Awaitable[str]:
        ...


class PromptRecombiner(Protocol):
    """Async callback interface for combining two parent prompts."""

    def __call__(self, value: PromptRecombinationInput, /) -> Awaitable[str]:
        ...


class PromptEvaluator(Protocol):
    """Async callback interface for scoring a prompt set."""

    def __call__(self, prompts: PromptSet, /) -> Awaitable[PromptEvaluationResult]:
        ...
