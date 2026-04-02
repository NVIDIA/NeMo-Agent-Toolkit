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
"""Evaluation input data models and evaluator configs."""

import typing
from collections.abc import Sequence
from typing import Protocol
from typing import runtime_checkable

from pydantic import BaseModel
from pydantic import Field

from .common import BaseModelRegistryTag
from .common import TypedBaseModel
from .intermediate_step import IntermediateStep
from .retry_mixin import RetryMixin


class EvaluatorBaseConfig(TypedBaseModel, BaseModelRegistryTag):
    pass


class EvaluatorLLMConfig(EvaluatorBaseConfig, RetryMixin):
    """Base config for evaluators that use an LLM as a judge."""

    llm_name: str = Field(description="LLM to use as a judge.")


EvaluatorBaseConfigT = typing.TypeVar("EvaluatorBaseConfigT", bound=EvaluatorBaseConfig)


class EvalInputItem(BaseModel):
    """A single input item for evaluation."""

    id: typing.Any = Field(description="Unique identifier for this evaluation item.")
    input_obj: typing.Any = Field(description="The input to the workflow (e.g., user question).")
    expected_output_obj: typing.Any = Field(description="The expected/ground truth output.")
    output_obj: typing.Any = Field(default=None, description="The actual workflow output. Populated during evaluation.")
    expected_trajectory: list[IntermediateStep] = Field(
        default_factory=list,
        description="Expected intermediate steps for trajectory evaluation.",
    )
    trajectory: list[IntermediateStep] = Field(
        default_factory=list,
        description="Actual intermediate steps from workflow execution. Populated during evaluation.",
    )
    full_dataset_entry: typing.Any = Field(description="The complete original dataset entry.")

    def copy_with_updates(self, **updates) -> "EvalInputItem":
        """Copy EvalInputItem with optional field updates."""
        item_data = self.model_dump()
        item_data.update(updates)
        return EvalInputItem(**item_data)


class EvalInput(BaseModel):
    """Container for evaluation input items."""

    eval_input_items: list[EvalInputItem] = Field(description="List of items to evaluate.")


@runtime_checkable
class EvalOutputItemLike(Protocol):
    """Structural contract for a single evaluation output item."""

    id: typing.Any
    score: typing.Any
    reasoning: typing.Any
    error: str | None


@runtime_checkable
class EvalOutputLike(Protocol):
    """Structural contract for a collection of evaluation output items."""

    average_score: typing.Any
    eval_output_items: Sequence[EvalOutputItemLike]
