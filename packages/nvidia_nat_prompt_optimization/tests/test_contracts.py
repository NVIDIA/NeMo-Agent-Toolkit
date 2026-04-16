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

from nvidia_prompt_optimization import PromptEvaluationItem
from nvidia_prompt_optimization import PromptEvaluationResult
from nvidia_prompt_optimization import PromptMutationInput
from nvidia_prompt_optimization import PromptOptimizationHistoryEntry
from nvidia_prompt_optimization import PromptOptimizationResult
from nvidia_prompt_optimization import PromptRecombinationInput


def test_prompt_mutation_input_contract():
    value = PromptMutationInput(original_prompt="Base", objective="Improve accuracy")

    assert value.original_prompt == "Base"
    assert value.objective == "Improve accuracy"
    assert value.oracle_feedback is None


def test_prompt_recombination_input_contract():
    value = PromptRecombinationInput(
        original_prompt="Parent A",
        parent_b="Parent B",
        objective="Improve accuracy",
    )

    assert value.original_prompt == "Parent A"
    assert value.parent_b == "Parent B"


def test_prompt_evaluation_result_contract():
    result = PromptEvaluationResult(
        metric_scores={"accuracy": 0.8},
        item_feedback=[
            PromptEvaluationItem(
                evaluator_name="accuracy",
                score=0.2,
                reasoning="Failed on edge case",
            )
        ],
    )

    assert result.metric_scores["accuracy"] == 0.8
    assert result.item_feedback[0].reasoning == "Failed on edge case"


def test_prompt_optimization_result_contract():
    result = PromptOptimizationResult(
        best_prompts={"system_prompt": "Be direct."},
        history=[
            PromptOptimizationHistoryEntry(
                generation=1,
                candidate_index=0,
                scalar_fitness=0.75,
                metric_scores={"accuracy": 0.75},
            )
        ],
        checkpoints={1: {"system_prompt": "Be direct."}},
    )

    assert result.best_prompts["system_prompt"] == "Be direct."
    assert result.history[0].generation == 1
    assert result.checkpoints[1]["system_prompt"] == "Be direct."
