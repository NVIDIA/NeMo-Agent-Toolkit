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
"""
Evaluator registration for DPO Tic-Tac-Toe workflow.
"""

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorBaseConfig


class GameOutcomeEvaluatorConfig(EvaluatorBaseConfig, name="dpo_game_outcome"):
    """Configuration for game outcome evaluator."""

    pass


@register_evaluator(config_type=GameOutcomeEvaluatorConfig)
async def register_game_outcome_evaluator(config: GameOutcomeEvaluatorConfig, builder: EvalBuilder):
    """Register the game outcome evaluator."""
    from .evaluator import GameOutcomeEvaluator

    evaluator = GameOutcomeEvaluator(builder.get_max_concurrency())

    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluator.evaluate,
        description="Evaluates game outcomes (Win/Lose/Draw)",
    )


class DPODataCollectorEvaluatorConfig(EvaluatorBaseConfig, name="dpo_data_collector"):
    """Configuration for DPO data collector evaluator."""

    pass


@register_evaluator(config_type=DPODataCollectorEvaluatorConfig)
async def register_dpo_data_collector_evaluator(config: DPODataCollectorEvaluatorConfig, builder: EvalBuilder):
    """Register the DPO data collector evaluator."""
    from .evaluator import DPODataCollectorEvaluator

    evaluator = DPODataCollectorEvaluator(builder.get_max_concurrency())

    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluator.evaluate,
        description="Collects DPO preference pairs from intermediate steps",
    )
