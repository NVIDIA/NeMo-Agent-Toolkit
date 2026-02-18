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

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorBaseConfig


class AccuracyEvaluatorConfig(EvaluatorBaseConfig, name="percent_games_won"):
    """Configuration for custom accuracy evaluator for RL with OpenPipe ART."""
    pass


@register_evaluator(config_type=AccuracyEvaluatorConfig)
async def register_accuracy_evaluator(config: AccuracyEvaluatorConfig, builder: EvalBuilder):
    """Register custom accuracy evaluator."""
    from .accuracy_evaluator import AccuracyEvaluator

    evaluator = AccuracyEvaluator(builder.get_max_concurrency())

    yield EvaluatorInfo(config=config,
                        evaluate_fn=evaluator.evaluate,
                        description="Custom accuracy evaluator for RL workflow outputs")


class AccuracyEvaluatorConfig(EvaluatorBaseConfig, name="step_value_computation"):
    """Configuration for custom accuracy evaluator for RL with OpenPipe ART."""
    pass


@register_evaluator(config_type=AccuracyEvaluatorConfig)
async def register_accuracy_evaluator_penalty(config: AccuracyEvaluatorConfig, builder: EvalBuilder):
    """Register custom accuracy evaluator."""
    from .accuracy_evaluator import AccuracyEvaluator

    evaluator = AccuracyEvaluator(builder.get_max_concurrency(), use_intermediate_steps=True)

    yield EvaluatorInfo(config=config,
                        evaluate_fn=evaluator.evaluate,
                        description="Custom accuracy evaluator for RL workflow outputs")
