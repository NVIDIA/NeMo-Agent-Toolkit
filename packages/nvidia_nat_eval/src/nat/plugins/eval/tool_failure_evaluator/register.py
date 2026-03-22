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

from pydantic import Field

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorBaseConfig


class ToolFailureEvaluatorConfig(EvaluatorBaseConfig, name="tool_failure"):  # type: ignore[call-arg]
    """Tool failure evaluator configuration."""

    max_concurrency: int = Field(default=8, description="Max concurrency for evaluation.")


@register_evaluator(config_type=ToolFailureEvaluatorConfig)
async def register_tool_failure_evaluator(config: ToolFailureEvaluatorConfig, builder: EvalBuilder):
    """Register the tool failure evaluator."""
    from .evaluator import ToolFailureEvaluator

    max_concurrency = config.max_concurrency or builder.get_max_concurrency()
    evaluator = ToolFailureEvaluator(max_concurrency=max_concurrency)

    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluator.evaluate,
        description="Tool call success rate (1.0 = no failures)",
    )
