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

from collections.abc import AsyncGenerator
from typing import Any

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig


class LogicEvaluatorConfig(FunctionGroupBaseConfig, name="logic_evaluator"):
    # Using a lambda so that each config instance receives a unique include list
    include: list[str] = Field(default_factory=lambda: ["if_then_else", "evaluate_condition"],
                               description="The list of functions to include in the logic evaluator function group.")


@register_function_group(config_type=LogicEvaluatorConfig)
async def logic_evaluator(_config: LogicEvaluatorConfig, _builder: Builder) -> AsyncGenerator[FunctionGroup, None]:
    """Create and register the logic evaluator function group.

    Args:
        _config: Logic evaluator function group configuration.
        _builder: Workflow builder (unused).

    Yields:
        FunctionGroup: The configured logic evaluator function group.
    """
    group = FunctionGroup(config=_config)

    async def _if_then_else(condition: bool, true_value: Any, false_value: Any) -> Any:
        """Return true_value if condition is True, otherwise return false_value."""
        return true_value if condition else false_value

    async def _evaluate_condition(value1: Any, operator: str, value2: Any) -> bool:
        """Evaluate a comparison between two values."""
        if operator == "==":
            return value1 == value2
        elif operator == "!=":
            return value1 != value2
        elif operator == ">":
            return value1 > value2
        elif operator == "<":
            return value1 < value2
        elif operator == ">=":
            return value1 >= value2
        elif operator == "<=":
            return value1 <= value2
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    group.add_function(name="if_then_else", fn=_if_then_else, description=_if_then_else.__doc__)
    group.add_function(name="evaluate_condition", fn=_evaluate_condition, description=_evaluate_condition.__doc__)

    yield group
