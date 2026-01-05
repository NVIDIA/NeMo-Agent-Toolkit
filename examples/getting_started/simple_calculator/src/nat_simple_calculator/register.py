# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig


class CalculatorToolConfig(FunctionGroupBaseConfig, name="calculator"):
    include: list[str] = Field(default_factory=lambda: ["add", "subtract", "multiply", "divide", "compare"],
                               description="The list of functions to include in the calculator function group.")


@register_function_group(config_type=CalculatorToolConfig)
async def calculator(_config: CalculatorToolConfig, _builder: Builder) -> AsyncGenerator[FunctionGroup, None]:
    """Create and register the calculator function group.

    Args:
        _config: Calculator function group configuration (unused).
        _builder: Workflow builder (unused).

    Yields:
        FunctionGroup: The configured calculator function group with add, subtract,
            multiply, divide, and compare operations.
    """
    import math

    group = FunctionGroup(config=_config)

    async def _add(numbers: list[float]) -> float:
        """Add two or more numbers together."""
        if len(numbers) < 2:
            raise ValueError("This tool only supports addition between two or more numbers.")
        return sum(numbers)

    async def _subtract(numbers: list[float]) -> float:
        """Subtract one number from another."""
        if len(numbers) != 2:
            raise ValueError("This tool only supports subtraction between two numbers.")
        a, b = numbers
        return a - b

    async def _multiply(numbers: list[float]) -> float:
        """Multiply two or more numbers together."""
        if len(numbers) < 2:
            raise ValueError("This tool only supports multiplication between two or more numbers.")
        return math.prod(numbers)

    async def _divide(numbers: list[float]) -> float:
        """Divide one number by another."""
        if len(numbers) != 2:
            raise ValueError("This tool only supports division between two numbers.")
        a, b = numbers
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    async def _compare(numbers: list[float]) -> str:
        """Compare two numbers."""
        if len(numbers) != 2:
            raise ValueError("This tool only supports comparison between two numbers.")
        a, b = numbers
        if a > b:
            return f"{a} is greater than {b}"
        if a < b:
            return f"{a} is less than {b}"
        return f"{a} is equal to {b}"

    group.add_function(name="add", fn=_add, description=_add.__doc__)
    group.add_function(name="subtract", fn=_subtract, description=_subtract.__doc__)
    group.add_function(name="multiply", fn=_multiply, description=_multiply.__doc__)
    group.add_function(name="divide", fn=_divide, description=_divide.__doc__)
    group.add_function(name="compare", fn=_compare, description=_compare.__doc__)

    yield group
