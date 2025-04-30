# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class InequalityToolConfig(FunctionBaseConfig, name="calculator_inequality"):
    pass


@register_function(config_type=InequalityToolConfig)
async def calculator_inequality(tool_config: InequalityToolConfig, builder: Builder):

    import re

    async def _calculator_inequality(text: str) -> str:
        numbers = re.findall(r"\d+", text)
        a = int(numbers[0])
        b = int(numbers[1])

        if a > b:
            return f"First number {a} is greater than the second number {b}"
        if a < b:
            return f"First number {a} is less than the second number {b}"

        return f"First number {a} is equal to the second number {b}"

    # Create a Generic AIQ Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _calculator_inequality,
        description=("This is a mathematical tool used to perform an inequality comparison between two numbers. "
                     "It takes two numbers as an input and determines if one is greater or are equal."))


class MultiplyToolConfig(FunctionBaseConfig, name="calculator_multiply"):
    pass


@register_function(config_type=MultiplyToolConfig)
async def calculator_multiply(config: MultiplyToolConfig, builder: Builder):

    import re

    async def _calculator_multiply(text: str) -> str:
        numbers = re.findall(r"\d+", text)
        a = int(numbers[0])
        b = int(numbers[1])

        return f"The product of {a} * {b} is {a * b}"

    # Create a Generic AIQ Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _calculator_multiply,
        description=("This is a mathematical tool used to multiply two numbers together. "
                     "It takes 2 numbers as an input and computes their numeric product as the output."))


class DivisionToolConfig(FunctionBaseConfig, name="calculator_divide"):
    pass


@register_function(config_type=DivisionToolConfig)
async def calculator_divide(config: DivisionToolConfig, builder: Builder):

    import re

    async def _calculator_divide(text: str) -> str:
        numbers = re.findall(r"\d+", text)
        a = int(numbers[0])
        b = int(numbers[1])

        return f"The result of {a} / {b} is {a / b}"

    # Create a Generic AIQ Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _calculator_divide,
        description=("This is a mathematical tool used to divide one number by another. "
                     "It takes 2 numbers as an input and computes their numeric quotient as the output."))


class SubtractToolConfig(FunctionBaseConfig, name="calculator_subtract"):
    pass


@register_function(config_type=SubtractToolConfig)
async def calculator_subtract(config: SubtractToolConfig, builder: Builder):

    import re

    async def _calculator_subtract(text: str) -> str:
        numbers = re.findall(r"\d+", text)
        a = int(numbers[0])
        b = int(numbers[1])

        return f"The result of {a} - {b} is {a - b}"

    # Create a Generic AIQ Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _calculator_subtract,
        description=("This is a mathematical tool used to subtract one number from another. "
                     "It takes 2 numbers as an input and computes their numeric difference as the output."))
