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
"""
Nested tool example for testing parent-child span tracking.

This module defines a `power_of_two` tool that internally calls the calculator's
multiply function, creating a nested tool call scenario for testing span lineage.
"""

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function import Function
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class PowerOfTwoConfig(FunctionBaseConfig, name="power_of_two"):
    """Configuration for the power_of_two function that wraps calculator__multiply."""

    multiply_fn: FunctionRef = Field(
        default=FunctionRef("calculator__multiply"),
        description="Reference to the multiply function to use internally.",
    )


@register_function(config_type=PowerOfTwoConfig)
async def power_of_two_function(config: PowerOfTwoConfig, builder: Builder):
    """
    Create a power_of_two function that internally calls calculator__multiply.

    This creates a nested tool call scenario:
    - react_agent calls power_of_two (parent_name = "react_agent")
    - power_of_two calls calculator__multiply (parent_name = "power_of_two")

    This allows testing of the parent_id and parent_name span attributes.
    """

    # Get the multiply function from the calculator function group
    multiply_fn: Function = await builder.get_function(config.multiply_fn)

    async def _power_of_two(number: float) -> str:
        """
        Calculate a number raised to the power of 2 by calling multiply internally.

        This is a wrapper tool that demonstrates nested tool calls.
        It internally calls the calculator's multiply function.

        Args:
            number: The number to square (raise to power of 2).

        Returns:
            A string describing the result of number^2.
        """
        logger.info("power_of_two called with number=%s, calling multiply internally", number)

        # Call multiply internally - this creates a nested tool call
        # The multiply function expects a list of numbers via .ainvoke()
        # Function objects are not directly callable - use .ainvoke() method
        result = await multiply_fn.ainvoke({"numbers": [number, number]})

        logger.info("multiply returned result=%s", result)

        return f"The power of 2 of {number} is {result} (computed via nested multiply call)"

    yield FunctionInfo.from_fn(
        _power_of_two,
        description=("Calculate a number raised to the power of 2. "
                     "This tool internally calls the multiply function, creating a nested tool call."),
    )
