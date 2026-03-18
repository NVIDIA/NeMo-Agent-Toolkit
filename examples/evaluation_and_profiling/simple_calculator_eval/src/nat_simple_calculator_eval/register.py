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

"""Custom components for the simple calculator evaluation example."""

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
    """Configuration for a helper function that calls `calculator__multiply`."""

    multiply_fn: FunctionRef = Field(
        default=FunctionRef("calculator__multiply"),
        description="Reference to the multiply function used internally.",
    )


@register_function(config_type=PowerOfTwoConfig)
async def power_of_two_function(config: PowerOfTwoConfig, builder: Builder):
    """Register a function that creates nested tool calls for trajectory inspection."""
    multiply_fn: Function = await builder.get_function(config.multiply_fn)

    async def _power_of_two(number: float) -> str:
        logger.info("power_of_two called with number=%s", number)
        result = await multiply_fn.ainvoke({"numbers": [number, number]})
        return f"The power of 2 of {number} is {result}."

    yield FunctionInfo.from_fn(
        _power_of_two,
        description=("Calculate a number raised to the power of 2. "
                     "This tool internally calls `calculator__multiply`."),
    )
