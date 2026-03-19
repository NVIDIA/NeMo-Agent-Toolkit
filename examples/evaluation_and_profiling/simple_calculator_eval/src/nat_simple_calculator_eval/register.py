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


class SquareViaMultiplyConfig(FunctionBaseConfig, name="square_via_multiply"):
    """Configuration for an internal square helper tool."""

    multiply_fn: FunctionRef = Field(
        default=FunctionRef("calculator__multiply"),
        description="Reference to the multiply function used internally.",
    )


@register_function(config_type=SquareViaMultiplyConfig)
async def square_via_multiply_function(config: SquareViaMultiplyConfig, builder: Builder):
    """Register a helper tool that computes a square through multiply."""
    multiply_fn: Function = await builder.get_function(config.multiply_fn)

    async def _square_via_multiply(number: float) -> float:
        logger.info("square_via_multiply called with number=%s", number)
        result = await multiply_fn.ainvoke({"numbers": [number, number]})
        return float(result)

    yield FunctionInfo.from_fn(
        _square_via_multiply,
        description="Compute the square of a number using calculator multiplication.",
    )


class CubeViaMultiplyChainConfig(FunctionBaseConfig, name="cube_via_multiply_chain"):
    """Configuration for an internal cube helper tool."""

    multiply_fn: FunctionRef = Field(
        default=FunctionRef("calculator__multiply"),
        description="Reference to the multiply function used internally.",
    )


@register_function(config_type=CubeViaMultiplyChainConfig)
async def cube_via_multiply_chain_function(config: CubeViaMultiplyChainConfig, builder: Builder):
    """Register a helper tool that computes a cube via chained multiply calls."""
    multiply_fn: Function = await builder.get_function(config.multiply_fn)

    async def _cube_via_multiply_chain(number: float) -> float:
        logger.info("cube_via_multiply_chain called with number=%s", number)
        squared = await multiply_fn.ainvoke({"numbers": [number, number]})
        cubed = await multiply_fn.ainvoke({"numbers": [float(squared), number]})
        return float(cubed)

    yield FunctionInfo.from_fn(
        _cube_via_multiply_chain,
        description="Compute the cube of a number using chained calculator multiplication.",
    )


class PowerBranchConfig(FunctionBaseConfig, name="power_branch"):
    """Configuration for a branching tool that calls two internal tools."""

    square_fn: FunctionRef = Field(
        default=FunctionRef("square_via_multiply"),
        description="Reference to the square helper function.",
    )
    cube_fn: FunctionRef = Field(
        default=FunctionRef("cube_via_multiply_chain"),
        description="Reference to the cube helper function.",
    )


@register_function(config_type=PowerBranchConfig)
async def power_branch_function(config: PowerBranchConfig, builder: Builder):
    """Register a branching tool that fans out to square and cube helpers."""
    square_fn: Function = await builder.get_function(config.square_fn)
    cube_fn: Function = await builder.get_function(config.cube_fn)

    async def _power_branch(number: float) -> str:
        logger.info("power_branch called with number=%s", number)
        square = await square_fn.ainvoke({"number": number})
        cube = await cube_fn.ainvoke({"number": number})
        return f"For {number}: square={square}, cube={cube}."

    yield FunctionInfo.from_fn(
        _power_branch,
        description=("For one number, compute both square and cube. "
                     "This tool always fans out to `square_via_multiply` and "
                     "`cube_via_multiply_chain`, which both use `calculator__multiply`."),
    )
