# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nat_simple_calculator.register import DivisionToolConfig
from nat_simple_calculator.register import InequalityToolConfig
from nat_simple_calculator.register import MultiplyToolConfig
from nat_simple_calculator.register import SubtractToolConfig

from nat.test.tool_test_runner import ToolTestRunner


async def test_inequality_tool():
    """Test inequality tool logic directly."""

    runner = ToolTestRunner()
    await runner.test_tool(config_type=InequalityToolConfig,
                           input_data="Is 8 greater than 15?",
                           expected_output="First number 8 is less than the second number 15")


async def test_inequality_tool_equal_case():
    """Test inequality tool with equal numbers."""

    runner = ToolTestRunner()
    await runner.test_tool(config_type=InequalityToolConfig,
                           input_data="Compare 5 and 5",
                           expected_output="First number 5 is equal to the second number 5")


async def test_inequality_tool_greater_case():
    """Test inequality tool with first number greater."""

    runner = ToolTestRunner()
    await runner.test_tool(config_type=InequalityToolConfig,
                           input_data="Is 15 greater than 8?",
                           expected_output="First number 15 is greater than the second number 8")


async def test_multiply_tool():
    """Test multiply tool logic directly."""

    runner = ToolTestRunner()
    await runner.test_tool(config_type=MultiplyToolConfig,
                           input_data="What is 2 times 4?",
                           expected_output="The product of 2 * 4 is 8")


async def test_multiply_tool_edge_cases():
    """Test multiply tool with various inputs."""

    runner = ToolTestRunner()

    # Test with zero
    await runner.test_tool(config_type=MultiplyToolConfig,
                           input_data="Multiply 0 and 5",
                           expected_output="The product of 0 * 5 is 0")

    # Test with larger numbers
    await runner.test_tool(config_type=MultiplyToolConfig,
                           input_data="Calculate 12 times 13",
                           expected_output="The product of 12 * 13 is 156")


async def test_division_tool():
    """Test division tool logic directly."""

    runner = ToolTestRunner()
    await runner.test_tool(config_type=DivisionToolConfig,
                           input_data="What is 8 divided by 2?",
                           expected_output="The result of 8 / 2 is 4.0")


async def test_division_tool_with_remainder():
    """Test division with decimal result."""

    runner = ToolTestRunner()
    await runner.test_tool(config_type=DivisionToolConfig,
                           input_data="Divide 7 by 2",
                           expected_output="The result of 7 / 2 is 3.5")


async def test_subtract_tool():
    """Test subtract tool logic directly."""

    runner = ToolTestRunner()
    await runner.test_tool(config_type=SubtractToolConfig,
                           input_data="What is 10 minus 3?",
                           expected_output="The result of 10 - 3 is 7")


async def test_subtract_tool_negative_result():
    """Test subtract tool with negative result."""

    runner = ToolTestRunner()
    await runner.test_tool(config_type=SubtractToolConfig,
                           input_data="Subtract 15 from 10",
                           expected_output="The result of 15 - 10 is 5")


async def test_tool_error_handling():
    """Test error handling for insufficient numbers."""

    runner = ToolTestRunner()
    result = await runner.test_tool(config_type=MultiplyToolConfig, input_data="Multiply just one number: 5")

    # Should return an error message
    assert "Provide at least 2 numbers" in result


async def test_tool_validation_too_many_numbers():
    """Test validation for too many numbers."""

    runner = ToolTestRunner()
    result = await runner.test_tool(config_type=MultiplyToolConfig, input_data="Multiply 2, 3, and 4 together")

    # Should return an error message about only supporting 2 numbers
    assert "only supports" in result and "2 numbers" in result


async def test_tool_with_mocked_dependencies():
    """
    Example of how to test a tool that depends on other components.

    While the calculator tools don't have dependencies, this shows the pattern
    for tools that do (like tools that call LLMs or access memory).
    """
    from nat.test.tool_test_runner import with_mocked_dependencies

    # This pattern would be used for tools with dependencies:
    async with with_mocked_dependencies() as (runner, mock_builder):
        # Mock any dependencies the tool needs
        mock_builder.mock_llm("gpt-4", "Mocked LLM response")
        mock_builder.mock_memory_client("user_memory", {"key": "value"})

        # Test the tool with mocked dependencies
        result = await runner.test_tool_with_builder(
            config_type=MultiplyToolConfig,  # Using simple tool for demo
            builder=mock_builder,
            input_data="2 times 3")

        assert "6" in result
