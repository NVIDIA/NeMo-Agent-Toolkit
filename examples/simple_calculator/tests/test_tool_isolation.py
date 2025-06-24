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
"""
Demonstration of isolated tool testing using the new ToolTestRunner.

This shows how tools can be tested in isolation without requiring:
- Full workflow setup
- LLM connections
- Complex configuration files
- All dependencies loaded

This addresses AIQ-940: "No clear way to test tools in isolation"
"""

import pytest
from aiq_simple_calculator.register import DivisionToolConfig
from aiq_simple_calculator.register import InequalityToolConfig
from aiq_simple_calculator.register import MultiplyToolConfig
from aiq_simple_calculator.register import SubtractToolConfig

from aiq.test.tool_test_runner import test_tool


class TestCalculatorToolsIsolation:
    """
    Test calculator tools in complete isolation from workflows, agents, and LLMs.

    These tests are:
    - Fast: No workflow loading, no LLM initialization
    - Simple: No config files needed
    - Focused: Test only the tool logic
    - Reliable: No external dependencies
    """

    @pytest.mark.asyncio
    async def test_inequality_tool_direct(self):
        """Test inequality tool logic directly."""

        result = await test_tool(config_type=InequalityToolConfig,
                                 input_data="Is 8 greater than 15?",
                                 expected_output="First number 8 is less than the second number 15")

        assert "8" in result
        assert "15" in result
        assert "less" in result

    @pytest.mark.asyncio
    async def test_inequality_tool_equal_case(self):
        """Test inequality tool with equal numbers."""

        result = await test_tool(config_type=InequalityToolConfig,
                                 input_data="Compare 5 and 5",
                                 expected_output="First number 5 is equal to the second number 5")

        assert "equal" in result

    @pytest.mark.asyncio
    async def test_inequality_tool_greater_case(self):
        """Test inequality tool with first number greater."""

        result = await test_tool(config_type=InequalityToolConfig,
                                 input_data="Is 15 greater than 8?",
                                 expected_output="First number 15 is greater than the second number 8")

        assert "greater" in result

    @pytest.mark.asyncio
    async def test_multiply_tool_direct(self):
        """Test multiply tool logic directly."""

        result = await test_tool(config_type=MultiplyToolConfig,
                                 input_data="What is 2 times 4?",
                                 expected_output="The product of 2 * 4 is 8")

        assert "8" in result
        assert "product" in result

    @pytest.mark.asyncio
    async def test_multiply_tool_edge_cases(self):
        """Test multiply tool with various inputs."""

        # Test with zero
        result = await test_tool(config_type=MultiplyToolConfig,
                                 input_data="Multiply 0 and 5",
                                 expected_output="The product of 0 * 5 is 0")
        assert "0" in result

        # Test with larger numbers
        result = await test_tool(config_type=MultiplyToolConfig,
                                 input_data="Calculate 12 times 13",
                                 expected_output="The product of 12 * 13 is 156")
        assert "156" in result

    @pytest.mark.asyncio
    async def test_division_tool_direct(self):
        """Test division tool logic directly."""

        result = await test_tool(config_type=DivisionToolConfig,
                                 input_data="What is 8 divided by 2?",
                                 expected_output="The result of 8 / 2 is 4.0")

        assert "4.0" in result
        assert "result" in result

    @pytest.mark.asyncio
    async def test_division_tool_with_remainder(self):
        """Test division with decimal result."""

        result = await test_tool(config_type=DivisionToolConfig,
                                 input_data="Divide 7 by 2",
                                 expected_output="The result of 7 / 2 is 3.5")

        assert "3.5" in result

    @pytest.mark.asyncio
    async def test_subtract_tool_direct(self):
        """Test subtract tool logic directly."""

        result = await test_tool(config_type=SubtractToolConfig,
                                 input_data="What is 10 minus 3?",
                                 expected_output="The result of 10 - 3 is 7")

        assert "7" in result
        assert "result" in result

    @pytest.mark.asyncio
    async def test_subtract_tool_negative_result(self):
        """Test subtract tool with negative result."""

        result = await test_tool(config_type=SubtractToolConfig,
                                 input_data="Subtract 15 from 10",
                                 expected_output="The result of 15 - 10 is 5")

        assert "5" in result

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test error handling for insufficient numbers."""

        result = await test_tool(config_type=MultiplyToolConfig, input_data="Multiply just one number: 5")

        # Should return an error message
        assert "Provide at least 2 numbers" in result

    @pytest.mark.asyncio
    async def test_tool_validation_too_many_numbers(self):
        """Test validation for too many numbers."""

        result = await test_tool(config_type=MultiplyToolConfig, input_data="Multiply 2, 3, and 4 together")

        # Should return an error message about only supporting 2 numbers
        assert "only supports" in result and "2 numbers" in result


# Performance comparison test
@pytest.mark.performance
class TestPerformanceComparison:
    """
    Demonstrate the performance benefits of isolated testing vs full workflow testing.
    """

    @pytest.mark.asyncio
    async def test_isolated_performance(self):
        """Test multiple tools quickly in isolation."""
        import time

        start_time = time.time()

        # Test multiple tools rapidly
        await test_tool(InequalityToolConfig, input_data="5 vs 3")
        await test_tool(MultiplyToolConfig, input_data="5 times 3")
        await test_tool(DivisionToolConfig, input_data="15 divided by 3")
        await test_tool(SubtractToolConfig, input_data="10 minus 7")

        end_time = time.time()

        # Isolated testing should be very fast (under 1 second for 4 tools)
        elapsed = end_time - start_time
        print(f"Isolated testing took {elapsed:.3f} seconds for 4 tools")
        assert elapsed < 1.0, f"Isolated testing should be fast, took {elapsed:.3f}s"


# Demo of advanced usage with mocked dependencies
class TestAdvancedToolIsolation:
    """
    Demonstrate testing tools that have dependencies on other components.

    This would be useful for tools that call LLMs, access memory, etc.
    """

    @pytest.mark.asyncio
    async def test_tool_with_mocked_dependencies(self):
        """
        Example of how to test a tool that depends on other components.

        While the calculator tools don't have dependencies, this shows the pattern
        for tools that do (like tools that call LLMs or access memory).
        """
        from aiq.test.tool_test_runner import with_mocked_dependencies

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


if __name__ == "__main__":
    """
    Run this file directly to see the tool isolation testing in action:

    python -m pytest examples/simple_calculator/tests/test_tool_isolation.py -v
    """
    pytest.main([__file__, "-v"])
