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

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.test.tool_test_runner import ToolTestRunner


class SimpleCalculatorToolConfig(FunctionBaseConfig, name="test_simple_calculator"):
    pass


@register_function(config_type=SimpleCalculatorToolConfig)
async def simple_calculator_tool(_config: SimpleCalculatorToolConfig, _builder: Builder):
    import re

    async def _calc_fn(input_data: str) -> str:
        """Simple calculator tool that adds two numbers."""
        match = re.findall(r"\d+", input_data)
        if match:
            nums = [int(num) for num in match]
            if len(nums) == 2:
                return f"The result of {nums[0]}+{nums[1]} is {nums[0]+nums[1]}"
        return "Invalid input"

    yield FunctionInfo.from_fn(_calc_fn, description=_calc_fn.__doc__)


# This test is to ensure ToolTestRunner is working correctly, and also a demonstration of how to test tools
# in complete isolation without requiring spinning up entire workflows, agents, and external services.
async def test_simple_calculator_tool():
    """Test simple calculator tool logic directly."""

    runner = ToolTestRunner()
    await runner.test_tool(config_type=SimpleCalculatorToolConfig,
                           input_data="2 + 3",
                           expected_output="The result of 2+3 is 5")


async def test_simple_calculator_tool_one_number():
    """Test with one number."""

    runner = ToolTestRunner()
    await runner.test_tool(config_type=SimpleCalculatorToolConfig, input_data="2", expected_output="Invalid input")


async def test_simple_calculator_tool_too_many_numbers():
    """Test too many numbers."""

    runner = ToolTestRunner()
    await runner.test_tool(config_type=SimpleCalculatorToolConfig,
                           input_data="2+2+2+2",
                           expected_output="Invalid input")


async def test_simple_calculator_tool_no_numbers():
    """Test with no numbers."""

    runner = ToolTestRunner()
    await runner.test_tool(config_type=SimpleCalculatorToolConfig, input_data="hello", expected_output="Invalid input")


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
            config_type=SimpleCalculatorToolConfig,  # Using simple tool for demo
            builder=mock_builder,
            input_data="2 + 3")

        assert "5" in result


async def test_tool_with_mocked_training_components():
    """
    Example of how to test a tool that depends on training components.

    This demonstrates the pattern for tools that use trainers, trainer adapters,
    or trajectory builders (like tools that perform fine-tuning operations).
    """
    from nat.test.tool_test_runner import with_mocked_dependencies

    async with with_mocked_dependencies() as (runner, mock_builder):
        # Mock training component dependencies
        mock_builder.mock_trainer("my_trainer", {"status": "training_complete"})
        mock_builder.mock_trainer_adapter("my_adapter", {"adapted": True})
        mock_builder.mock_trajectory_builder("my_trajectory_builder", {"trajectories": []})

        # Verify that mocked training components can be retrieved
        trainer = await mock_builder.get_trainer("my_trainer", None, None)
        assert trainer is not None
        assert await trainer.train() == {"status": "training_complete"}

        adapter = await mock_builder.get_trainer_adapter("my_adapter")
        assert adapter is not None
        assert await adapter.adapt() == {"adapted": True}

        trajectory_builder = await mock_builder.get_trajectory_builder("my_trajectory_builder")
        assert trajectory_builder is not None
        assert await trajectory_builder.build() == {"trajectories": []}
