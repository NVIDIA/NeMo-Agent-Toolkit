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

import typing
from pathlib import Path

import pytest
import pytest_asyncio

if typing.TYPE_CHECKING:
    from nat.builder.workflow import Workflow


@pytest_asyncio.fixture(name="workflow", scope="module")
async def workflow_fixture():
    from nat.runtime.loader import load_workflow
    from nat.test.utils import locate_example_config
    from nat_simple_calculator.register import CalculatorToolConfig

    config_file: Path = locate_example_config(CalculatorToolConfig)
    async with load_workflow(config_file) as workflow:
        yield workflow


async def run_calculator_tool(workflow: "Workflow", workflow_input: str, expected_result: str):
    async with workflow.run(workflow_input) as runner:
        result = await runner.result(to_type=str)
    result = result.lower()
    assert expected_result in result


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.parametrize("workflow_input, expected_result", [
    ("Is 8 less than 15?", "yes"),
    ("Is 15 less than 7?", "no"),
])
async def test_inequality_less_than_tool_workflow(workflow: "Workflow", workflow_input: str, expected_result: str):
    await run_calculator_tool(workflow, workflow_input, expected_result)


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.parametrize("workflow_input, expected_result", [("Is 15 greater than 8?", "yes"),
                                                             ("Is 7 greater than 8?", "no")])
async def test_inequality_greater_than_tool_workflow(workflow: "Workflow", workflow_input: str, expected_result: str):
    await run_calculator_tool(workflow, workflow_input, expected_result)


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.parametrize("workflow_input, expected_result", [("Is 8 plus 8 equal to 16?", "yes"),
                                                             ("Is 8 plus 8 equal to 15?", "no")])
async def test_inequality_equal_to_tool_workflow(workflow: "Workflow", workflow_input: str, expected_result: str):
    await run_calculator_tool(workflow, workflow_input, expected_result)


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.parametrize("workflow_input, expected_result",
                         [
                             ("What is 1+2?", "3"),
                             ("What is 1+2+3?", "6"),
                             ("What is 1+2+3+4+5?", "15"),
                             ("What is 1+2+3+4+5+6+7+8+9+10?", "55"),
                         ])
async def test_add_tool_workflow(workflow: "Workflow", workflow_input: str, expected_result: str):
    await run_calculator_tool(workflow, workflow_input, expected_result)


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.parametrize("workflow_input, expected_result", [
    ("What is 10-3?", "7"),
    ("What is 1-2?", "-1"),
])
async def test_subtract_tool_workflow(workflow: "Workflow", workflow_input: str, expected_result: str):
    await run_calculator_tool(workflow, workflow_input, expected_result)


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.parametrize("workflow_input, expected_result",
                         [
                             ("What is 2*3?", "6"),
                             ("What is 2*3*4?", "24"),
                             ("What is 2*3*4*5?", "120"),
                             ("What is 2*3*4*5*6*7*8*9*10?", "3628800"),
                             ("What is the product of -2 and 4?", "-8"),
                         ])
async def test_multiply_tool_workflow(workflow: "Workflow", workflow_input: str, expected_result: str):
    await run_calculator_tool(workflow, workflow_input, expected_result)


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.parametrize("workflow_input, expected_result",
                         [
                             ("What is 12 divided by 2?", "6"),
                             ("What is 12 divided by 3?", "4"),
                             ("What is -12 divided by 2?", "-6"),
                             ("What is 12 divided by -3?", "-4"),
                             ("What is -12 divided by -3?", "4"),
                         ])
async def test_division_tool_workflow(workflow: "Workflow", workflow_input: str, expected_result: str):
    await run_calculator_tool(workflow, workflow_input, expected_result)
