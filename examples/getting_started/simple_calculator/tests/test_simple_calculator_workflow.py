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

import logging
from pathlib import Path

import pytest
import pytest_asyncio

from nat.builder.workflow import Workflow
from nat.runtime.loader import load_workflow
from nat.test.utils import locate_example_config
from nat_simple_calculator.register import CalculatorToolConfig

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture(scope="module")
async def workflow():
    config_file: Path = locate_example_config(CalculatorToolConfig)
    async with load_workflow(config_file) as workflow:
        yield workflow


async def run_calculator_tool(workflow: Workflow, workflow_input: str, expected_result: str):
    async with workflow.run(workflow_input) as runner:
        result = await runner.result(to_type=str)
    result = result.lower()
    assert expected_result in result


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_inequality_less_than_tool_workflow(workflow):
    await run_calculator_tool(workflow, "Is 8 less than 15?", "yes")
    await run_calculator_tool(workflow, "Is 15 less than 7?", "no")


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_inequality_greater_than_tool_workflow(workflow):
    await run_calculator_tool(workflow, "Is 15 greater than 8?", "yes")
    await run_calculator_tool(workflow, "Is 7 greater than 8?", "no")


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_inequality_equal_to_tool_workflow(workflow):
    await run_calculator_tool(workflow, "Is 8 plus 8 equal to 16?", "yes")
    await run_calculator_tool(workflow, "Is 8 plus 8 equal to 15?", "no")


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_add_tool_workflow(workflow):
    await run_calculator_tool(workflow, "What is 1+2?", "3")
    await run_calculator_tool(workflow, "What is 1+2+3?", "6")
    await run_calculator_tool(workflow, "What is 1+2+3+4+5?", "15")
    await run_calculator_tool(workflow, "What is 1+2+3+4+5+6+7+8+9+10?", "55")


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_subtract_tool_workflow(workflow):
    await run_calculator_tool(workflow, "What is 10-3?", "7")
    await run_calculator_tool(workflow, "What is 1-2?", "-1")


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_multiply_tool_workflow(workflow):
    await run_calculator_tool(workflow, "What is 2*3?", "6")
    await run_calculator_tool(workflow, "What is 2*3*4?", "24")
    await run_calculator_tool(workflow, "What is 2*3*4*5?", "120")
    await run_calculator_tool(workflow, "What is 2*3*4*5*6*7*8*9*10?", "3628800")
    await run_calculator_tool(workflow, "What is the product of -2 and 4?", "-8")


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_division_tool_workflow(workflow):
    await run_calculator_tool(workflow, "What is 12 divided by 2?", "6")
    await run_calculator_tool(workflow, "What is 12 divided by 3?", "4")
    await run_calculator_tool(workflow, "What is -12 divided by 2?", "-6")
    await run_calculator_tool(workflow, "What is 12 divided by -3?", "-4")
    await run_calculator_tool(workflow, "What is -12 divided by -3?", "4")
