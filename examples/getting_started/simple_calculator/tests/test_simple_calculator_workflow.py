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

from nat.runtime.loader import load_workflow
from nat.test.utils import locate_example_config
from nat_simple_calculator.register import DivisionToolConfig
from nat_simple_calculator.register import InequalityToolConfig
from nat_simple_calculator.register import MultiplyToolConfig

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_inequality_tool_workflow():

    config_file: Path = locate_example_config(InequalityToolConfig)

    async with load_workflow(config_file) as workflow:

        async with workflow.run("Is 8 greater than 15?") as runner:

            result = await runner.result(to_type=str)

        result = result.lower()
        assert "no" in result


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_multiply_tool_workflow():

    config_file: Path = locate_example_config(MultiplyToolConfig)

    async with load_workflow(config_file) as workflow:

        async with workflow.run("What is the product of 2 * 4?") as runner:

            result = await runner.result(to_type=str)

        result = result.lower()
        assert "8" in result


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_division_tool_workflow():

    config_file: Path = locate_example_config(DivisionToolConfig)

    async with load_workflow(config_file) as workflow:

        async with workflow.run("What is 12 divided by 2?") as runner:

            result = await runner.result(to_type=str)

        result = result.lower()
        assert "6" in result
