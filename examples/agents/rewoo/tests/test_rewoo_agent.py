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
import os
import typing
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from nat.runtime.loader import load_workflow

if typing.TYPE_CHECKING:
    from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)


@pytest.fixture(name="config_file", scope="module")
def config_file_fixture() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "../configs", "config.yml")


@pytest_asyncio.fixture(name="workflow", scope="module")
async def workflow_fixture(config_file: str) -> AsyncGenerator["SessionManager"]:
    async with load_workflow(config_file) as workflow:
        yield workflow


@pytest.mark.usefixtures("nvidia_api_key", "tavily_api_key")
@pytest.mark.integration
@pytest.mark.parametrize(
    "question,answer",
    [("Which city held the Olympic game in the year represented by the bigger number of 1996 and 2004?", "athens"),
     ("Which U.S. historical event occurred in the year obtained by multiplying 48 and 37?",
      "declaration of independence"),
     ("Which country hosted the FIFA World Cup in the year obtained by dividing 6054 by 3?", "russia"),
     ("Which renowned physicist was born in the year resulting from subtracting 21 from 1900?", "albert einstein"),
     ("Which city hosted the Summer Olympics in the year obtained by subtracting 4 from the larger number"
      "between 2008 and 2012?",
      "beijing")])
async def test_full_workflow(question: str, answer: str, workflow: "SessionManager"):

    async with workflow.run(question) as runner:
        result = await runner.result(to_type=str)

    assert answer in result.lower()
