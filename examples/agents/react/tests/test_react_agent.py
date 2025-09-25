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

import pytest

from nat.runtime.loader import load_workflow

logger = logging.getLogger(__name__)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(CUR_DIR, "../configs")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.parametrize("question, answer",
                         [
                             ("What are LLMs", "large language models"),
                             ("who was Djikstra?", "Dutch computer scientist"),
                             ("what is the goldilocks zone?", "temperatures are just right"),
                         ],
                         ids=["llms", "dijkstra", "goldilocks"])
@pytest.mark.parametrize("config_file",
                         [os.path.join(CFG_DIR, "config.yml"), os.path.join(CFG_DIR, "config-reasoning.yml")],
                         ids=["standard", "reasoning"])
async def test_full_workflow(config_file: str, question: str, answer: str):
    async with load_workflow(config_file) as workflow:
        async with workflow.run(question) as runner:
            result = await runner.result(to_type=str)

    assert answer.lower() in result.lower(), f"Expected '{answer}' in '{result}'"
