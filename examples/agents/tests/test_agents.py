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

import json
import logging
import os

import pytest

from nat.runtime.loader import load_workflow

logger = logging.getLogger(__name__)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.dirname(CUR_DIR)


@pytest.fixture(name="question", scope="module")
def question_fixture() -> str:
    return "What are LLMs"


@pytest.fixture(name="answer", scope="module")
def answer_fixture() -> str:
    return "large language model"


@pytest.fixture(name="rewoo_data", scope="module")
def rewoo_data_fixture() -> list[dict]:
    data_path = os.path.join(AGENTS_DIR, "data/rewoo.json")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(name="rewoo_question")
def rewoo_question_fixture(request: pytest.FixtureRequest, rewoo_data: list[dict]) -> str:
    return rewoo_data[request.param]["question"]


@pytest.fixture(name="rewoo_answer")
def rewoo_answer_fixture(request: pytest.FixtureRequest, rewoo_data: list[dict]) -> str:
    return rewoo_data[request.param]["answer"].lower()


async def _test_full_workflow(config_file: str, question: str, expected_answer: str):
    async with load_workflow(config_file) as workflow:
        async with workflow.run(question) as runner:
            result = await runner.result(to_type=str)

    assert expected_answer in result.lower(), f"Expected '{expected_answer}' in '{result}'"


@pytest.mark.usefixtures("nvidia_api_key", "tavily_api_key")
@pytest.mark.integration
@pytest.mark.parametrize("rewoo_question, rewoo_answer", [(i, i) for i in range(4)],
                         ids=[f"qa_{i+1}" for i in range(4)],
                         indirect=True)
async def test_rewoo_full_workflow(rewoo_question: str, rewoo_answer: str):
    config_file = os.path.join(AGENTS_DIR, "rewoo/configs/config.yml")
    await _test_full_workflow(config_file, rewoo_question, rewoo_answer)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.parametrize(
    "config_file",
    [
        os.path.join(AGENTS_DIR, "mixture_of_agents/configs/config.yml"),
        os.path.join(AGENTS_DIR, "react/configs/config.yml"),
        os.path.join(AGENTS_DIR, "react/configs/config-reasoning.yml"),
        os.path.join(AGENTS_DIR, "tool_calling/configs/config.yml"),
        os.path.join(AGENTS_DIR, "tool_calling/configs/config-reasoning.yml"),
    ],
    ids=["mixture_of_agents", "react", "react-reasoning", "tool_calling", "tool_calling-reasoning"])
async def test_agent_full_workflow(config_file: str, question: str, answer: str):
    await _test_full_workflow(config_file, question, answer)
