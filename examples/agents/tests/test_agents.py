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
import re
from pathlib import Path

import pytest

from nat.test.utils import run_workflow


@pytest.fixture(name="agents_dir", scope="session")
def agents_dir_fixture(examples_dir: Path) -> Path:
    return examples_dir / "agents"


@pytest.fixture(name="question", scope="module")
def question_fixture() -> str:
    return "What are LLMs"


@pytest.fixture(name="answer", scope="module")
def answer_fixture() -> str:
    return "large language model"


@pytest.fixture(name="rewoo_data", scope="module")
def rewoo_data_fixture(agents_dir: Path) -> list[dict]:
    data_path = agents_dir / "data/rewoo.json"
    assert data_path.exists(), f"Data file {data_path} does not exist"
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(name="rewoo_question")
def rewoo_question_fixture(request: pytest.FixtureRequest, rewoo_data: list[dict]) -> str:
    return rewoo_data[request.param]["question"]


@pytest.fixture(name="rewoo_answer")
def rewoo_answer_fixture(request: pytest.FixtureRequest, rewoo_data: list[dict]) -> str:
    return rewoo_data[request.param]["answer"].lower()


@pytest.mark.usefixtures("nvidia_api_key", "tavily_api_key")
@pytest.mark.integration
@pytest.mark.parametrize("rewoo_question, rewoo_answer", [(i, i) for i in range(5)],
                         ids=[f"qa_{i+1}" for i in range(5)],
                         indirect=True)
async def test_rewoo_full_workflow(agents_dir: Path, rewoo_question: str, rewoo_answer: str):
    config_file = agents_dir / "rewoo/configs/config.yml"
    await run_workflow(config_file=config_file, question=rewoo_question, expected_answer=rewoo_answer)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.parametrize(
    "config_file",
    [
        # These are all expected to be relative to the agents_dir
        "mixture_of_agents/configs/config.yml",
        "react/configs/config.yml",
        "react/configs/config-reasoning.yml",
        "tool_calling/configs/config.yml",
        "tool_calling/configs/config-reasoning.yml",
    ],
    ids=["mixture_of_agents", "react", "react-reasoning", "tool_calling", "tool_calling-reasoning"])
async def test_agent_full_workflow(agents_dir: Path, config_file: str, question: str, answer: str):
    await run_workflow(config_file=agents_dir / config_file, question=question, expected_answer=answer)


# Code examples from `docs/source/resources/running-tests.md`
# Intentionally not using the fixtures defined above to keep the examples clear
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_react_agent_full_workflow(examples_dir: Path):
    config_file = examples_dir / "agents/react/configs/config.yml"
    await run_workflow(config_file=config_file, question="What are LLMs?", expected_answer="Large Language Model")


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_react_agent_full_workflow_validate_re(examples_dir: Path):
    config_file = examples_dir / "agents/react/configs/config.yml"
    result = await run_workflow(config_file=config_file,
                                question="What are LLMs?",
                                expected_answer="",
                                assert_expected_answer=False)
    assert re.match(r".*large language model.*", result, re.IGNORECASE) is not None
