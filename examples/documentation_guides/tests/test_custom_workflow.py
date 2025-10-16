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

from nat.test.utils import run_workflow

logger = logging.getLogger(__name__)


@pytest.fixture(name="custom_workflow_dir", scope="session")
def custom_workflow_dir_fixture(workflows_dir: Path) -> Path:
    return workflows_dir / "custom_workflow"


@pytest.fixture(name="question", scope="module")
def question_fixture() -> str:
    return "How do I trace only specific parts of my LangChain application?"


@pytest.fixture(name="answer", scope="module")
def answer_fixture() -> str:
    # Since the results are not deterministic, we just check for anything looking remotely like a correct answer
    return "trace"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_custom_full_workflow(custom_workflow_dir: Path, question: str, answer: str):
    config_file = custom_workflow_dir / "custom_config.yml"
    await run_workflow(config_file=config_file, question=question, expected_answer=answer)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "tavily_api_key")
async def test_search_full_workflow(custom_workflow_dir: Path, question: str, answer: str):
    # Technically this is the same as the custom workflow test, but it requires a second key
    config_file = custom_workflow_dir / "search_config.yml"
    await run_workflow(config_file=config_file, question=question, expected_answer=answer)
