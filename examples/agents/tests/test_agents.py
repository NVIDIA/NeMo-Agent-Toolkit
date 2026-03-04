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

import re
from pathlib import Path

import pytest

from nat.test.utils import run_workflow


def _extract_serve_response_text(response_json: dict) -> str:
    """Extract the answer text from a nat serve response payload.

    Handles both simple string responses and OpenAI-style chat completion responses.
    """
    response_value = response_json.get('value', {})
    if isinstance(response_value, str):
        return response_value
    combined = []
    for choice in response_value.get('choices', []):
        combined.append(choice.get('message', {}).get('content', ''))
    return "\n".join(combined)


def _assert_expected_answer(result: str, expected_answer: str) -> None:
    """Assert that the expected answer appears in the result, normalizing whitespace and case."""
    normalized = ' '.join(result.split())
    assert expected_answer.lower() in normalized.lower(), f"Expected '{expected_answer}' in '{result}'"


# ---------------------------------------------------------------------------
# ReWOO agent tests -- one workflow build shared across all 5 questions
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "tavily_api_key")
class TestReWOONatRun:

    @pytest.mark.parametrize("qa_idx", range(5), ids=[f"qa_{i+1}" for i in range(5)])
    async def test_question(self, rewoo_session_manager, rewoo_data: list[dict], qa_idx: int):
        qa = rewoo_data[qa_idx]
        async with rewoo_session_manager.session() as session:
            async with session.run(qa["question"]) as runner:
                result = await runner.result(to_type=str)
                _assert_expected_answer(result, qa["answer"])


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "tavily_api_key")
class TestReWOONatServe:

    @pytest.mark.parametrize("qa_idx", range(5), ids=[f"qa_{i+1}" for i in range(5)])
    async def test_question(self, rewoo_nat_client, rewoo_data: list[dict], qa_idx: int):
        qa = rewoo_data[qa_idx]
        resp = await rewoo_nat_client.post("/generate",
                                           json={"messages": [{
                                               "role": "user", "content": qa["question"]
                                           }]})
        resp.raise_for_status()
        response_text = _extract_serve_response_text(resp.json())
        _assert_expected_answer(response_text, qa["answer"])


# ---------------------------------------------------------------------------
# Tool Calling responses API agent test
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.usefixtures("openai_api_key")
async def test_tool_calling_responses_api(agents_dir: Path, question: str, answer: str):
    await run_workflow(config_file=agents_dir / "tool_calling/configs/config-responses-api.yml",
                       question=question,
                       expected_answer=answer)


@pytest.mark.integration
@pytest.mark.usefixtures("openai_api_key")
async def test_nat_run_tool_calling_responses_api(tool_calling_responses_api_nat_client, question: str, answer: str):
    resp = await tool_calling_responses_api_nat_client.post("/generate", json={"input_message": question})
    resp.raise_for_status()
    response_text = _extract_serve_response_text(resp.json())
    _assert_expected_answer(response_text, answer)


# ---------------------------------------------------------------------------
# Other agent tests -- fixture parametrized by config (class-scoped)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
class TestAgentNatRun:

    async def test_question(self, agent_session_manager, question: str, answer: str):
        async with agent_session_manager.session() as session:
            async with session.run(question) as runner:
                result = await runner.result(to_type=str)
                _assert_expected_answer(result, answer)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
class TestAgentNatServe:

    async def test_question(self, agent_nat_client, question: str, answer: str):
        resp = await agent_nat_client.post("/generate", json={"messages": [{"role": "user", "content": question}]})
        resp.raise_for_status()
        response_text = _extract_serve_response_text(resp.json())
        _assert_expected_answer(response_text, answer)


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
    assert re.search(r"large language model", result, re.IGNORECASE) is not None
