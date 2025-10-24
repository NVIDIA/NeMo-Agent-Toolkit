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

import asyncio
import json
import time
import types
import typing
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from nat.runtime.loader import load_config
from nat.test.utils import run_workflow

if typing.TYPE_CHECKING:
    import langsmith.client


@pytest.fixture(name="config_dir", scope="session")
def config_dir_fixture(examples_dir: Path) -> Path:
    return examples_dir / "observability/simple_calculator_observability/configs"


@pytest.fixture(name="nvidia_api_key", autouse=True, scope='module')
def nvidia_api_key_fixture(nvidia_api_key):
    return nvidia_api_key


@pytest.fixture(name="question", scope="module")
def question_fixture() -> str:
    return "What is 2 * 4?"


@pytest.fixture(name="expected_answer", scope="module")
def expected_answer_fixture() -> str:
    return "8"


@pytest.fixture(name="weave_project_name", scope="module")
async def fixture_weave_project_name(weave: types.ModuleType, wandb_api_key) -> AsyncGenerator[str]:
    # This currently has the following problems:
    # 1. Ideally we would create a new project for each test run to avoid conflicts, and then delete the project.
    #    However, W&B does not currently support deleting projects via the API.
    # 2. We don't have a way (that I know of) to identifiy traces from this specific test run, such that we only delete
    #    those traces.
    project_name = "weave_test_e2e"
    client = weave.init(project_name)
    yield project_name

    client.finish(use_progress_bar=False)
    call_ids = [c.id for c in client.get_calls()]
    if len(call_ids) > 0:
        client.delete_calls(call_ids)


@pytest.mark.integration
@pytest.mark.usefixtures("wandb_api_key")
async def test_weave_full_workflow(config_dir: Path, weave_project_name: str, question: str, expected_answer: str):
    config_file = config_dir / "config-weave.yml"
    config = load_config(config_file)
    config.general.telemetry.tracing["weave"].project = weave_project_name

    await run_workflow(config=config, question=question, expected_answer=expected_answer)


@pytest.mark.integration
async def test_phoenix_full_workflow(config_dir: Path, phoenix_trace_url: str, question: str, expected_answer: str):
    config_file = config_dir / "config-phoenix.yml"
    config = load_config(config_file)
    config.general.telemetry.tracing["phoenix"].endpoint = phoenix_trace_url

    await run_workflow(config=config, question=question, expected_answer=expected_answer)


@pytest.mark.integration
async def test_otel_full_workflow(tmp_path: Path, config_dir: Path, question: str, expected_answer: str):
    otel_file = tmp_path / "otel-trace.jsonl"

    config_file = config_dir / "config-otel-file.yml"
    config = load_config(config_file)
    config.general.telemetry.tracing["otel_file"].output_path = str(otel_file.absolute())

    await run_workflow(config=config, question=question, expected_answer=expected_answer)

    assert otel_file.exists()

    traces = []
    called_multiply = False
    with open(otel_file, encoding="utf-8") as fh:
        for line in fh:
            trace = json.loads(line)
            traces.append(trace)

            if not called_multiply:
                function_name = trace.get('function_ancestry', {}).get('function_name')
                called_multiply = function_name == "calculator_multiply"

    assert len(traces) > 0
    assert called_multiply


@pytest.mark.integration
async def test_langfuse_full_workflow(config_dir: Path, langfuse_trace_url: str, question: str, expected_answer: str):
    config_file = config_dir / "config-langfuse.yml"
    config = load_config(config_file)
    config.general.telemetry.tracing["langfuse"].endpoint = langfuse_trace_url

    await run_workflow(config=config, question=question, expected_answer=expected_answer)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("langsmith_api_key")
async def test_langsmith_full_workflow(config_dir: Path,
                                       langsmith_client: "langsmith.client.Client",
                                       langsmith_project_name: str,
                                       question: str,
                                       expected_answer: str):
    config_file = config_dir / "config-langsmith.yml"
    config = load_config(config_file)
    config.general.telemetry.tracing["langsmith"].project = langsmith_project_name

    await run_workflow(config=config, question=question, expected_answer=expected_answer)

    done = False
    runlist = []
    deadline = time.time() + 10
    while not done and time.time() < deadline:
        # Wait for traces to be ingested
        await asyncio.sleep(0.5)
        runs = langsmith_client.list_runs(project_name=langsmith_project_name, is_root=True)
        runlist = [run for run in runs]
        if len(runlist) > 0:
            done = True

    assert done, "Timed out waiting for LangSmith run to be ingested"
    # Since we have a newly created project, the above workflow should have created exactly one root run
    assert len(runlist) == 1
