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
import random
import time
import types
import typing
from collections.abc import Generator
from pathlib import Path

import pytest

from nat.runtime.loader import load_config
from nat.test.utils import run_workflow

if typing.TYPE_CHECKING:
    from weave.trace.weave_client import WeaveClient


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


@pytest.fixture(name="weave_attribute_key")
def weave_attribute_key_fixture() -> str:
    # Create a unique identifier for this test run, and use it as an attribute on all traces
    return "test_run"


@pytest.fixture(name="weave_identifier")
def weave_identifier_fixture() -> str:
    # Create a unique identifier for this test run, and use it as an attribute on all traces
    return f'test_run_{time.time()}_{random.random()}'


@pytest.fixture(name="weave_project_name")
def fixture_weave_project_name() -> str:
    return "weave_test_e2e"


@pytest.fixture(name="weave_query")
def fixture_weave_query(weave_attribute_key: str, weave_identifier: str) -> dict:
    return {"$expr": {"$eq": [{"$getField": f"attributes.{weave_attribute_key}"}, {"$literal": weave_identifier}]}}


@pytest.fixture(name="weave_client")
def fixture_weave_client(weave: types.ModuleType, weave_project_name: str, wandb_api_key: str,
                         weave_query: dict) -> "Generator[WeaveClient]":
    client = weave.init(weave_project_name)
    yield client

    client.flush()
    calls = client.get_calls(query=weave_query)
    call_ids = [c.id for c in calls]
    if len(call_ids) > 0:
        client.delete_calls(call_ids)


@pytest.mark.integration
@pytest.mark.usefixtures("wandb_api_key")
async def test_weave_full_workflow(config_dir: Path,
                                   weave_project_name: str,
                                   weave_attribute_key: str,
                                   weave_identifier: str,
                                   weave_client: "WeaveClient",
                                   weave_query: dict,
                                   question: str,
                                   expected_answer: str):
    config_file = config_dir / "config-weave.yml"
    config = load_config(config_file)
    config.general.telemetry.tracing["weave"].project = weave_project_name
    config.general.telemetry.tracing["weave"].attributes = {weave_attribute_key: weave_identifier, "other_attr": 123}

    await run_workflow(config=config, question=question, expected_answer=expected_answer)

    weave_client.flush()
    calls = weave_client.get_calls(query=weave_query)
    assert len(calls) > 0
    for call in calls:
        assert call.attributes is not None
        assert call.attributes.get("other_attr") == 123


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
