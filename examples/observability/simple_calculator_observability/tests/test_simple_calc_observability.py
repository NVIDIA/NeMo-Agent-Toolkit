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
import importlib
import json
import os
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
    import galileo.log_streams
    import galileo.projects
    import langsmith.client
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


@pytest.fixture(name="aiq_compatibility_span_prefix")
def aiq_compatibility_span_prefix_fixture():
    from nat.data_models import span

    orig_span_prefix = os.environ.get("NAT_SPAN_PREFIX")
    os.environ["NAT_SPAN_PREFIX"] = "aiq"
    importlib.reload(span)
    yield

    if orig_span_prefix is not None:
        os.environ["NAT_SPAN_PREFIX"] = orig_span_prefix
    else:
        del os.environ["NAT_SPAN_PREFIX"]

    importlib.reload(span)


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
                called_multiply = function_name == "calculator.multiply"

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

    runlist = []
    deadline = time.time() + 10
    while len(runlist) == 0 and time.time() < deadline:
        # Wait for traces to be ingested
        await asyncio.sleep(0.5)
        runs = langsmith_client.list_runs(project_name=langsmith_project_name, is_root=True)
        runlist = [run for run in runs]

    # Since we have a newly created project, the above workflow should have created exactly one root run
    assert len(runlist) == 1


@pytest.mark.integration
@pytest.mark.usefixtures("galileo_api_key")
async def test_galileo_full_workflow(config_dir: Path,
                                     galileo_project: "galileo.projects.Project",
                                     galileo_log_stream: "galileo.log_streams.LogStream",
                                     question: str,
                                     expected_answer: str):
    config_file = config_dir / "config-galileo.yml"
    config = load_config(config_file)
    config.general.telemetry.tracing["galileo"].project = galileo_project.name
    config.general.telemetry.tracing["galileo"].logstream = galileo_log_stream.name

    await run_workflow(config=config, question=question, expected_answer=expected_answer)

    import galileo.search

    sessions = []
    deadline = time.time() + 10
    while len(sessions) == 0 and time.time() < deadline:
        # Wait for traces to be ingested
        await asyncio.sleep(0.5)
        results = galileo.search.get_sessions(project_id=galileo_project.id, log_stream_id=galileo_log_stream.id)
        sessions = results.records or []

    assert len(sessions) == 1

    traces = galileo.search.get_traces(project_id=galileo_project.id, log_stream_id=galileo_log_stream.id)
    assert len(traces.records) == 1

    spans = galileo.search.get_spans(project_id=galileo_project.id, log_stream_id=galileo_log_stream.id)
    assert len(spans.records) > 1


@pytest.mark.integration
@pytest.mark.usefixtures("catalyst_keys", "aiq_compatibility_span_prefix")
async def test_catalyst_full_workflow(config_dir: Path,
                                      catalyst_project_name,
                                      catalyst_dataset_name,
                                      question: str,
                                      expected_answer: str):
    config_file = config_dir / "config-catalyst.yml"
    config = load_config(config_file)
    config.general.telemetry.tracing["catalyst"].project = catalyst_project_name
    config.general.telemetry.tracing["catalyst"].dataset = catalyst_dataset_name

    await run_workflow(config=config, question=question, expected_answer=expected_answer)

    from ragaai_catalyst import Dataset
    ds = Dataset(catalyst_project_name)

    dataset_found = False

    # Allow some time for the traces to be uploaded
    await asyncio.sleep(5)
    deadline = time.time() + 10
    while not dataset_found and time.time() < deadline:
        dataset_found = catalyst_dataset_name in ds.list_datasets()
        if not dataset_found:
            await asyncio.sleep(0.5)

    assert dataset_found
