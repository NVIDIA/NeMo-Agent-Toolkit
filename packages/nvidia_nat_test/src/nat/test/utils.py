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
import importlib.resources
import inspect
import json
import subprocess
import time
import typing
from contextlib import asynccontextmanager
from pathlib import Path

if typing.TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from httpx import AsyncClient

    from nat.data_models.config import Config
    from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
    from nat.utils.type_utils import StrPath


def locate_repo_root() -> Path:
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=False, capture_output=True, text=True)
    assert result.returncode == 0, f"Failed to get git root: {result.stderr}"
    return Path(result.stdout.strip())


def locate_example_src_dir(example_config_class: type) -> Path:
    """
    Locate the example src directory for an example's config class.
    """
    package_name = inspect.getmodule(example_config_class).__package__
    return importlib.resources.files(package_name)


def locate_example_dir(example_config_class: type) -> Path:
    """
    Locate the example directory for an example's config class.
    """
    src_dir = locate_example_src_dir(example_config_class)
    example_dir = src_dir.parent.parent
    return example_dir


def locate_example_config(example_config_class: type,
                          config_file: str = "config.yml",
                          assert_exists: bool = True) -> Path:
    """
    Locate the example config file for an example's config class, assumes the example contains a 'configs' directory
    """
    example_dir = locate_example_src_dir(example_config_class)
    config_path = example_dir.joinpath("configs", config_file).absolute()
    if assert_exists:
        assert config_path.exists(), f"Config file {config_path} does not exist"

    return config_path


async def run_workflow(*,
                       config: "Config | None" = None,
                       config_file: "StrPath | None" = None,
                       question: str,
                       expected_answer: str,
                       assert_expected_answer: bool = True,
                       **kwargs) -> str:
    """
    Test specific wrapper for `nat.utils.run_workflow` to run a workflow with a question and validate the expected
    answer. This variant always sets the result type to `str`.
    """
    from nat.utils import run_workflow as nat_run_workflow

    result = await nat_run_workflow(config=config, config_file=config_file, prompt=question, to_type=str, **kwargs)

    if assert_expected_answer:
        assert expected_answer.lower() in result.lower(), f"Expected '{expected_answer}' in '{result}'"

    return result


async def serve_workflow(*,
                         config_path: Path,
                         question: str,
                         expected_answer: str,
                         assert_expected_answer: bool = True,
                         port: int = 8000,
                         pipeline_timeout: int = 60,
                         request_timeout: int = 30) -> dict:
    """
    Execute a workflow using `nat serve`, and issue a POST request to the `/generate` endpoint with the given question.

    Intended to be analogous to `run_workflow` but for the REST API serving mode.
    """
    import requests
    workflow_url = f"http://localhost:{port}"
    workflow_cmd = ["nat", "serve", "--port", str(port), "--config_file", str(config_path.absolute())]
    proc = subprocess.Popen(workflow_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.poll() is None, f"NAT server process failed to start: {proc.stdout.read()}"

    response_payload = {}
    try:
        deadline = time.time() + pipeline_timeout  # timeout waiting for the workflow to respond
        response = None
        while response is None and time.time() < deadline:
            try:
                response = requests.post(url=f"{workflow_url}/generate",
                                         json={"messages": [{
                                             "role": "user", "content": question
                                         }]},
                                         timeout=request_timeout)
            except Exception:
                await asyncio.sleep(0.1)

        assert response is not None, f"deadline exceeded waiting for workflow response: {proc.stdout.read()}"
        response.raise_for_status()
        response_payload = response.json()
        combined_response = []
        response_value = response_payload.get('value', {})
        if isinstance(response_value, str):
            response_text = response_value
        else:
            for choice in response_value.get('choices', []):
                combined_response.append(choice.get('message', {}).get('content', ''))

            response_text = "\n".join(combined_response)

        if assert_expected_answer:
            assert expected_answer.lower() in response_text.lower(), \
                f"Unexpected response: {response.text}"
    finally:
        # Teardown
        i = 0
        while proc.poll() is None and i < 5:
            if i == 0:
                proc.terminate()
            else:
                proc.kill()
            await asyncio.sleep(0.1)
            i += 1

        assert proc.poll() is not None, "NAT server process failed to terminate"

    return response_payload


@asynccontextmanager
async def build_nat_client(
        config: "Config",
        worker_class: "type[FastApiFrontEndPluginWorker] | None" = None) -> "AsyncIterator[AsyncClient]":
    """
    Build a NAT client for testing purposes.

    Creates a test client with an ASGI transport for the specified configuration.
    The client is backed by a FastAPI application built from the provided worker class.

    Args:
        config: The NAT configuration to use for building the client.
        worker_class: Optional worker class to use. Defaults to FastApiFrontEndPluginWorker.

    Yields:
        An AsyncClient instance configured for testing.
    """
    from asgi_lifespan import LifespanManager
    from httpx import ASGITransport
    from httpx import AsyncClient

    from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker

    if worker_class is None:
        worker_class = FastApiFrontEndPluginWorker

    worker = worker_class(config)
    app = worker.build_app()

    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client


def validate_workflow_output(workflow_output_file: Path) -> None:
    """
    Validate the contents of the workflow output file.
    WIP: output format should be published as a schema and this validation should be done against that schema.
    """
    # Ensure the workflow_output.json file was created
    assert workflow_output_file.exists(), "The workflow_output.json file was not created"

    # Read and validate the workflow_output.json file
    try:
        with open(workflow_output_file, encoding="utf-8") as f:
            result_json = json.load(f)
    except json.JSONDecodeError as err:
        raise RuntimeError("Failed to parse workflow_output.json as valid JSON") from err

    assert isinstance(result_json, list), "The workflow_output.json file is not a list"
    assert len(result_json) > 0, "The workflow_output.json file is empty"
    assert isinstance(result_json[0], dict), "The workflow_output.json file is not a list of dictionaries"

    # Ensure required keys exist
    required_keys = ["id", "question", "answer", "generated_answer", "intermediate_steps"]
    for key in required_keys:
        assert all(item.get(key) for item in result_json), f"The '{key}' key is missing in workflow_output.json"
