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

import os
import typing
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

if typing.TYPE_CHECKING:
    from nat.data_models.config import Config


@pytest.fixture(name="simple_calculator_config_file", scope="module")
def simple_calculator_config_file_fixture() -> Path:
    cur_dir = Path(__file__).resolve().parent
    example_dir = cur_dir.parent
    config_file = example_dir / "configs/config-metadata.yml"
    assert config_file.exists(), f"Config file {config_file} does not exist"
    return config_file


@pytest.fixture(name="set_nat_config_file_env_var", autouse=True)
def fixture_set_nat_config_file_env_var(restore_environ, simple_calculator_config_file: Path) -> str:
    str_path = str(simple_calculator_config_file.absolute())
    os.environ["NAT_CONFIG_FILE"] = str_path
    return str_path


@asynccontextmanager
async def _build_client(config: "Config", worker_class: type | None = None):
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


@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.integration
async def test_full_workflow(simple_calculator_config_file: Path):
    from nat.runtime.loader import load_config
    config = load_config(simple_calculator_config_file)

    async with _build_client(config) as client:
        response = await client.post("/get_request_metadata",
                                     headers={
                                         "accept": "application/json",
                                         "Content-Type": "application/json",
                                         "Authorization": "Bearer token123"
                                     },
                                     json={"unused": "show me request details"})
        assert response.status_code == 200, f"Unexpected status code: {response.status_code}, {response.text}"
        result = response.json()
        assert "value" in result, f"Response payload missing expected `value` key: {result}"
        assert "/get_request_metadata" in result["value"], f"Response payload missing expected route: {result}"
