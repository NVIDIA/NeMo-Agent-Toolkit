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

import shutil
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from _utils.dask_utils import await_job
from nat.data_models.config import Config
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker


@pytest.fixture(name="test_config")
def test_config_fixture() -> Config:
    config = Config()
    config.general.front_end = FastApiFrontEndConfig(evaluate=FastApiFrontEndConfig.EndpointBase(
        path="/evaluate", method="POST", description="Test evaluate endpoint"))

    return config


@pytest_asyncio.fixture(autouse=True)
async def patch_evaluation_run(register_test_workflow):

    class MockEvaluationRun:
        """
        The MagicMock and AsyncMock classes are not serializable by Dask, so we create a simple mock class here.
        """

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self

        async def run_and_evaluate(self, *args, **kwargs):
            from nat.eval.config import EvaluationRunOutput
            from nat.eval.evaluator.evaluator_model import EvalInput
            from nat.profiler.data_models import ProfilerResults
            result = EvaluationRunOutput(workflow_output_file="/fake/output/path.json",
                                         evaluator_output_files=[],
                                         workflow_interrupted=False,
                                         eval_input=EvalInput(eval_input_items=[]),
                                         evaluation_results=[],
                                         usage_stats=None,
                                         profiler_results=ProfilerResults())

            return result

    with patch("nat.front_ends.fastapi.fastapi_front_end_plugin_worker.EvaluationRun",
               new_callable=MockEvaluationRun) as mock_eval_run:
        yield mock_eval_run


@pytest_asyncio.fixture(name="test_client")
async def test_client_fixture(test_config: Config) -> TestClient:
    worker = FastApiFrontEndPluginWorker(test_config)
    app = FastAPI()
    worker.set_cors_config(app)

    with patch("nat.front_ends.fastapi.fastapi_front_end_plugin_worker.SessionManager") as MockSessionManager:

        # Mock session manager
        mock_session = MagicMock()
        MockSessionManager.return_value = mock_session

        await worker.add_evaluate_route(app, session_manager=mock_session)

    return TestClient(app)


def create_job(test_client: TestClient, config_file: str, job_id: str | None = None):
    """Helper to create an evaluation job."""
    payload = {"config_file": config_file}
    if job_id:
        payload["job_id"] = job_id

    return test_client.post("/evaluate", json=payload)


@pytest.mark.asyncio
async def test_create_job(test_client: TestClient, eval_config_file: str):
    """Test creating a new evaluation job."""
    response = create_job(test_client, eval_config_file)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "submitted"
    await await_job(data["job_id"])


@pytest.mark.asyncio
async def test_get_job_status(test_client: TestClient, eval_config_file: str):
    """Test getting the status of a specific job."""
    create_response = create_job(test_client, eval_config_file)
    job_id = create_response.json()["job_id"]
    await await_job(job_id)

    status_response = test_client.get(f"/evaluate/job/{job_id}")
    assert status_response.status_code == 200
    data = status_response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "success"
    assert data["config_file"] == eval_config_file


def test_get_job_status_not_found(test_client: TestClient):
    """Test getting status of a non-existent job."""
    response = test_client.get("/evaluate/job/non-existent-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job non-existent-id not found"


@pytest.mark.asyncio
async def test_get_last_job(test_client: TestClient, eval_config_file: str):
    """Test getting the last created job."""
    for i in range(3):
        job_id = f"job-{i}"
        create_job(test_client, eval_config_file, job_id=job_id)
        await await_job(job_id)

    response = test_client.get("/evaluate/job/last")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "job-2"


def test_get_last_job_not_found(test_client: TestClient):
    """Test getting last job when no jobs exist."""
    response = test_client.get("/evaluate/job/last")
    assert response.status_code == 404
    assert response.json()["detail"] == "No jobs found"


@pytest.mark.parametrize("set_job_id", [False, True])
@pytest.mark.asyncio
async def test_get_all_jobs(test_client: TestClient, eval_config_file: str, set_job_id: bool):
    """Test retrieving all jobs."""
    for i in range(3):
        job_id = f"job-{i}" if set_job_id else None
        create_response = create_job(test_client, eval_config_file, job_id=job_id)
        job_id = create_response.json()["job_id"]
        await await_job(job_id)

    response = test_client.get("/evaluate/jobs")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3


@pytest.mark.parametrize("status,expected_count", [
    ("success", 3),
    ("interrupted", 0),
])
@pytest.mark.asyncio
async def test_get_jobs_by_status(test_client: TestClient, eval_config_file: str, status: str, expected_count: int):
    """Test getting jobs filtered by status."""
    for _ in range(3):
        response = create_job(test_client, eval_config_file)
        await await_job(response.json()["job_id"])

    response = test_client.get(f"/evaluate/jobs?status={status}")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == expected_count

    if status == "submitted":
        assert all(job["status"] == "submitted" for job in data)


@pytest.mark.asyncio
async def test_create_job_with_reps(test_client: TestClient, eval_config_file: str):
    """Test creating a new evaluation job with custom repetitions."""
    response = test_client.post("/evaluate", json={"config_file": eval_config_file, "reps": 3})
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "submitted"
    await await_job(data["job_id"])


@pytest.mark.asyncio
async def test_create_job_with_expiry(test_client: TestClient, eval_config_file: str):
    """Test creating a new evaluation job with custom expiry time."""
    response = test_client.post(
        "/evaluate",
        json={
            "config_file": eval_config_file,
            "expiry_seconds": 1800  # 30 minutes
        })
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "submitted"
    await await_job(data["job_id"])


@pytest.mark.asyncio
async def test_create_job_with_job_id(test_client: TestClient, eval_config_file: str):
    """Test creating a new evaluation job with a specific job ID."""
    job_id = "test-job-123"
    response = test_client.post("/evaluate", json={"config_file": eval_config_file, "job_id": job_id})
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "submitted"
    await await_job(job_id)


@pytest.mark.parametrize("job_id", ["test/job/123", "..", ".", "/abolute/path"
                                    "../relative", "/"])
def test_invalid_job_id(test_client: TestClient, eval_config_file: str, job_id: str):
    """Test creating a job with an invalid job ID."""
    response = test_client.post("/evaluate", json={"config_file": eval_config_file, "job_id": job_id})

    # We aren't concerned about the exact status code, but it should be in the 4xx range
    assert response.status_code >= 400 and response.status_code < 500


def test_invalid_config_file_doesnt_exist(test_client: TestClient):
    """Test creating a job with a config file that doesn't exist."""
    response = test_client.post("/evaluate", json={"config_file": "doesnt/exist/config.json"})
    # We aren't concerned about the exact status code, but it should be in the 4xx range
    assert response.status_code >= 400 and response.status_code < 500


@pytest.mark.asyncio
async def test_config_file_outside_curdir(test_client: TestClient, eval_config_file: str, tmp_path: Path):
    """Test creating a job with a config file outside the current directory."""
    dest_config_file = tmp_path / "config.yml"
    shutil.copy(eval_config_file, dest_config_file)
    assert dest_config_file.exists()

    response = test_client.post("/evaluate", json={"config_file": str(dest_config_file)})
    # We aren't concerned about the exact status code, but it should be in the 4xx range
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "submitted"
    await await_job(data["job_id"])
