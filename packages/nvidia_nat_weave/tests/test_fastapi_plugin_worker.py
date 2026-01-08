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

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.plugins.weave.fastapi_plugin_worker import WeaveFastAPIPluginWorker
from nat.plugins.weave.register import WeaveTelemetryExporter
from nat.test.functions import EchoFunctionConfig
from nat.test.utils import build_nat_client


@pytest.fixture(name="setup_env", autouse=True)
def fixture_setup_env() -> None:
    """Set up environment variables for tests."""
    # Set a dummy config file path for tests that don't use Dask
    if "NAT_CONFIG_FILE" not in os.environ:
        temp_dir = tempfile.gettempdir()
        os.environ["NAT_CONFIG_FILE"] = os.path.join(temp_dir, "dummy_nat_config.yml")
    yield


@pytest.fixture(name="mock_weave", autouse=True)
def fixture_mock_weave(monkeypatch):
    """Mock weave.init and weave client context to avoid authentication issues in unit tests."""
    mock_weave_client = MagicMock()

    # Mock weave.init
    monkeypatch.setattr("weave.init", lambda *args, **kwargs: mock_weave_client, raising=False)

    # Mock the weave client context to return the mock client
    monkeypatch.setattr("weave.trace.context.weave_client_context.require_weave_client", lambda: mock_weave_client,
                        raising=False)
    monkeypatch.setattr("weave.trace.context.weave_client_context.get_weave_client", lambda: mock_weave_client,
                        raising=False)

    yield mock_weave_client


async def test_weave_feedback_endpoint_with_weave_configured() -> None:
    """Test that the feedback endpoint is registered when Weave telemetry is configured."""

    config = Config(
        general=GeneralConfig(front_end=FastApiFrontEndConfig(),
                              telemetry={"tracing": {
                                  "weave": WeaveTelemetryExporter(project="test-project")
                              }}),
        workflow=EchoFunctionConfig(),
    )

    async with build_nat_client(config, worker_class=WeaveFastAPIPluginWorker) as client:
        # Test that the feedback endpoint exists
        response = await client.post("/feedback",
                                     json={
                                         "observability_trace_id": "test-trace-id", "reaction_type": "👍"
                                     })

        # The endpoint should exist (not 404) even if it fails with 500 due to missing Weave setup
        # In a real scenario with Weave properly initialized, this would return 200
        assert response.status_code in [200, 500], \
            f"Expected 200 or 500, got {response.status_code}"


async def test_feedback_endpoint_not_registered_without_weave() -> None:
    """Test that the feedback endpoint is not registered when Weave telemetry is not configured."""

    config = Config(
        general=GeneralConfig(front_end=FastApiFrontEndConfig(), ),
        workflow=EchoFunctionConfig(),
    )

    async with build_nat_client(config, worker_class=WeaveFastAPIPluginWorker) as client:
        # Test that the feedback endpoint does not exist
        response = await client.post("/feedback",
                                     json={
                                         "observability_trace_id": "test-trace-id", "reaction_type": "👍"
                                     })

        # Should return 404 since Weave is not configured
        assert response.status_code == 404


async def test_feedback_endpoint_requires_parameters() -> None:
    """Test that the feedback endpoint validates required parameters."""

    config = Config(
        general=GeneralConfig(front_end=FastApiFrontEndConfig(),
                              telemetry={"tracing": {
                                  "weave": WeaveTelemetryExporter(project="test-project")
                              }}),
        workflow=EchoFunctionConfig(),
    )

    async with build_nat_client(config, worker_class=WeaveFastAPIPluginWorker) as client:
        # Test with missing observability_trace_id
        response = await client.post("/feedback", json={"reaction_type": "👍"})
        assert response.status_code == 422

        # Test with missing reaction_type
        response = await client.post("/feedback", json={"observability_trace_id": "test-trace-id"})
        assert response.status_code == 422


async def test_weave_worker_adds_standard_routes() -> None:
    """Test that WeaveFastAPIPluginWorker still adds all standard routes."""

    config = Config(
        general=GeneralConfig(front_end=FastApiFrontEndConfig()),
        workflow=EchoFunctionConfig(),
    )

    async with build_nat_client(config, worker_class=WeaveFastAPIPluginWorker) as client:
        # Test that standard workflow endpoint exists
        response = await client.post("/generate", json={"message": "Hello"})
        assert response.status_code == 200
        assert response.json() == {"value": "Hello"}
