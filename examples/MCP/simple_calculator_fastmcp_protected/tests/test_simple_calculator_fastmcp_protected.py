# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
To run simple_calculator_fastmcp_protected e2e tests:
pytest -v -o log_cli=true --log-cli-level=INFO --run_integration --run_slow \
              examples/MCP/simple_calculator_fastmcp_protected/tests/test_simple_calculator_fastmcp_protected.py
"""
import asyncio
import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest


@pytest.fixture(name="nat_fastmcp_protected_host", scope="module")
def nat_fastmcp_protected_host_fixture() -> str:
    return os.environ.get("NAT_CI_FASTMCP_PROTECTED_HOST", "localhost")


@pytest.fixture(name="nat_fastmcp_protected_port", scope="module")
def nat_fastmcp_protected_port_fixture() -> str:
    return os.environ.get("NAT_CI_FASTMCP_PROTECTED_PORT", "9912")


@pytest.fixture(name="fastmcp_protected_base_url", scope="module")
def fastmcp_protected_base_url_fixture(
    nat_fastmcp_protected_host: str,
    nat_fastmcp_protected_port: str,
) -> str:
    return f"http://{nat_fastmcp_protected_host}:{nat_fastmcp_protected_port}"


@pytest.fixture(name="fastmcp_protected_process", scope="module")
async def fastmcp_protected_process_fixture(
    nat_fastmcp_protected_host: str,
    nat_fastmcp_protected_port: str,
    root_repo_dir: Path,
) -> subprocess.Popen:
    config_path = (root_repo_dir / "examples/MCP/simple_calculator_fastmcp_protected/configs/config-server.yml")

    env = os.environ.copy()
    env.pop("NAT_LOG_LEVEL", None)
    env.setdefault("NAT_CALCULATOR_RESOURCE_CLIENT_ID", "nat-mcp-resource-server")
    env.setdefault("NAT_CALCULATOR_RESOURCE_CLIENT_SECRET", "dummy-secret")

    cmd = [
        "nat",
        "fastmcp",
        "server",
        "run",
        "--config_file",
        str(config_path),
        "--host",
        nat_fastmcp_protected_host,
        "--port",
        nat_fastmcp_protected_port,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    assert proc.poll() is None, f"FastMCP server process failed to start: {proc.stdout.read()}"

    yield proc

    # Teardown
    i = 0
    while proc.poll() is None and i < 5:
        if i == 0:
            proc.terminate()
        else:
            proc.kill()
        await asyncio.sleep(0.1)
        i += 1

    assert proc.poll() is not None, "FastMCP server process failed to terminate"


@pytest.fixture(name="fastmcp_protected_ready", scope="module")
async def fastmcp_protected_ready_fixture(
    fastmcp_protected_process: subprocess.Popen,
    fastmcp_protected_base_url: str,
):
    discovery_url = f"{fastmcp_protected_base_url}/.well-known/oauth-protected-resource/mcp"
    deadline = time.time() + 30
    while time.time() < deadline:
        assert fastmcp_protected_process.poll() is None, \
            f"FastMCP server process has exited unexpectedly: {fastmcp_protected_process.stdout.read()}"
        try:
            response = httpx.get(discovery_url, timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.2)

    raise TimeoutError("FastMCP protected server did not expose discovery metadata in time")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("fastmcp_protected_ready")
async def test_fastmcp_protected_requires_auth(fastmcp_protected_base_url: str):
    response = httpx.get(f"{fastmcp_protected_base_url}/mcp", timeout=5.0)
    assert response.status_code == 401
