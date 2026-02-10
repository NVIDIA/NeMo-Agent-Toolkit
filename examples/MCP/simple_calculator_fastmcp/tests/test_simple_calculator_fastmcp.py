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
To run simple_calculator_fastmcp e2e tests:
pytest -v -o log_cli=true --log-cli-level=INFO --run_integration --run_slow \
              examples/MCP/simple_calculator_fastmcp/tests/test_simple_calculator_fastmcp.py
"""

import asyncio
import os
import subprocess
import time
from pathlib import Path

import pytest


@pytest.fixture(name="nat_fastmcp_host", scope="module")
def nat_fastmcp_host_fixture() -> str:
    return os.environ.get("NAT_CI_FASTMCP_HOST", "localhost")


@pytest.fixture(name="nat_fastmcp_port", scope="module")
def nat_fastmcp_port_fixture() -> str:
    return os.environ.get("NAT_CI_FASTMCP_PORT", "9902")


@pytest.fixture(name="nat_fastmcp_url", scope="module")
def nat_fastmcp_url_fixture(nat_fastmcp_host: str, nat_fastmcp_port: str) -> str:
    return f"http://{nat_fastmcp_host}:{nat_fastmcp_port}/mcp"


@pytest.fixture(name="simple_calc_fastmcp_process", scope="module")
async def simple_calc_fastmcp_process_fixture(
    nat_fastmcp_host: str,
    nat_fastmcp_port: str,
    root_repo_dir: Path,
) -> subprocess.Popen:
    config_file = (root_repo_dir /
                   "examples/getting_started/simple_calculator/src/nat_simple_calculator/configs/config.yml")

    env = os.environ.copy()
    env.pop("NAT_LOG_LEVEL", None)
    cmd = [
        "nat",
        "fastmcp",
        "server",
        "run",
        "--config_file",
        str(config_file.absolute()),
        "--host",
        nat_fastmcp_host,
        "--port",
        nat_fastmcp_port
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


@pytest.fixture(name="simple_calc_fastmcp_avail", scope="module")
async def simple_calc_fastmcp_avail_fixture(simple_calc_fastmcp_process: subprocess.Popen, nat_fastmcp_url: str):
    """
    Wait for the FastMCP server to become available, then verify that the calculator__subtract tool is registered.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    deadline = time.time() + 30  # 30 second timeout
    while time.time() < deadline:
        assert simple_calc_fastmcp_process.poll() is None, \
            f"FastMCP server process has exited unexpectedly: {simple_calc_fastmcp_process.stdout.read()}"
        try:
            async with streamablehttp_client(nat_fastmcp_url) as (
                    read_stream,
                    write_stream,
                    _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    assert 'calculator__subtract' in (t.name for t in tools.tools)
                    return
        except Exception:
            pass

        await asyncio.sleep(0.1)

    raise TimeoutError("FastMCP server did not become available after 30 seconds")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "simple_calc_fastmcp_avail")
async def test_fastmcp_workflow(root_repo_dir: Path, nat_fastmcp_url: str):
    """
    This example runs two separate workflows, one which serves the calculator tool via FastMCP,
    along with the MCP client workflow. For the test we will launch the FastMCP server in a subprocess,
    then run the client workflow via the API.
    """
    from pydantic import HttpUrl

    from nat.runtime.loader import load_config
    from nat.test.utils import run_workflow

    config_path = root_repo_dir / "examples/MCP/simple_calculator_fastmcp/configs/config-mcp-client.yml"
    config = load_config(config_path)
    config.function_groups["mcp_math"].server.url = HttpUrl(nat_fastmcp_url)

    await run_workflow(config=config, question="Is 2 * 4 greater than 5?", expected_answer="yes")
