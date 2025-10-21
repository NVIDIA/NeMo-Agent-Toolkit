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
import os
import subprocess
import time
from pathlib import Path

import pytest


@pytest.fixture(name="nat_mcp_host", scope="module")
def nat_mcp_host_fixture() -> str:
    return os.environ.get("NAT_CI_MCP_HOST", "localhost")


@pytest.fixture(name="nat_mcp_port", scope="module")
def nat_mcp_port_fixture() -> str:
    return os.environ.get("NAT_CI_MCP_PORT", "9901")


@pytest.fixture(name="nat_mcp_url", scope="module")
def nat_mcp_url_fixture(nat_mcp_host: str, nat_mcp_port: str) -> str:
    return f"http://{nat_mcp_host}:{nat_mcp_port}/mcp"


@pytest.fixture(name="simple_calc_mcp_process", scope="module")
async def simple_calc_mcp_process_fixture(nat_mcp_host: str, nat_mcp_port: str) -> subprocess.Popen:
    from nat.test.utils import locate_example_config
    from nat_simple_calculator.register import DivisionToolConfig

    config_file: Path = locate_example_config(DivisionToolConfig)

    env = os.environ.copy()
    env.pop("NAT_LOG_LEVEL", None)
    cmd = [
        "nat",
        "mcp",
        "serve",
        "--config_file",
        str(config_file.absolute()),
        "--host",
        nat_mcp_host,
        "--port",
        nat_mcp_port
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    assert proc.poll() is None, f"MCP server process failed to start: {proc.stdout.read()}"

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

    assert proc.poll() is not None, "MCP server process failed to terminate"


@pytest.fixture(name="simple_calc_mcp_avail", scope="module")
async def simple_calc_mcp_avail_fixture(simple_calc_mcp_process: subprocess.Popen, nat_mcp_url: str):
    """
    Wait for the MCP server to become available, then verify that the calculator_subtract tool is registered."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    deadline = time.time() + 30  # 30 second timeout
    while time.time() < deadline:
        assert simple_calc_mcp_process.poll() is None, \
            f"MCP server process has exited unexpectedly: {simple_calc_mcp_process.stdout.read()}"
        try:
            async with streamablehttp_client(nat_mcp_url) as (
                    read_stream,
                    write_stream,
                    _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    assert 'calculator_subtract' in (t.name for t in tools.tools)
                    return
        except Exception:
            pass

        await asyncio.sleep(0.1)

    raise TimeoutError("MCP server did not become available after 30 seconds")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "simple_calc_mcp_avail")
async def test_mcp_workflow(root_repo_dir: Path, nat_mcp_url: str):
    """
    This example runs two separate workflows, one which serves the calculator tool via MCP, along with the MCP client
    workflow. For the test we will launch the MCP server in a subprocess, then run the client workflow via the API.
    """
    from pydantic import HttpUrl

    from nat.runtime.loader import load_config
    from nat.test.utils import run_workflow

    config_path = root_repo_dir / "examples/MCP/simple_calculator_mcp/configs/config-mcp-client.yml"
    config = load_config(config_path)
    config.function_groups["mcp_math"].server.url = HttpUrl(nat_mcp_url)

    await run_workflow(config=config, question="Is 2 * 4 greater than 5?", expected_answer="yes")
