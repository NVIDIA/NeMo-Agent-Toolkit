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

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(name="nat_a2a_host", scope="module")
def nat_a2a_host_fixture() -> str:
    return os.environ.get("NAT_CI_A2A_HOST", "localhost")


@pytest.fixture(name="nat_a2a_port", scope="module")
def nat_a2a_port_fixture() -> str:
    return os.environ.get("NAT_CI_A2A_PORT", "10000")


@pytest.fixture(name="nat_a2a_url", scope="module")
def nat_a2a_url_fixture(nat_a2a_host: str, nat_a2a_port: str) -> str:
    return f"http://{nat_a2a_host}:{nat_a2a_port}"


@pytest.fixture(name="simple_calc_a2a_server_process", scope="module")
async def simple_calc_a2a_server_process_fixture(nat_a2a_host: str, nat_a2a_port: str) -> subprocess.Popen:
    from nat.test.utils import locate_example_config
    from nat_simple_calculator.register import CalculatorToolConfig

    config_file: Path = locate_example_config(CalculatorToolConfig)

    env = os.environ.copy()
    env.pop("NAT_LOG_LEVEL", None)
    cmd = [
        "nat",
        "a2a",
        "serve",
        "--config_file",
        str(config_file.absolute()),
        "--host",
        nat_a2a_host,
        "--port",
        nat_a2a_port
    ]

    logger.info("Starting A2A server with command: %s", ' '.join(cmd))
    logger.info("Config file: %s", config_file)
    logger.info("Server URL: http://%s:%s", nat_a2a_host, nat_a2a_port)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    # Give it a moment to start
    await asyncio.sleep(0.5)

    if proc.poll() is not None:
        output = proc.stdout.read() if proc.stdout else "No output"
        raise RuntimeError(f"A2A server process failed to start. Exit code: {proc.returncode}\nOutput:\n{output}")

    logger.info("A2A server process started with PID: %s", proc.pid)

    yield proc

    # Teardown
    logger.info("Shutting down A2A server (PID: %s)", proc.pid)
    i = 0
    while proc.poll() is None and i < 5:
        if i == 0:
            proc.terminate()
        else:
            proc.kill()
        await asyncio.sleep(0.1)
        i += 1

    if proc.poll() is None:
        raise RuntimeError("A2A server process failed to terminate")

    logger.info("A2A server terminated with exit code: %s", proc.returncode)


@pytest.fixture(name="simple_calc_a2a_server_avail", scope="module")
async def simple_calc_a2a_server_avail_fixture(simple_calc_a2a_server_process: subprocess.Popen, nat_a2a_url: str):
    """
    Wait for the A2A server to become available, then verify that calculator skills are registered.
    """
    logger.info("Waiting for A2A server to become available at %s", nat_a2a_url)

    deadline = time.time() + 30  # 30 second timeout
    attempt = 0
    last_error = None

    while time.time() < deadline:
        attempt += 1

        # Check if process is still running
        if simple_calc_a2a_server_process.poll() is not None:
            output = simple_calc_a2a_server_process.stdout.read(
            ) if simple_calc_a2a_server_process.stdout else "No output"
            raise RuntimeError(
                f"A2A server process has exited unexpectedly with code {simple_calc_a2a_server_process.returncode}\n"
                f"Output:\n{output}")

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Discover the agent card using A2A protocol standard path
                discover_url = f"{nat_a2a_url}/.well-known/agent-card.json"
                if attempt % 50 == 0:  # Log every 5 seconds (50 * 0.1s)
                    logger.info("Attempt %s: Trying to connect to %s", attempt, discover_url)

                response = await client.get(discover_url)

                if response.status_code == 200:
                    agent_card = response.json()
                    logger.info("Successfully connected to A2A server!")
                    logger.info("Agent card: %s", agent_card.get('name', 'Unknown'))

                    # Verify calculator skills are registered
                    skills = agent_card.get("skills", [])
                    skill_names = [skill.get("name", "") for skill in skills]
                    logger.info("Found %s skills: %s", len(skills), skill_names)

                    # Check for at least one calculator skill (transformed name format)
                    calculator_skills = [name for name in skill_names if "Calculator" in name]
                    if len(calculator_skills) > 0:
                        logger.info("Found calculator skills: %s", calculator_skills)
                        return
                    else:
                        raise AssertionError(f"No calculator skills found in: {skill_names}")
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    if attempt % 50 == 0:
                        logger.warning("Server responded with status %s", response.status_code)
        except httpx.ConnectError as e:
            last_error = f"Connection error: {e}"
            if attempt % 50 == 0:
                logger.debug("Connection failed: %s", e)
        except (httpx.TimeoutException, httpx.HTTPError, AssertionError) as e:
            last_error = f"Error: {type(e).__name__}: {e}"
            if attempt % 50 == 0:
                logger.debug("Error during connection attempt: %s", e)

        await asyncio.sleep(0.1)

    # Timeout reached - provide detailed error
    raise TimeoutError(f"A2A server did not become available after 30 seconds ({attempt} attempts)\n"
                       f"Last error: {last_error}\n"
                       f"Server URL: {nat_a2a_url}\n"
                       f"Process status: {'running' if simple_calc_a2a_server_process.poll() is None else 'exited'}")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "simple_calc_a2a_server_avail")
async def test_a2a_workflow(root_repo_dir: Path, nat_a2a_url: str):
    """
    This example runs two separate workflows: one which serves the calculator tool through A2A,
    along with the A2A client workflow. For the test we will launch the A2A server in a subprocess,
    then run the client workflow through the API.
    """
    from pydantic import HttpUrl

    from nat.runtime.loader import load_config
    from nat.test.utils import run_workflow

    logger.info("Starting workflow test")
    logger.info("Root repo dir: %s", root_repo_dir)
    logger.info("A2A server URL: %s", nat_a2a_url)

    config_path = root_repo_dir / "examples/A2A/math_assistant_a2a/configs/config.yml"
    logger.info("Loading config from: %s", config_path)

    config = load_config(config_path)
    config.function_groups["calculator_a2a"].url = HttpUrl(nat_a2a_url)

    logger.info("Running workflow with question: 'Is 2 * 4 greater than 5?'")
    await run_workflow(config=config,
                       question="Is 2 * 4 greater than 5?",
                       expected_answer="yes",
                       session_kwargs={"user_id": "test-user"})
    logger.info("Workflow completed successfully!")
