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

import json
import os
from pathlib import Path

import pytest

from nat.runtime.loader import load_config
from nat.runtime.loader import load_workflow
from nat.test.utils import build_nat_client

AGENT_CONFIGS = [
    "mixture_of_agents/configs/config.yml",
    "react/configs/config.yml",
    "react/configs/config-reasoning.yml",
    "tool_calling/configs/config.yml",
    "tool_calling/configs/config-reasoning.yml",
]
AGENT_IDS = ["mixture_of_agents", "react", "react-reasoning", "tool_calling", "tool_calling-reasoning"]


@pytest.fixture(name="agents_dir", scope="session")
def fixture_agents_dir(examples_dir: Path) -> Path:
    return examples_dir / "agents"


@pytest.fixture(name="question", scope="session")
def fixture_question() -> str:
    return "What are LLMs"


@pytest.fixture(name="answer", scope="session")
def fixture_answer() -> str:
    return "large language model"


@pytest.fixture(name="rewoo_data", scope="session")
def fixture_rewoo_data(agents_dir: Path) -> list[dict]:
    data_path = agents_dir / "data/rewoo.json"
    assert data_path.exists(), f"Data file {data_path} does not exist"
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(name="rewoo_session_manager", scope="class")
async def fixture_rewoo_session_manager(agents_dir: Path):
    """Build the ReWOO workflow once, share across all tests in the class."""
    async with load_workflow(agents_dir / "rewoo/configs/config.yml") as session_manager:
        yield session_manager


async def _build_nat_client(config_path: Path):
    config = load_config(config_path)
    old_val = os.environ.get("NAT_CONFIG_FILE")
    os.environ["NAT_CONFIG_FILE"] = str(config_path.absolute())
    try:
        async with build_nat_client(config) as client:
            yield client
    finally:
        if old_val is None:
            os.environ.pop("NAT_CONFIG_FILE", None)
        else:
            os.environ["NAT_CONFIG_FILE"] = old_val


@pytest.fixture(name="rewoo_nat_client", scope="class")
async def fixture_rewoo_nat_client(agents_dir: Path):
    """Build the ReWOO ASGI client once, share across all tests in the class."""
    config_path = agents_dir / "rewoo/configs/config.yml"
    async for client in _build_nat_client(config_path):
        yield client


@pytest.fixture(name="tool_calling_responses_api_nat_client", scope="module")
async def fixture_tool_calling_responses_api_nat_client(agents_dir: Path):
    """Build the Tool Calling Responses API ASGI client once, share across all tests in the class."""
    config_path = agents_dir / "tool_calling/configs/config-responses-api.yml"
    async for client in _build_nat_client(config_path):
        yield client


@pytest.fixture(name="agent_session_manager", scope="class", params=AGENT_CONFIGS, ids=AGENT_IDS)
async def fixture_agent_session_manager(request: pytest.FixtureRequest, agents_dir: Path):
    """Build each agent workflow once per config, share across all tests in the class."""
    async with load_workflow(agents_dir / request.param) as session_manager:
        yield session_manager


@pytest.fixture(name="agent_nat_client", scope="class", params=AGENT_CONFIGS, ids=AGENT_IDS)
async def fixture_agent_nat_client(request: pytest.FixtureRequest, agents_dir: Path):
    """Build each agent ASGI client once per config, share across all tests in the class."""
    config_path = agents_dir / request.param
    async for client in _build_nat_client(config_path):
        yield client
