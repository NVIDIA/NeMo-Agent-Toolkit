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

import logging
import os
from pathlib import Path

import pytest

try:
    from nat_profiler_agent.register import ProfilerAgentConfig
    from nat_profiler_agent.tool.flow_chart import FlowChartConfig
    from nat_profiler_agent.tool.token_usage import TokenUsageConfig
    PROFILER_AGENT_AVAILABLE = True
except ImportError:
    PROFILER_AGENT_AVAILABLE = False

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def require_profiler_agent(fail_missing: bool = False):
    if not PROFILER_AGENT_AVAILABLE:
        reason = "nat_profiler_agent is not installed"
        if fail_missing:
            raise RuntimeError(reason)
        pytest.skip(reason=reason)


@pytest.fixture(name="phoenix_url", scope="session")
def phoenix_url_fixture(fail_missing: bool) -> str:
    """
    To run these tests, a phoenix server must be running.
    The phoenix server can be started by running the following command:
    docker run -p 6006:6006 -p 4317:4317  arizephoenix/phoenix:latest
    """
    import requests

    url = os.getenv("NAT_CI_PHOENIX_URL", "http://localhost:6006/v1/traces")
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        return url
    except Exception as e:
        reason = f"Unable to connect to Phoenix server at {url}: {e}"
        if fail_missing:
            raise RuntimeError(reason)
        pytest.skip(reason=reason)


@pytest.fixture(name="df_path")
def df_path_fixture() -> Path:
    return Path(__file__).parent / "test_spans.csv"


@pytest.mark.integration
async def test_flow_chart_tool(df_path: Path):
    async with WorkflowBuilder() as builder:
        await builder.add_function("flow_chart", FlowChartConfig())
        flow_chart_tool = await builder.get_tool("flow_chart", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        result = await flow_chart_tool.ainvoke(input={"df_path": str(df_path)})
        assert len(result.trace_id_to_flow_info) == 1
        flow_info = result.trace_id_to_flow_info.popitem()[1]
        assert flow_info.flow_chart_path is not None and Path(flow_info.flow_chart_path).exists()


@pytest.mark.integration
async def test_token_usage_tool(df_path: Path):
    async with WorkflowBuilder() as builder:
        await builder.add_function("token_usage", TokenUsageConfig())
        token_usage_tool = await builder.get_tool("token_usage", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        result = await token_usage_tool.ainvoke(input={"df_path": str(df_path)})
        assert len(result.trace_id_to_token_usage) == 1
        token_usage_info = result.trace_id_to_token_usage.popitem()[1]
        assert (token_usage_info.token_usage_detail_chart_path is not None
                and Path(token_usage_info.token_usage_detail_chart_path).exists())


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_full_workflow(phoenix_url: str):
    from nat.runtime.loader import load_config
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow

    # This workflow requires a prior trace to be ingested into Phoenix.
    cur_dir = Path(__file__).parent
    examples_dir = cur_dir.parent.parent.parent
    simple_calc_observe_config = (examples_dir /
                                  "observability/simple_calculator_observability/configs/config-phoenix.yml")

    await run_workflow(simple_calc_observe_config, "add 1 and 2", "3", assert_expected_answer=False)

    config_file: Path = locate_example_config(ProfilerAgentConfig)
    config = load_config(config_file)
    config.general.telemetry.tracing["phoenix"].endpoint = phoenix_url

    await run_workflow(config_file, "Show me the token usage of last run", "tokens")
