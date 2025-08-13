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
from pathlib import Path

import pytest

try:
    from nat_profiler_agent.tool.flow_chart import FlowChartConfig
    from nat_profiler_agent.tool.token_usage import TokenUsageConfig
    PROFILER_AGENT_AVAILABLE = True
except ImportError:
    PROFILER_AGENT_AVAILABLE = False

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder

logger = logging.getLogger(__name__)


# To run this test, a phoenix server must be running.
# The phoenix server can be started by running the following command:
# docker run -p 6006:6006 -p 4317:4317  arizephoenix/phoenix:latest
@pytest.mark.skipif(not PROFILER_AGENT_AVAILABLE, reason="aiq_profiler_agent is not installed")
async def test_flow_chart_tool():
    async with WorkflowBuilder() as builder:
        await builder.add_function("flow_chart", FlowChartConfig())
        flow_chart_tool = builder.get_tool("flow_chart", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        data_path = Path(__file__).parent / "test_spans.csv"
        result = await flow_chart_tool.ainvoke(input={"df_path": str(data_path)})
        assert len(result.trace_id_to_flow_info) == 1
        flow_info = result.trace_id_to_flow_info.popitem()[1]
        assert flow_info.flow_chart_path is not None and Path(flow_info.flow_chart_path).exists()


@pytest.mark.skipif(not PROFILER_AGENT_AVAILABLE, reason="aiq_profiler_agent is not installed")
async def test_token_usage_tool():
    async with WorkflowBuilder() as builder:
        await builder.add_function("token_usage", TokenUsageConfig())
        token_usage_tool = builder.get_tool("token_usage", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        data_path = Path(__file__).parent / "test_spans.csv"
        result = await token_usage_tool.ainvoke(input={"df_path": str(data_path)})
        assert len(result.trace_id_to_token_usage) == 1
        token_usage_info = result.trace_id_to_token_usage.popitem()[1]
        assert (token_usage_info.token_usage_detail_chart_path is not None
                and Path(token_usage_info.token_usage_detail_chart_path).exists())
