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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import requests
from aiq_alert_triage_agent.telemetry_metrics_host_heartbeat_check_tool import \
    TelemetryMetricsHostHeartbeatCheckToolConfig

from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.data_models.component_ref import LLMRef


async def test_telemetry_metrics_host_heartbeat_check_tool():
    # Test cases with expected API responses and outcomes
    test_cases = [
        # Test 1: Host is up and reporting metrics
        {
            'host_id': 'host1',
            'api_response': {
                'data': {
                    'result': [{
                        'metric': {
                            'instance': 'host1:9100'
                        },
                        'value': [1234567890, '1']  # Timestamp and "up" value
                    }]
                }
            },
            'expected_success': True,
            'mock_llm_conclusion': 'Host host1 is up and reporting metrics normally.'
        },
        # Test 2: Host is down (no metrics reported)
        {
            'host_id': 'host2',
            'api_response': {
                'data': {
                    'result': []  # Empty result indicates no metrics reported
                }
            },
            'expected_success': True,
            'mock_llm_conclusion': 'Host host2 appears to be down - no heartbeat metrics reported.'
        },
        # Test 3: API error scenario
        {
            'host_id': 'host3',
            'api_error': requests.exceptions.RequestException('Connection failed'),
            'expected_success': False
        }
    ]

    # Configure the tool
    config = TelemetryMetricsHostHeartbeatCheckToolConfig(
        llm_name=LLMRef(value="dummy"),
        offline_mode=False,  # Important: testing in live mode
        metrics_url="http://test-monitoring-system:9090")

    # Set up mock builder and LLM
    mock_builder = AsyncMock()
    mock_llm = MagicMock()
    mock_builder.get_llm.return_value = mock_llm

    # Initialize workflow builder and add the function
    async with WorkflowBuilder() as builder:
        builder.get_llm = mock_builder.get_llm
        await builder.add_function("telemetry_metrics_host_heartbeat_check", config)
        heartbeat_check_tool = builder.get_tool("telemetry_metrics_host_heartbeat_check",
                                                wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        # Run test cases
        for case in test_cases:
            # Mock the requests.get call
            with patch('requests.get') as mock_get, \
                 patch('aiq_alert_triage_agent.utils.llm_ainvoke') as mock_llm_invoke:

                if 'api_error' in case:
                    # Simulate API error
                    mock_get.side_effect = case['api_error']
                else:
                    # Mock successful API response
                    mock_response = MagicMock()
                    mock_response.json.return_value = case['api_response']
                    mock_get.return_value = mock_response

                if case['expected_success']:
                    # Set up LLM mock response for successful cases
                    mock_llm_invoke.return_value = case['mock_llm_conclusion']

                    # Invoke tool and verify results
                    result = await heartbeat_check_tool.ainvoke(input=case['host_id'])

                    # Verify the result matches expected LLM conclusion
                    assert result == case['mock_llm_conclusion']

                    # Verify API call was made correctly
                    mock_get.assert_called_once()
                    args, kwargs = mock_get.call_args
                    assert kwargs['params']['query'] == f'up{{instance=~"{case["host_id"]}:9100"}}'

                    # Verify LLM was called
                    mock_llm_invoke.assert_called_once()
                else:
                    # Test error case
                    with pytest.raises(requests.exceptions.RequestException):
                        await heartbeat_check_tool.ainvoke(input=case['host_id'])
