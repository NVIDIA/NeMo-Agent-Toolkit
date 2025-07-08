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

import json
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import requests
from aiq_alert_triage_agent.telemetry_metrics_host_performance_check_tool import \
    TelemetryMetricsHostPerformanceCheckToolConfig
from aiq_alert_triage_agent.telemetry_metrics_host_performance_check_tool import _get_llm_analysis_input
from aiq_alert_triage_agent.telemetry_metrics_host_performance_check_tool import _timeseries_stats

from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.data_models.component_ref import LLMRef


async def test_telemetry_metrics_host_performance_check_tool():
    # Test cases with expected API responses and outcomes
    test_cases = [
        # Test 1: Normal CPU usage pattern
        {
            'host_id': 'host1',
            'api_response': {
                'data': {
                    'result': [{
                        'values': [
                            [1642435200, "45.2"],  # Example timestamp and CPU usage
                            [1642438800, "47.8"],
                            [1642442400, "42.5"],
                        ]
                    }]
                }
            },
            'expected_success': True,
            'mock_llm_conclusion': 'CPU usage for host1 shows normal patterns with average utilization around 45%.'
        },
        # Test 2: High CPU usage pattern
        {
            'host_id':
                'host2',
            'api_response': {
                'data': {
                    'result': [{
                        'values': [
                            [1642435200, "85.2"],
                            [1642438800, "87.8"],
                            [1642442400, "92.5"],
                        ]
                    }]
                }
            },
            'expected_success':
                True,
            'mock_llm_conclusion':
                'Host host2 shows consistently high CPU utilization above 85%, indicating potential performance issues.'
        },
        # Test 3: API error scenario
        {
            'host_id': 'host3',
            'api_error': requests.exceptions.RequestException('Connection failed'),
            'expected_success': False
        }
    ]

    # Configure the tool
    config = TelemetryMetricsHostPerformanceCheckToolConfig(
        llm_name=LLMRef(value="dummy"),
        offline_mode=False,  # Testing in live mode
        metrics_url="http://test-monitoring-system:9090")

    # Set up mock builder and LLM
    mock_builder = AsyncMock()
    mock_llm = MagicMock()
    mock_builder.get_llm.return_value = mock_llm

    # Initialize workflow builder and add the function
    async with WorkflowBuilder() as builder:
        builder.get_llm = mock_builder.get_llm
        await builder.add_function("telemetry_metrics_host_performance_check", config)
        performance_check_tool = builder.get_tool("telemetry_metrics_host_performance_check",
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
                    result = await performance_check_tool.ainvoke(input=case['host_id'])

                    # Verify the result matches expected LLM conclusion
                    assert result == case['mock_llm_conclusion']

                    # Verify API call was made correctly
                    mock_get.assert_called_once()
                    args, kwargs = mock_get.call_args

                    # Verify the query parameters
                    params = kwargs['params']
                    host_id = case["host_id"]
                    assert params['query'] == f'(100 - cpu_usage_idle{{cpu="cpu-total",instance=~"{host_id}:9100"}})'
                    assert 'step' in params
                    # Should parse without error
                    datetime.fromisoformat(params['start'].replace('Z', '+00:00'))
                    datetime.fromisoformat(params['end'].replace('Z', '+00:00'))

                    # Verify LLM was called with processed data
                    mock_llm_invoke.assert_called_once()
                    # Verify LLM was called with correctly formatted data input
                    llm_call_args = mock_llm_invoke.call_args
                    user_prompt = llm_call_args[1]['user_prompt']
                    assert user_prompt.startswith('Timeseries:\n')  # Check format starts with timeseries
                    assert '\n\nTime Series Statistics' in user_prompt  # Check statistics section exists
                    assert all(stat in user_prompt for stat in [
                        'Number of Data Points:', 'Maximum Value:', 'Minimum Value:', 'Mean Value:', 'Median Value:'
                    ])  # Check all statistics are present

                else:
                    # Test error case
                    with pytest.raises(requests.exceptions.RequestException):
                        await performance_check_tool.ainvoke(input=case['host_id'])


def test_timeseries_stats():
    # Test case 1: Normal sequence of values
    ts1 = [45.2, 47.8, 42.5, 44.1, 46.3]
    result1 = _timeseries_stats(ts1)

    # Verify all expected statistics are present
    assert 'Number of Data Points: 5' in result1
    assert 'Maximum Value: 47.8' in result1
    assert 'Minimum Value: 42.5' in result1
    assert 'Mean Value: 45.18' in result1  # 225.9/5
    assert 'Median Value: 45.2' in result1

    # Test case 2: Single value
    ts2 = [42.0]
    result2 = _timeseries_stats(ts2)
    assert 'Number of Data Points: 1' in result2
    assert 'Maximum Value: 42.0' in result2
    assert 'Minimum Value: 42.0' in result2
    assert 'Mean Value: 42.00' in result2
    assert 'Median Value: 42.0' in result2

    # Test case 3: Empty list
    ts3 = []
    result3 = _timeseries_stats(ts3)
    assert "No data points" == result3

    # Test case 4: List with integer values
    ts4 = [1, 2, 3, 4, 5]
    result4 = _timeseries_stats(ts4)
    assert 'Number of Data Points: 5' in result4
    assert 'Maximum Value: 5' in result4
    assert 'Minimum Value: 1' in result4
    assert 'Mean Value: 3.00' in result4
    assert 'Median Value: 3' in result4


def test_get_llm_analysis_input():
    # Test case 1: Normal sequence of timestamp-value pairs
    def to_timestamp(date_str):
        return int(datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp())

    timestamp_value_list1 = [[to_timestamp("2025-04-17 12:00:00"),
                              "45.2"], [to_timestamp("2025-04-17 13:00:00"), "47.8"],
                             [to_timestamp("2025-04-17 14:00:00"), "42.5"]]
    result1 = _get_llm_analysis_input(timestamp_value_list1)

    # Parse the JSON part of the output
    timeseries_str = result1.split('\n\n')[0].replace('Timeseries:\n', '')
    timeseries_data = json.loads(timeseries_str)

    # Verify timestamp conversion and format
    assert len(timeseries_data) == 3
    assert timeseries_data[0][0] == "2025-04-17 12:00:00"
    assert timeseries_data[0][1] == "45.2"

    # Verify statistics section exists and contains all required fields
    assert 'Time Series Statistics' in result1
    assert 'Number of Data Points: 3' in result1
    assert 'Maximum Value: 47.8' in result1
    assert 'Minimum Value: 42.5' in result1
    assert 'Mean Value: 45.17' in result1
    assert 'Median Value: 45.2' in result1

    # Test case 2: Single timestamp-value pair
    timestamp_value_list2 = [[to_timestamp("2025-04-20 10:00:00"), "82.0"]]
    result2 = _get_llm_analysis_input(timestamp_value_list2)

    timeseries_str2 = result2.split('\n\n')[0].replace('Timeseries:\n', '')
    timeseries_data2 = json.loads(timeseries_str2)

    assert len(timeseries_data2) == 1
    assert timeseries_data2[0][0] == "2025-04-20 10:00:00"
    assert timeseries_data2[0][1] == "82.0"
    assert 'Number of Data Points: 1' in result2

    # Test case 3: Empty list
    timestamp_value_list3 = []
    result3 = _get_llm_analysis_input(timestamp_value_list3)
    assert "No data points" == result3

    # Test case 4: Mixed numeric types (integers and floats)
    timestamp_value_list4 = [
        [to_timestamp("2025-04-17 12:00:00"), "100"],  # Integer value
        [to_timestamp("2025-04-17 13:00:00"), "47.8"],  # Float value
        [to_timestamp("2025-04-17 14:00:00"), "50"]  # Integer value
    ]
    result4 = _get_llm_analysis_input(timestamp_value_list4)

    timeseries_str4 = result4.split('\n\n')[0].replace('Timeseries:\n', '')
    timeseries_data4 = json.loads(timeseries_str4)

    assert len(timeseries_data4) == 3
    assert all(isinstance(entry[1], str) for entry in timeseries_data4)  # All values should be strings
    assert 'Maximum Value: 100' in result4
    assert 'Minimum Value: 47.8' in result4
