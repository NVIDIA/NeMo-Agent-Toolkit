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
from unittest.mock import MagicMock
from unittest.mock import patch

from nat_alert_triage_agent.host_performance_check_tool import _parse_stdout_lines
from nat_alert_triage_agent.prompts import HostPerformanceCheckPrompts

EXAMPLE_CPU_USAGE_OUTPUT = """
03:45:00 PM  CPU    %usr   %nice    %sys %iowait    %irq   %soft  %steal  %guest  %gnice   %idle
03:45:01 PM  all    60.00    0.00    5.00    1.00     0.00    0.50     0.00     0.00     0.00    33.50
03:45:01 PM    0     95.00    0.00    3.00    0.50     0.00    0.50     0.00     0.00     0.00     1.00
03:45:01 PM    1     25.00    0.00    7.00    1.50     0.00    0.50     0.00     0.00     0.00    66.00"""

EXAMPLE_MEMORY_USAGE_OUTPUT = """
              total        used        free      shared  buff/cache   available
Mem:           7989        1234         512          89        6243        6521
Swap:          2047           0        2047"""

EXAMPLE_DISK_IO_OUTPUT = """
Device            r/s     w/s    rkB/s    wkB/s  rrqm/s  wrqm/s  %util  await  svctm
sca              20.0    80.0   1024.0   4096.0    0.0     0.0    98.0   120.0   1.2"""

EXAMPLE_LLM_PARSED_OUTPUT = json.dumps(
    {
        "cpu_usage": [{
            "timestamp": "03:45:01 PM",
            "cpu": "all",
            "user": 60.00,
            "nice": 0.00,
            "system": 5.00,
            "iowait": 1.00,
            "irq": 0.00,
            "softirq": 0.50,
            "steal": 0.00,
            "guest": 0.00,
            "gnice": 0.00,
            "idle": 33.50,
        },
                      {
                          "timestamp": "03:45:01 PM",
                          "cpu": "0",
                          "user": 95.00,
                          "nice": 0.00,
                          "system": 3.00,
                          "iowait": 0.50,
                          "irq": 0.00,
                          "softirq": 0.50,
                          "steal": 0.00,
                          "guest": 0.00,
                          "gnice": 0.00,
                          "idle": 1.00,
                      },
                      {
                          "timestamp": "03:45:01 PM",
                          "cpu": "1",
                          "user": 25.00,
                          "nice": 0.00,
                          "system": 7.00,
                          "iowait": 1.50,
                          "irq": 0.00,
                          "softirq": 0.50,
                          "steal": 0.00,
                          "guest": 0.00,
                          "gnice": 0.00,
                          "idle": 66.00,
                      }],
        "memory_usage": {
            "total": 7989,
            "used": 1234,
            "free": 512,
            "shared": 89,
            "buff_cache": 6243,
            "available": 6521,
        },
        "swap_usage": {
            "total": 2047,
            "used": 0,
            "free": 2047,
        },
        "disk_io": [{
            "device": "sca",
            "read_per_sec": 20.0,
            "write_per_sec": 80.0,
            "read_kB_per_sec": 1024.0,
            "write_kB_per_sec": 4096.0,
            "read_merge_per_sec": 0.0,
            "write_merge_per_sec": 0.0,
            "util_percent": 98.0,
            "await_ms": 120.0,
            "service_time_ms": 1.2,
        }]
    },
    sort_keys=True)


async def test_parse_stdout_lines_success():
    # Test data
    test_stdout_lines = [EXAMPLE_CPU_USAGE_OUTPUT, EXAMPLE_MEMORY_USAGE_OUTPUT, EXAMPLE_DISK_IO_OUTPUT]

    # Create mock config with parsing_prompt
    mock_config = MagicMock()
    mock_config.parsing_prompt = HostPerformanceCheckPrompts.PARSING_PROMPT

    # Mock the LLM response
    with patch('nat_alert_triage_agent.utils.llm_ainvoke') as mock_llm:
        mock_llm.return_value = EXAMPLE_LLM_PARSED_OUTPUT

        # Call the function
        result = await _parse_stdout_lines(
            config=mock_config,
            builder=None,  # unused, mocked
            stdout_lines=test_stdout_lines)

        # Verify the result
        assert result == EXAMPLE_LLM_PARSED_OUTPUT

        # Verify llm_ainvoke was called with correct prompt
        mock_llm.assert_called_once()
        call_args = mock_llm.call_args[1]
        assert 'config' in call_args
        assert 'builder' in call_args
        assert 'user_prompt' in call_args
        input_data = "\n".join(test_stdout_lines)
        assert call_args['user_prompt'] == HostPerformanceCheckPrompts.PARSING_PROMPT.format(input_data=input_data)


async def test_parse_stdout_lines_llm_error():
    # Simulate LLM throwing an exception
    with patch('nat_alert_triage_agent.utils.llm_ainvoke') as mock_llm:
        mock_llm.side_effect = Exception("LLM error")
        mock_llm.return_value = None

        # Create mock config with parsing_prompt
        mock_config = MagicMock()
        mock_config.parsing_prompt = HostPerformanceCheckPrompts.PARSING_PROMPT

        result = await _parse_stdout_lines(
            config=mock_config,
            builder=None,  # unused, mocked
            stdout_lines=["Some test output"])

        # Verify error is properly captured in response
        assert result == ('{"error": "Failed to parse stdout from the playbook run.",'
                          ' "exception": "LLM error", "raw_response": "None"}')
