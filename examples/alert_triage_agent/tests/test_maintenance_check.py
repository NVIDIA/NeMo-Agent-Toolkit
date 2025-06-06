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

import importlib
import importlib.resources
import inspect
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from aiq_alert_triage_agent.maintenance_check import NO_ONGOING_MAINTENANCE_STR
from aiq_alert_triage_agent.maintenance_check import MaintenanceCheckToolConfig
from aiq_alert_triage_agent.maintenance_check import _get_active_maintenance
from aiq_alert_triage_agent.maintenance_check import _load_maintenance_data
from aiq_alert_triage_agent.maintenance_check import _parse_alert_data
from aiq_alert_triage_agent.register import AlertTriageAgentWorkflowConfig

from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.data_models.component_ref import LLMRef


def test_load_maintenance_data():
    # Load paths from config like in test_utils.py
    package_name = inspect.getmodule(AlertTriageAgentWorkflowConfig).__package__
    config_file: Path = importlib.resources.files(package_name).joinpath("configs",
                                                                         "config_offline_mode.yml").absolute()
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
        maintenance_data_path = config["functions"]["maintenance_check"]["static_data_path"]
    maintenance_data_path_abs = importlib.resources.files(package_name).joinpath("../../../../",
                                                                                 maintenance_data_path).absolute()

    # Test successful loading with actual maintenance data file
    df = _load_maintenance_data(maintenance_data_path_abs)

    # Verify DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    required_columns = {"host_id", "maintenance_start", "maintenance_end"}
    assert all(col in df.columns for col in required_columns)

    # Verify data types
    assert pd.api.types.is_datetime64_dtype(df["maintenance_start"])
    assert pd.api.types.is_datetime64_dtype(df["maintenance_end"])

    # Test with missing required columns
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        try:
            # Create CSV with missing columns
            f.write("host_id,some_other_column\n")
            f.write("test-host,value\n")
            f.flush()

            with pytest.raises(ValueError, match="Missing required columns: maintenance_end, maintenance_start"):
                _load_maintenance_data(f.name)
        finally:
            os.unlink(f.name)

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        _load_maintenance_data("nonexistent.csv")


@pytest.mark.parametrize(
    "input_msg,expected",
    [
        pytest.param("Alert received: {'host_id': 'server1', 'timestamp': '2024-03-21T10:00:00.000'} - Please check", {
            "host_id": "server1", "timestamp": "2024-03-21T10:00:00.000"
        },
                     id="valid_json_with_surrounding_text"),
        pytest.param('{"host_id": "server2", "timestamp": "2024-03-21T11:00:00.000"}', {
            "host_id": "server2", "timestamp": "2024-03-21T11:00:00.000"
        },
                     id="clean_json_without_surrounding_text"),
        pytest.param("{'host_id': 'server3', 'timestamp': '2024-03-21T12:00:00.000'}", {
            "host_id": "server3", "timestamp": "2024-03-21T12:00:00.000"
        },
                     id="json_with_single_quotes"),
        pytest.param("This is a message with no JSON", None, id="no_json_in_input"),
        pytest.param("Alert: {invalid json format} received", None, id="invalid_json_format"),
        pytest.param("{'host_id': 'server1'} {'host_id': 'server2'}", None, id="multiple_json_objects"),
        pytest.param(
            ("Nested JSON Alert: {'host_id': 'server4', 'details': {'location': 'rack1', 'metrics': "
             "{'cpu': 90, 'memory': 85}}, 'timestamp': '2024-03-21T13:00:00.000'}"),
            {
                "host_id": "server4",
                "details": {
                    "location": "rack1", "metrics": {
                        "cpu": 90, "memory": 85
                    }
                },
                "timestamp": "2024-03-21T13:00:00.000"
            },
            id="nested_json_structure"),
        pytest.param("Alert received:\n{'host_id': 'server5', 'timestamp': '2024-03-21T14:00:00.000'}\nPlease check", {
            "host_id": "server5", "timestamp": "2024-03-21T14:00:00.000"
        },
                     id="json_with_newlines"),
    ])
def test_parse_alert_data(input_msg, expected):
    result = _parse_alert_data(input_msg)
    assert result == expected


def test_get_active_maintenance():
    # Create test data
    test_data = {
        'host_id': ['host1', 'host1', 'host2', 'host3', 'host4'],
        'maintenance_start': [
            '2024-03-21 09:00:00',  # Active maintenance with end time
            '2024-03-21 14:00:00',  # Future maintenance
            '2024-03-21 09:00:00',  # Ongoing maintenance (no end time)
            '2024-03-21 08:00:00',  # Past maintenance
            '2024-03-21 09:00:00',  # Different host
        ],
        'maintenance_end': [
            '2024-03-21 11:00:00',
            '2024-03-21 16:00:00',
            None,
            '2024-03-21 09:00:00',
            '2024-03-21 11:00:00',
        ]
    }
    df = pd.DataFrame(test_data)
    df['maintenance_start'] = pd.to_datetime(df['maintenance_start'])
    df['maintenance_end'] = pd.to_datetime(df['maintenance_end'])

    # Test 1: Active maintenance with end time
    alert_time = datetime(2024, 3, 21, 10, 0, 0)
    result = _get_active_maintenance(df, 'host1', alert_time)
    assert result is not None
    start_str, end_str = result
    assert start_str == '2024-03-21 09:00:00'
    assert end_str == '2024-03-21 11:00:00'

    # Test 2: No active maintenance (future maintenance)
    alert_time = datetime(2024, 3, 21, 13, 0, 0)
    result = _get_active_maintenance(df, 'host1', alert_time)
    assert result is None

    # Test 3: Ongoing maintenance (no end time)
    alert_time = datetime(2024, 3, 21, 10, 0, 0)
    result = _get_active_maintenance(df, 'host2', alert_time)
    assert result is not None
    start_str, end_str = result
    assert start_str == '2024-03-21 09:00:00'
    assert end_str == ''  # Empty string for ongoing maintenance

    # Test 4: Past maintenance
    alert_time = datetime(2024, 3, 21, 10, 0, 0)
    result = _get_active_maintenance(df, 'host3', alert_time)
    assert result is None

    # Test 5: Non-existent host
    alert_time = datetime(2024, 3, 21, 10, 0, 0)
    result = _get_active_maintenance(df, 'host5', alert_time)
    assert result is None


async def test_maintenance_check_tool():
    # Create a temporary maintenance data file
    test_data = {
        'host_id': ['host1', 'host2'],
        'maintenance_start': ['2024-03-21 09:00:00', '2024-03-21 09:00:00'],
        'maintenance_end': ['2024-03-21 11:00:00', None]
    }
    # Test cases
    test_cases = [
        # Test 1: Valid alert during maintenance
        {
            'input': "{'host_id': 'host1', 'timestamp': '2024-03-21T10:00:00.000'}",
            'expected_maintenance': True,
            'mock_summary': 'Maintenance summary report'
        },
        # Test 2: Valid alert not during maintenance
        {
            'input': "{'host_id': 'host1', 'timestamp': '2024-03-21T12:00:00.000'}", 'expected_maintenance': False
        },
        # Test 3: Invalid JSON format
        {
            'input': "Invalid JSON data", 'expected_maintenance': False
        },
        # Test 4: Missing required fields
        {
            'input': "{'host_id': 'host1'}",  # Missing timestamp
            'expected_maintenance': False
        },
        # Test 5: Invalid timestamp format
        {
            'input': "{'host_id': 'host1', 'timestamp': 'invalid-time'}", 'expected_maintenance': False
        },
        # Test 6: Host under ongoing maintenance (no end time)
        {
            'input': "{'host_id': 'host2', 'timestamp': '2024-03-21T10:00:00.000'}",
            'expected_maintenance': True,
            'mock_summary': 'Ongoing maintenance summary'
        }
    ]

    # Create a temporary CSV file to store test maintenance data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        try:
            # Write test data to CSV file
            df = pd.DataFrame(test_data)
            df.to_csv(f.name, index=False)
            f.flush()

            # Set up mock builder and LLM
            mock_builder = AsyncMock()
            mock_llm = MagicMock()
            mock_builder.get_llm.return_value = mock_llm

            # Configure maintenance check tool
            config = MaintenanceCheckToolConfig(
                llm_name=LLMRef(value="dummy"),
                description="direct test",
                static_data_path=f.name,
            )

            # Initialize workflow builder and add maintenance check function
            async with WorkflowBuilder() as builder:
                builder.get_llm = mock_builder.get_llm
                await builder.add_function("maintenance_check", config)
                maintenance_check_tool = builder.get_tool("maintenance_check", wrapper_type=LLMFrameworkEnum.LANGCHAIN)

                # Run test cases
                for case in test_cases:
                    # Mock the alert summarization function
                    with patch('aiq_alert_triage_agent.maintenance_check._summarize_alert') as mock_summarize:
                        if case['expected_maintenance']:
                            mock_summarize.return_value = case['mock_summary']

                        # Invoke maintenance check tool with test input
                        result = await maintenance_check_tool.ainvoke(input=case['input'])

                        # Verify results based on whether maintenance was expected
                        if case['expected_maintenance']:
                            assert result == case['mock_summary']
                            mock_summarize.assert_called_once()
                            mock_summarize.reset_mock()
                        else:
                            assert result == NO_ONGOING_MAINTENANCE_STR
                            mock_summarize.assert_not_called()

        finally:
            # Clean up temporary file
            os.unlink(f.name)
