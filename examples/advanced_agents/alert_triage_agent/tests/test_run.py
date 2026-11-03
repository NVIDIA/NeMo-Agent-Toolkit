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
from pathlib import Path
from unittest.mock import patch

import pytest

from nat_alert_triage_agent import run

DEFAULT_CONFIG_FILE = 'examples/advanced_agents/alert_triage_agent/configs/config_live_mode.yml'
DEFAULT_CONFIG_PATH = (Path(__file__).resolve().parents[1] / 'configs' / 'config_live_mode.yml').resolve()
REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    run.app.config['TESTING'] = True
    with run.app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    run.processed_alerts = []
    run.ENV_FILE = '.placeholder_env_file_value'
    run.CONFIG_FILE = '.placeholder_config_file_value'


def test_hsts_header(client):
    """Test that HSTS header is properly set."""
    response = client.get('/')
    assert response.headers['Strict-Transport-Security'] == 'max-age=31536000; includeSubDomains; preload'


@pytest.mark.parametrize('alert',
                         [{
                             "alert_id": 1,
                             "alert_name": "InstanceDown",
                             "host_id": "test-instance-1.example.com",
                             "severity": "critical",
                             "description": "Test description",
                             "summary": "Test summary",
                             "timestamp": "2025-04-28T05:00:00.000000"
                         },
                          {
                              "alert_id": 2,
                              "alert_name": "CPUUsageHighError",
                              "host_id": "test-instance-2.example.com",
                              "severity": "warning",
                              "description": "High CPU usage",
                              "summary": "CPU at 95%",
                              "timestamp": "2025-04-28T06:00:00.000000"
                          }])
def test_receive_single_alert(client, alert):
    """Test receiving a single alert with different alert types."""
    with patch('nat_alert_triage_agent.run.start_process') as mock_start_process:
        response = client.post('/alerts', data=json.dumps(alert), content_type='application/json')

        data = json.loads(response.data)
        assert response.status_code == 200
        assert data['received_alert_count'] == 1
        assert data['total_launched'] == 1
        mock_start_process.assert_called_once()


def test_receive_multiple_alerts(client):
    """Test receiving multiple alerts in a single request with different counts."""
    alert_count = 3
    test_alerts = [{
        "alert_id": i,
        "alert_name": f"TestAlert{i}",
        "host_id": f"test-instance-{i}.example.com",
        "severity": "critical",
        "timestamp": "2025-04-28T05:00:00.000000"
    } for i in range(alert_count)]

    with patch('nat_alert_triage_agent.run.start_process') as mock_start_process:
        response = client.post('/alerts', data=json.dumps(test_alerts), content_type='application/json')

        data = json.loads(response.data)
        assert response.status_code == 200
        assert data['received_alert_count'] == alert_count
        assert data['total_launched'] == alert_count
        assert mock_start_process.call_count == alert_count

        # post again to test that the total_launched is cumulative
        response = client.post('/alerts', data=json.dumps(test_alerts), content_type='application/json')

        data = json.loads(response.data)
        assert response.status_code == 200
        assert data['received_alert_count'] == alert_count
        assert data['total_launched'] == alert_count * 2
        assert mock_start_process.call_count == alert_count * 2


@pytest.mark.parametrize(
    'invalid_data,expected_error',
    [
        pytest.param('invalid json', 'Invalid JSON', id='invalid_syntax'),
        pytest.param('{incomplete json', 'Invalid JSON', id='incomplete_json'),
        pytest.param('[1, 2, 3]', "Alerts not represented as dictionaries",
                     id='wrong_alert_format'),  # Valid JSON but invalid alert format
        pytest.param('{"key": "value"}', "`alert_id` is absent in the alert payload",
                     id='missing_alert_id')  # Valid JSON but invalid alert format
    ])
def test_invalid_json(client, invalid_data, expected_error):
    """Test handling of various invalid JSON data formats."""
    response = client.post('/alerts', data=invalid_data, content_type='application/json')

    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] == expected_error


@pytest.mark.parametrize(
    'args,expected',
    [
        pytest.param(['--host', '127.0.0.1', '--port', '8080', '--env_file', '/custom/.env'], {
            'host': '127.0.0.1', 'port': 8080, 'env_file': '/custom/.env', 'config_file': DEFAULT_CONFIG_FILE
        },
                     id='custom_host_port_env_file'),
        pytest.param([], {
            'host': '0.0.0.0', 'port': 5000, 'env_file': '.env', 'config_file': DEFAULT_CONFIG_FILE
        },
                     id='default_args'),
        pytest.param(['--port', '3000'], {
            'host': '0.0.0.0', 'port': 3000, 'env_file': '.env', 'config_file': DEFAULT_CONFIG_FILE
        },
                     id='partial_override'),
        pytest.param(['--config_file', '/custom/config.yml'], {
            'host': '0.0.0.0', 'port': 5000, 'env_file': '.env', 'config_file': '/custom/config.yml'
        },
                     id='custom_config_file')
    ])
def test_parse_args(args, expected):
    """Test command line argument parsing with different argument combinations."""
    with patch('sys.argv', ['script.py'] + args):
        parsed_args = run.parse_args()
        assert parsed_args.host == expected['host']
        assert parsed_args.port == expected['port']
        assert parsed_args.env_file == expected['env_file']
        assert parsed_args.config_file == expected['config_file']
        if expected['config_file'] == DEFAULT_CONFIG_FILE:
            default_path = (REPO_ROOT / parsed_args.config_file).resolve()
            assert default_path == DEFAULT_CONFIG_PATH
            assert default_path.is_file()


@pytest.mark.parametrize('exit_code,expect_message',
                         [pytest.param(0, False, id='clean_exit'), pytest.param(2, True, id='nonzero_exit')])
def test_start_process_config_file_and_exit_logging(capsys, exit_code, expect_message):
    """Test that start_process runs the configured workflow file and logs a triage run that fails."""
    alert = {"alert_id": 1, "alert_name": "InstanceDown", "host_id": "test-instance-1.example.com"}

    with patch('nat_alert_triage_agent.run.subprocess.Popen') as mock_popen:
        process = mock_popen.return_value.__enter__.return_value
        process.wait.return_value = exit_code
        process.returncode = exit_code

        run.start_process(alert, run.ENV_FILE)

        cmd = mock_popen.call_args.args[0]
        assert f"--config_file={run.CONFIG_FILE}" in cmd
        captured = capsys.readouterr()
        assert (f"exited with {exit_code}" in captured.out) == expect_message
