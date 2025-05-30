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
from unittest.mock import patch

from aiq_alert_triage_agent.monitoring_process_check_tool import _run_ansible_playbook_for_monitor_process_check
from aiq_alert_triage_agent.playbooks import MONITOR_PROCESS_CHECK_PLAYBOOK


async def test_run_ansible_playbook_for_monitor_process_check():
    # Test data
    ansible_host = "test.example.com"
    ansible_user = "testuser"
    ansible_port = 22
    ansible_private_key_path = "/path/to/key.pem"

    # Mock playbook output
    mock_playbook_output = {
        "task_results": [{
            "task": "Check process status",
            "host": ansible_host,
            "result": {
                "cmd":
                    "ps aux | grep monitoring",
                "stdout_lines": [
                    "user1  1234  0.0  0.2  12345 5678 ?  Ss  10:00  0:00 /usr/bin/monitoring-agent",
                    "user1  5678  2.0  1.0  23456 7890 ?  Sl  10:01  0:05 /usr/bin/monitoring-collector"
                ]
            }
        },
                         {
                             "task": "Check service status",
                             "host": ansible_host,
                             "result": {
                                 "cmd":
                                     "systemctl status monitoring-service",
                                 "stdout_lines": [
                                     "● monitoring-service.service - Monitoring Service", "   Active: active (running)"
                                 ]
                             }
                         }]
    }

    # Mock the run_ansible_playbook function
    with patch("aiq_alert_triage_agent.utils.run_ansible_playbook", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_playbook_output

        # Call the function
        result = await _run_ansible_playbook_for_monitor_process_check(
            ansible_host=ansible_host,
            ansible_user=ansible_user,
            ansible_port=ansible_port,
            ansible_private_key_path=ansible_private_key_path)

        # Verify run_ansible_playbook was called with correct arguments
        mock_run.assert_called_once_with(playbook=MONITOR_PROCESS_CHECK_PLAYBOOK,
                                         ansible_host=ansible_host,
                                         ansible_user=ansible_user,
                                         ansible_port=ansible_port,
                                         ansible_private_key_path=ansible_private_key_path)

        # Verify the result structure
        assert isinstance(result, list)
        assert len(result) == 2

        # Verify first task details
        first_task = result[0]
        assert first_task["task"] == "Check process status"
        assert first_task["host"] == ansible_host
        assert first_task["cmd"] == "ps aux | grep monitoring"
        assert len(first_task["stdout_lines"]) == 2
        assert "monitoring-agent" in first_task["stdout_lines"][0]
        assert "monitoring-collector" in first_task["stdout_lines"][1]

        # Verify second task details
        second_task = result[1]
        assert second_task["task"] == "Check service status"
        assert second_task["host"] == ansible_host
        assert second_task["cmd"] == "systemctl status monitoring-service"
        assert len(second_task["stdout_lines"]) == 2
        assert "monitoring-service.service" in second_task["stdout_lines"][0]
        assert "Active: active" in second_task["stdout_lines"][1]
