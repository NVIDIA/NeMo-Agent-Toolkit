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

import subprocess
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from nat_alert_triage_agent.hardware_check_tool import _get_ipmi_monitor_data


# Fixtures for inputs and expected command
@pytest.fixture
def ipmi_args():
    return "1.1.1.1", "test_user", "test_pass"


@pytest.fixture
def expected_cmd(ipmi_args):
    ip, user, pwd = ipmi_args
    return [
        "ipmimonitoring",
        "-h",
        ip,
        "-u",
        user,
        "-p",
        pwd,
        "--privilege-level=USER",
    ]


# Fixture to mock subprocess.run
@pytest.fixture
def mock_run():
    with patch('subprocess.run') as m:
        yield m


# Parameterized test covering both success and failure
@pytest.mark.parametrize(
    "stdout, side_effect, expected",
    [
        # success case: subprocess returns stdout
        pytest.param("Sample IPMI output", None, "Sample IPMI output", id="success"),
        # failure case: subprocess raises CalledProcessError
        pytest.param(
            "unused output",
            subprocess.CalledProcessError(returncode=1, cmd=["ipmimonitoring"], stderr="Command failed"),
            None,  # expected None when ipmimonitoring command raises error
            id="failure"),
    ])
def test_get_ipmi_monitor_data(mock_run, ipmi_args, expected_cmd, stdout, side_effect, expected):
    # configure mock
    if side_effect:
        mock_run.side_effect = side_effect
    else:
        mock_result = MagicMock()
        mock_result.stdout = stdout
        mock_run.return_value = mock_result

    # invoke
    result = _get_ipmi_monitor_data(*ipmi_args)

    # assertions
    assert result == expected
    mock_run.assert_called_once_with(expected_cmd, capture_output=True, text=True, check=True)
