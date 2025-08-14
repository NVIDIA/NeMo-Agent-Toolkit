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
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from nat_alert_triage_agent.register import AlertTriageAgentWorkflowConfig
from nat_alert_triage_agent.utils import _DATA_CACHE
from nat_alert_triage_agent.utils import _LLM_CACHE
from nat_alert_triage_agent.utils import _get_llm
from nat_alert_triage_agent.utils import load_column_or_static
from nat_alert_triage_agent.utils import preload_offline_data
from nat_alert_triage_agent.utils import run_ansible_playbook

from nat.builder.framework_enum import LLMFrameworkEnum


async def test_get_llm():
    # Clear the cache before test
    _LLM_CACHE.clear()

    llm_name_1 = "test_llm"
    llm_name_2 = "different_llm"
    wrapper_type = LLMFrameworkEnum.LANGCHAIN

    # Create mock builder
    mock_builder = MagicMock()
    llms = {
        (llm_name_1, wrapper_type): object(),
        (llm_name_2, wrapper_type): object(),
    }
    mock_builder.get_llm = AsyncMock(side_effect=lambda llm_name, wrapper_type: llms[(llm_name, wrapper_type)])

    # Test first call - should create new LLM
    result = await _get_llm(mock_builder, llm_name_1, wrapper_type)

    # Verify LLM was created with correct parameters
    mock_builder.get_llm.assert_called_once_with(llm_name=llm_name_1, wrapper_type=wrapper_type)
    assert result is llms[(llm_name_1, wrapper_type)]

    # Verify cache state after first call
    assert len(_LLM_CACHE) == 1
    assert _LLM_CACHE[(llm_name_1, wrapper_type)] is llms[(llm_name_1, wrapper_type)]

    # Test second call with same parameters - should return cached LLM
    result2 = await _get_llm(mock_builder, llm_name_1, wrapper_type)

    # Verify get_llm was not called again
    mock_builder.get_llm.assert_called_once()
    assert result2 is llms[(llm_name_1, wrapper_type)]

    # Verify cache state hasn't changed
    assert len(_LLM_CACHE) == 1
    assert _LLM_CACHE[(llm_name_1, wrapper_type)] is llms[(llm_name_1, wrapper_type)]

    # Test with different parameters - should create new LLM
    result3 = await _get_llm(mock_builder, llm_name_2, wrapper_type)

    # Verify get_llm was called again with new parameters
    assert mock_builder.get_llm.call_count == 2
    mock_builder.get_llm.assert_called_with(llm_name=llm_name_2, wrapper_type=wrapper_type)
    assert result3 is llms[(llm_name_2, wrapper_type)]

    # Verify cache state after adding second LLM
    assert len(_LLM_CACHE) == 2
    assert _LLM_CACHE[(llm_name_1, wrapper_type)] is llms[(llm_name_1, wrapper_type)]
    assert _LLM_CACHE[(llm_name_2, wrapper_type)] is llms[(llm_name_2, wrapper_type)]


def test_preload_offline_data():
    # Clear the data cache before test
    _DATA_CACHE.clear()
    _DATA_CACHE.update({'offline_data': None, 'benign_fallback_offline_data': None})

    # Load paths from config
    package_name = inspect.getmodule(AlertTriageAgentWorkflowConfig).__package__
    config_file: Path = importlib.resources.files(package_name).joinpath("configs",
                                                                         "config_offline_mode.yml").absolute()
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
        offline_data_path = config["workflow"]["offline_data_path"]
        benign_fallback_data_path = config["workflow"]["benign_fallback_data_path"]
    offline_data_path_abs = importlib.resources.files(package_name).joinpath("../../../../",
                                                                             offline_data_path).absolute()
    benign_fallback_data_path_abs = importlib.resources.files(package_name).joinpath(
        "../../../../", benign_fallback_data_path).absolute()

    # Test successful loading with actual test files
    preload_offline_data(offline_data_path_abs, benign_fallback_data_path_abs)

    # Verify data was loaded correctly
    assert len(_DATA_CACHE) == 2
    assert isinstance(_DATA_CACHE['offline_data'], pd.DataFrame)
    assert isinstance(_DATA_CACHE['benign_fallback_offline_data'], dict)
    assert not _DATA_CACHE['offline_data'].empty
    assert len(_DATA_CACHE['benign_fallback_offline_data']) > 0

    # Test error cases
    with pytest.raises(ValueError, match="offline_data_path must be provided"):
        preload_offline_data(None, benign_fallback_data_path)

    with pytest.raises(ValueError, match="benign_fallback_data_path must be provided"):
        preload_offline_data(offline_data_path, None)

    # Test with non-existent files
    with pytest.raises(FileNotFoundError):
        preload_offline_data("nonexistent.csv", benign_fallback_data_path)

    with pytest.raises(FileNotFoundError):
        preload_offline_data(offline_data_path, "nonexistent.json")


def test_load_column_or_static():
    # Clear and initialize the data cache with test data
    _DATA_CACHE.clear()
    _DATA_CACHE.update({
        'offline_data': None,
        'benign_fallback_offline_data': {
            'static_column': 'static_value',
            'another_static': 'another_value',
            'potentially_null_column': 'static_value_for_nulls'
        }
    })

    # Create test DataFrame
    df = pd.DataFrame({
        'host_id': ['host1', 'host2', 'host3'],
        'string_column': ['value1', 'value2', 'value3'],
        'integer_column': [1, 2, 3]
    })

    # Test successful DataFrame column access
    assert load_column_or_static(df, 'host1', 'string_column') == 'value1'
    assert load_column_or_static(df, 'host2', 'integer_column') == 2

    # Test fallback to static JSON when column not in DataFrame
    assert load_column_or_static(df, 'host1', 'static_column') == 'static_value'
    assert load_column_or_static(df, 'host2', 'another_static') == 'another_value'

    # Test fallback to static JSON when DataFrame value is None, empty string, or NaN
    df_with_nulls = pd.DataFrame({
        'host_id': ['host1', 'host2', 'host3', 'host4'],
        'potentially_null_column': [None, '', pd.NA, 'value4'],
    })
    assert load_column_or_static(df_with_nulls, 'host1', 'potentially_null_column') == 'static_value_for_nulls'
    assert load_column_or_static(df_with_nulls, 'host2', 'potentially_null_column') == 'static_value_for_nulls'
    assert load_column_or_static(df_with_nulls, 'host3', 'potentially_null_column') == 'static_value_for_nulls'
    assert load_column_or_static(df_with_nulls, 'host4', 'potentially_null_column') == 'value4'

    # Test error when column not found in either source
    with pytest.raises(KeyError, match="Column 'nonexistent' not found in test and benign fallback data"):
        load_column_or_static(df, 'host1', 'nonexistent')

    # Test error when host_id not found
    with pytest.raises(KeyError, match="No row for host_id='unknown_host' in DataFrame"):
        load_column_or_static(df, 'unknown_host', 'string_column')

    # Test error when multiple rows found for same host_id
    df_duplicate = pd.DataFrame({
        'host_id': ['host1', 'host1', 'host2'], 'string_column': ['value1', 'value1_dup', 'value2']
    })
    with pytest.raises(ValueError, match="Multiple rows found for host_id='host1' in DataFrame"):
        load_column_or_static(df_duplicate, 'host1', 'string_column')

    # Test error when benign fallback data not preloaded
    _DATA_CACHE['benign_fallback_offline_data'] = None
    with pytest.raises(ValueError, match="Benign fallback test data not preloaded. Call `preload_offline_data` first."):
        load_column_or_static(df, 'host1', 'static_column')


def _mock_ansible_runner(status="successful", rc=0, events=None, stdout=None):
    """
    Build a dummy ansible_runner.Runner-like object.
    """
    runner = MagicMock()
    runner.status = status
    runner.rc = rc
    # Only set .events if given
    if events is not None:
        runner.events = events
    else:
        # Simulate no events
        if stdout is not None:
            runner.stdout = MagicMock()
            runner.stdout.read.return_value = stdout
        else:
            runner.stdout = None
        # Leave runner.events unset or empty
        runner.events = []

    return runner


@pytest.mark.parametrize(
    "status, rc, events, stdout, expected_tasks, expected_raw",
    [
        # 1) Successful run with two events
        (
            "successful",
            0,
            [
                {
                    "event": "runner_on_ok",
                    "event_data": {
                        "task": "test task", "host": "host1", "res": {
                            "changed": True, "stdout": "hello"
                        }
                    },
                    "stdout": "Task output",
                },
                {
                    "event": "runner_on_failed",
                    "event_data": {
                        "task": "failed task", "host": "host1", "res": {
                            "failed": True, "msg": "error"
                        }
                    },
                    "stdout": "Error output",
                },
            ],
            None,
            # Build expected task_results from events
            lambda evs: [{
                "task": ev["event_data"]["task"],
                "host": ev["event_data"]["host"],
                "status": ev["event"],
                "stdout": ev["stdout"],
                "result": ev["event_data"]["res"], }
                         for ev in evs if ev["event"] in ("runner_on_ok", "runner_on_failed")],
            None,
        ),
        # 2) No events but stdout present
        ("failed", 1, None, "Command failed output", lambda _: [], "Command failed output"),
        # 3) No events and no stdout
        ("failed", 1, None, None, lambda _: [], "No output captured."),
    ],
)
async def test_run_ansible_playbook_various(status, rc, events, stdout, expected_tasks, expected_raw):
    # Ansible parameters
    playbook = [{"name": "test task", "command": "echo hello"}]
    ansible_host = "test.example.com"
    ansible_user = "testuser"
    ansible_port = 22
    ansible_private_key_path = "/path/to/key.pem"

    runner = _mock_ansible_runner(status=status, rc=rc, events=events, stdout=stdout)

    # Patch ansible_runner.run
    with patch("ansible_runner.run", return_value=runner) as mock_run:
        result = await run_ansible_playbook(playbook,
                                            ansible_host,
                                            ansible_user,
                                            ansible_port,
                                            ansible_private_key_path)

        # Verify the call
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["playbook"] == playbook
        inv = call_kwargs["inventory"]["all"]["hosts"]["host1"]
        assert inv["ansible_host"] == ansible_host
        assert inv["ansible_user"] == ansible_user
        assert inv["ansible_ssh_private_key_file"] == ansible_private_key_path
        assert inv["ansible_port"] == ansible_port

        # Verify returned dict
        assert result["ansible_status"] == status
        assert result["return_code"] == rc
        assert result["task_results"] == expected_tasks(events or [])
        if not events:
            assert result["raw_output"] == expected_raw
