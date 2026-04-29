# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for NAT Harbor NemoAgent bridge behavior."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from nat_harbor.agents.installed.nemo_agent import NemoAgent


def _make_agent(tmp_path: Path, **kwargs) -> NemoAgent:
    return NemoAgent(
        logs_dir=tmp_path / "logs",
        model_name="nvidia/meta/llama-3.3-70b-instruct",
        **kwargs,
    )


def test_resolve_workflow_packages_merges_and_dedupes(tmp_path: Path) -> None:
    agent = _make_agent(
        tmp_path,
        workflow_package="pkg_alpha",
        workflow_packages="pkg_alpha,pkg_beta,pkg_gamma",
    )
    assert agent._resolve_workflow_packages() == ["pkg_alpha", "pkg_beta", "pkg_gamma"]


def test_build_run_command_honors_python_bin(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path, python_bin="/opt/harbor-venv/bin/python")
    command = agent._build_run_command("hello")
    assert "/opt/harbor-venv/bin/python /installed-agent/nemo_agent_run_wrapper.py" in command


def test_build_run_command_fails_if_upstream_wrapper_shape_changes(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path, python_bin="/opt/harbor-venv/bin/python")

    with patch(
            "nat_harbor.agents.installed.nemo_agent.HarborNemoAgent._build_run_command",
            return_value="python3 /installed-agent/renamed_wrapper.py",
    ):
        with pytest.raises(RuntimeError, match="python_bin override"):
            agent._build_run_command("hello")


def test_library_mode_flag_resolves_in_agent(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path, library_mode=True)
    assert agent._resolved_flags["library_mode"] is True


def test_inline_generated_config_omits_api_key_from_logs(tmp_path: Path) -> None:
    api_key = "nvapi-secret-for-test"
    agent = _make_agent(tmp_path, extra_env={"NVIDIA_API_KEY": api_key})

    config_path = Path(agent._resolve_inline_config_path())

    assert config_path.exists()
    assert api_key not in config_path.read_text(encoding="utf-8")
    assert agent._build_env()["NVIDIA_API_KEY"] == api_key


def test_install_skips_local_mode_by_default(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    mock_env = AsyncMock()
    mock_env.type = Mock(return_value="local")

    asyncio.run(agent.install(mock_env))

    mock_env.exec.assert_not_called()
    policy_file = agent.logs_dir / "setup" / "install-policy.json"
    assert policy_file.exists()
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    assert policy["environment_type"] == "local"
    assert policy["install_executed"] is False


def test_install_allows_local_mode_with_opt_in(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path, local_install_policy="allow")
    mock_env = AsyncMock()
    mock_env.type = Mock(return_value="local")

    with patch(
            "nat_harbor.agents.installed.nemo_agent.HarborNemoAgent.install",
            new=AsyncMock(),
    ) as install_mock:
        asyncio.run(agent.install(mock_env))

    install_mock.assert_awaited_once_with(mock_env)
    policy = json.loads((agent.logs_dir / "setup" / "install-policy.json").read_text(encoding="utf-8"))
    assert policy["install_executed"] is True
    assert policy["local_install_allowed"] is True


def test_setup_installs_multiple_local_packages_in_order(tmp_path: Path) -> None:
    pkg_a = tmp_path / "pkg_a"
    pkg_b = tmp_path / "pkg_b"
    pkg_a.mkdir(parents=True)
    pkg_b.mkdir(parents=True)
    (pkg_a / "pyproject.toml").write_text("[project]\nname='pkg_a'\n", encoding="utf-8")
    (pkg_b / "pyproject.toml").write_text("[project]\nname='pkg_b'\n", encoding="utf-8")

    agent = _make_agent(tmp_path, workflow_packages=f"{pkg_a},{pkg_b}")
    mock_env = AsyncMock()
    mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")

    with patch(
            "nat_harbor.agents.installed.nemo_agent.BaseInstalledAgent.setup",
            new=AsyncMock(),
    ):
        asyncio.run(agent.setup(mock_env))

    upload_targets = [call.kwargs["target_dir"] for call in mock_env.upload_dir.call_args_list]
    assert "/installed-agent/workflow-package-0" in upload_targets
    assert "/installed-agent/workflow-package-1" in upload_targets

    install_commands = [
        call.kwargs["command"] for call in mock_env.exec.call_args_list
        if "pip install --no-deps /installed-agent/workflow-package-" in call.kwargs.get("command", "")
    ]
    assert len(install_commands) == 2
    assert "/installed-agent/workflow-package-0" in install_commands[0]
    assert "/installed-agent/workflow-package-1" in install_commands[1]
