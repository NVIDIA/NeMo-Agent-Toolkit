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
"""Tests for NAT Harbor local-mode runtime helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from nat_harbor.agents.installed.nemo_agent_run_wrapper import normalize_result_text
from nat_harbor.agents.installed.policy import is_local_install_allowed
from nat_harbor.agents.installed.policy import resolve_local_install_policy
from nat_harbor.environments.local import LocalEnvironment
from nat_harbor.environments.local import is_shell_profile_write


def _make_local_environment(tmp_path: Path) -> LocalEnvironment:
    trial_paths = TrialPaths(tmp_path / "trial")
    return LocalEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="test-env",
        session_id="test-session",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(),
    )


def test_shell_profile_write_detection() -> None:
    assert is_shell_profile_write("echo PATH=foo >> ~/.bashrc")
    assert is_shell_profile_write("sed -i 's/x/y/' ~/.zshrc")
    assert not is_shell_profile_write("echo hello > /tmp/output.txt")


def test_local_environment_type() -> None:
    assert LocalEnvironment.type() == "local"


def test_local_environment_command_translation_preserves_longer_paths(tmp_path) -> None:
    env = object.__new__(LocalEnvironment)
    env._path_map = [("/app", tmp_path / "app")]

    translated = env._translate_command("cp /app/result.json /application/result.json")

    assert str(tmp_path / "app" / "result.json") in translated
    assert "/application/result.json" in translated


def test_local_environment_timeout_kills_child_processes(tmp_path: Path) -> None:
    env = _make_local_environment(tmp_path)
    marker_path = tmp_path / "child-survived"

    async def run_timeout() -> None:
        result = await env.exec(
            "bash -c 'sleep 0.5; touch \"$MARKER_PATH\"' & wait",
            env={"MARKER_PATH": str(marker_path)},
            timeout_sec=0.1,
        )

        assert result.return_code == 124
        assert result.stderr == "Command timed out"

        await asyncio.sleep(0.8)
        assert not marker_path.exists()

    asyncio.run(run_timeout())


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("skip", "skip"),
        ("allow", "allow"),
        (True, "allow"),
        (False, "skip"),
    ],
)
def test_resolve_local_install_policy(raw, normalized: str) -> None:
    assert resolve_local_install_policy(raw) == normalized


def test_is_local_install_allowed() -> None:
    assert is_local_install_allowed("allow") is True
    assert is_local_install_allowed("skip") is False


def test_normalize_result_text_keeps_valid_json() -> None:
    payload = json.dumps([{"fn": {"name": "sum"}}])
    assert normalize_result_text(payload) == payload


def test_normalize_result_text_extracts_command_style_json() -> None:
    cmd = """echo '[{"fn":{"name":"multiply"}}]' > /app/result.json"""
    extracted = normalize_result_text(cmd)
    assert extracted == '[{"fn":{"name":"multiply"}}]'
