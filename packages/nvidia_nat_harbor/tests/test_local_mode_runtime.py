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

import json
from pathlib import Path

import pytest

from nat_harbor.agents.installed.local_install_policy import is_local_install_allowed
from nat_harbor.agents.installed.local_install_policy import resolve_local_install_policy
from nat_harbor.agents.installed.nemo_agent_run_wrapper import normalize_result_text
from nat_harbor.environments.local_mode_policy import LocalWritePolicy
from nat_harbor.environments.local_mode_policy import is_shell_profile_write


def test_shell_profile_write_detection() -> None:
    assert is_shell_profile_write("echo PATH=foo >> ~/.bashrc")
    assert is_shell_profile_write("sed -i 's/x/y/' ~/.zshrc")
    assert not is_shell_profile_write("echo hello > /tmp/output.txt")


def test_local_write_policy_allows_inside_root(tmp_path: Path) -> None:
    policy = LocalWritePolicy(allowed_write_roots=(tmp_path,))
    policy.assert_allowed_write_path(tmp_path / "a" / "b.txt", "test-op")


def test_local_write_policy_blocks_outside_root(tmp_path: Path) -> None:
    outside = Path("/tmp/forbidden-path")
    policy = LocalWritePolicy(allowed_write_roots=(tmp_path,))
    with pytest.raises(PermissionError, match="outside allowed roots"):
        policy.assert_allowed_write_path(outside, "test-op")


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
    assert is_local_install_allowed("allow", None) is True
    assert is_local_install_allowed("skip", None) is False
    assert is_local_install_allowed("skip", "true") is True


def test_normalize_result_text_keeps_valid_json() -> None:
    payload = json.dumps([{"fn": {"name": "sum"}}])
    assert normalize_result_text(payload) == payload


def test_normalize_result_text_extracts_command_style_json() -> None:
    cmd = """echo '[{"fn":{"name":"multiply"}}]' > /app/result.json"""
    extracted = normalize_result_text(cmd)
    assert extracted == '[{"fn":{"name":"multiply"}}]'

