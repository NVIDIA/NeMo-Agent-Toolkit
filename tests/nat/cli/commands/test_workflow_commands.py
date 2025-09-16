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
from pathlib import Path
from unittest.mock import patch
import pytest
from nat.cli.commands.workflow.workflow_commands import _get_nat_dependency, get_repo_root

def test_get_repo_root(project_dir: str):
    assert get_repo_root() == Path(project_dir)

@patch('nat.cli.entrypoint.get_version')
@pytest.mark.parametrize(
    "versioned, expected_dep",
    [(True, "nvidia-nat[langchain]~=1.2"), (False, "nvidia-nat[langchain]")],
)
def test_get_nat_dependency(mock_get_version, versioned, expected_dep):
    mock_get_version.return_value = "1.2.3"
    result = _get_nat_dependency(versioned=versioned)
    assert result == expected_dep

def test_nat_workflow_create(tmp_path):
    """Test that 'nat workflow create' command creates expected structure."""
    result = subprocess.run(
        ["nat", "workflow", "create", "--no-install", "--workflow-dir", str(tmp_path), "test_workflow"],
        capture_output=True,
        text=True,
        check=True)

    assert result.returncode == 0

    workflow_root = tmp_path / "test_workflow"
    src_dir = workflow_root / "src"
    test_workflow_src = src_dir / "test_workflow"

    expected_output_paths = [
        workflow_root,
        workflow_root / "pyproject.toml",
        src_dir,
        test_workflow_src,
        test_workflow_src / "__init__.py",
        test_workflow_src / "register.py",
        test_workflow_src / "configs",
        test_workflow_src / "data",
        test_workflow_src / "configs" / "config.yml",
    ]

    for expected_output_path in expected_output_paths:
        assert expected_output_path.exists()

    expected_symlinks_and_targets = [
        (workflow_root / "configs", test_workflow_src / "configs"),
        (workflow_root / "data", test_workflow_src / "data"),
    ]

    for expected_symlink, target in expected_symlinks_and_targets:
        assert expected_symlink.is_symlink()
        assert expected_symlink.resolve() == target.resolve()
