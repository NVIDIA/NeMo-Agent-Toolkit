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
from click.testing import CliRunner
from nat.cli.commands.workflow.workflow_commands import create_command

def test_create_workflow_with_valid_name(tmp_path):
    """Ensure CLI succeeds with a valid workflow name."""
    runner = CliRunner()
    result = runner.invoke(
        create_command,
        ["my-workflow", "--no-install", "--workflow-dir", str(tmp_path)]
    )
    assert result.exit_code == 0
    assert "Workflow 'my-workflow' created successfully in" in result.output


    # Verify the workflow directory was actually created
    workflow_dir = tmp_path / "my-workflow"
    assert workflow_dir.is_dir()

    # Cleanup created workflow
    if workflow_dir.exists():
        for child in workflow_dir.iterdir():
            child.unlink()
        workflow_dir.rmdir()

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from nat.cli.commands.workflow.workflow_commands import _get_nat_dependency
from nat.cli.commands.workflow.workflow_commands import get_repo_root


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
    # Run the nat workflow create command
    result = subprocess.run(
        ["nat", "workflow", "create", "--no-install", "--workflow-dir", str(tmp_path), "test_workflow"],
        capture_output=True,
        text=True,
        check=True)

    # Verify the command succeeded
    assert result.returncode == 0

    # Define the expected paths
    workflow_root = tmp_path / "test_workflow"
    src_dir = workflow_root / "src"
    test_workflow_src = src_dir / "test_workflow"

    # Group all expected output paths
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

    # Verify all expected paths exist
    for expected_output_path in expected_output_paths:
        assert expected_output_path.exists()

    # Define expected symlinks
    expected_symlinks_and_targets = [
        (workflow_root / "configs", test_workflow_src / "configs"),
        (workflow_root / "data", test_workflow_src / "data"),
    ]

    # Verify symlinks exist and are symlinks
    for expected_symlink, target in expected_symlinks_and_targets:
        assert expected_symlink.is_symlink()
        assert expected_symlink.resolve() == target.resolve()
