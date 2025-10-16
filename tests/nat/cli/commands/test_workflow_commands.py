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

from nat.cli.commands.workflow.workflow_commands import _get_nat_dependency
from nat.cli.commands.workflow.workflow_commands import _get_nat_version
from nat.cli.commands.workflow.workflow_commands import _is_nat_version_prerelease
from nat.cli.commands.workflow.workflow_commands import get_repo_root


def test_get_repo_root(project_dir: str):
    assert get_repo_root() == Path(project_dir)


@patch('nat.cli.entrypoint.get_version')
def test_get_nat_version_unknown(mock_get_version):
    mock_get_version.return_value = "unknown"
    assert _get_nat_version() is None


@patch('nat.cli.entrypoint.get_version')
@pytest.mark.parametrize(
    "input_version, expected",
    [
        ("1.2.3", "1.2"),
        ("1.2.0", "1.2"),
        ("1.2.3a1", "1.2.3a1"),
        ("1.2.0rc2", "1.2.0rc2"),
        ("1.2", "1.2"),
    ],
)
def test_get_nat_version_variants(mock_get_version, input_version, expected):
    mock_get_version.return_value = input_version
    assert _get_nat_version() == expected


@patch('nat.cli.entrypoint.get_version')
@pytest.mark.parametrize(
    "input_version, expected",
    [
        ("1.2.3", False),
        ("1.2.0", False),
        ("1.2.3a1", True),
        ("1.2.0rc2", True),
        ("1.2", False),
        ("unknown", False),
    ],
)
def test_is_nat_version_prerelease(mock_get_version, input_version, expected):
    mock_get_version.return_value = input_version
    assert _is_nat_version_prerelease() == expected


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


def test_create_workflow_with_invalid_name(tmp_path):
    """Ensure CLI fails with an invalid workflow name."""
    result = subprocess.run(
        ["nat", "workflow", "create", "--no-install", "--workflow-dir", str(tmp_path), " "],
        capture_output=True,
        text=True,
        check=False  # Expect failure, so don't raise exception
    )
    assert result.returncode != 0
    assert "Workflow name cannot be empty" in result.stderr
