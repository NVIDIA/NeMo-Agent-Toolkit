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

import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from nat.cli.commands.workflow.workflow_commands import _get_nat_dependency
from nat.cli.commands.workflow.workflow_commands import _get_nat_version
from nat.cli.commands.workflow.workflow_commands import _is_nat_version_prerelease
from nat.cli.commands.workflow.workflow_commands import create_command
from nat.cli.commands.workflow.workflow_commands import delete_command
from nat.cli.commands.workflow.workflow_commands import get_repo_root
from nat.cli.commands.workflow.workflow_commands import reinstall_command


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


@pytest.mark.parametrize("workflow_name, module_name", [("test_workflow", "test_workflow"),
                                                        ("my-workflow", "my_workflow")])
def test_nat_workflow_create(tmp_path, workflow_name: str, module_name: str):
    """Test that 'nat workflow create' command creates expected structure."""
    # Run the nat workflow create command
    result = subprocess.run(
        ["nat", "workflow", "create", "--no-install", "--workflow-dir", str(tmp_path), workflow_name],
        capture_output=True,
        text=True,
        check=True)

    # Verify the command succeeded
    assert result.returncode == 0

    # Define the expected paths
    workflow_root = tmp_path / workflow_name
    src_dir = workflow_root / "src"
    test_workflow_src = src_dir / module_name

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

    # The generated config must refer to the function by the name the generated module registers it under
    workflow_module = (test_workflow_src / f"{module_name}.py").read_text()
    registered_names = re.findall(r'\(FunctionBaseConfig, name="([^"]+)"\)', workflow_module)
    assert len(registered_names) == 1

    config = yaml.safe_load((test_workflow_src / "configs" / "config.yml").read_text())
    assert config["functions"][module_name]["_type"] == registered_names[0]
    assert module_name in config["workflow"]["tool_names"]


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


def test_create_workflow_that_already_exists_fails(tmp_path):
    (tmp_path / "test_workflow").mkdir()

    result = CliRunner().invoke(create_command, ["--no-install", "--workflow-dir", str(tmp_path), "test_workflow"])

    assert result.exit_code == 1
    assert "Workflow 'test_workflow' already exists" in result.output


def test_create_workflow_with_failed_install_fails_and_cleans_up(tmp_path):
    install_error = subprocess.CalledProcessError(1, ["uv", "pip", "install"], stderr="no matching distribution")

    with patch("nat.cli.commands.workflow.workflow_commands.subprocess.run", side_effect=install_error):
        result = CliRunner().invoke(create_command, ["--workflow-dir", str(tmp_path), "test_workflow"])

    assert result.exit_code == 1
    assert "An error occurred while creating the workflow" in result.output
    assert "no matching distribution" in result.output
    # The partial workflow is removed, so the same command can be run again
    assert not (tmp_path / "test_workflow").exists()


def test_create_workflow_with_failed_install_keeps_other_files(tmp_path):
    (tmp_path / "other").mkdir()
    install_error = subprocess.CalledProcessError(1, ["uv", "pip", "install"])

    with patch("nat.cli.commands.workflow.workflow_commands.subprocess.run", side_effect=install_error):
        result = CliRunner().invoke(create_command, ["--workflow-dir", str(tmp_path), "test_workflow"])

    assert result.exit_code == 1
    assert (tmp_path / "other").is_dir()


def test_create_workflow_in_missing_directory_fails(tmp_path):
    result = CliRunner().invoke(create_command,
                                ["--no-install", "--workflow-dir", str(tmp_path / "missing"), "test_workflow"])

    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_reinstall_missing_workflow_fails():
    result = CliRunner().invoke(reinstall_command, ["nat_workflow_that_is_not_installed"])

    assert result.exit_code == 1
    assert "Workflow 'nat_workflow_that_is_not_installed' does not exist" in result.output


def test_delete_missing_workflow_fails():
    result = CliRunner().invoke(delete_command, ["--yes", "nat_workflow_that_is_not_installed"])

    assert result.exit_code == 1
    assert "Workflow 'nat_workflow_that_is_not_installed' does not exist" in result.output


def test_delete_workflow_with_failed_uninstall_fails(tmp_path):
    uninstall_error = subprocess.CalledProcessError(2, ["uv", "pip", "uninstall"], stderr="uninstall failed")

    with patch("nat.cli.commands.workflow.workflow_commands.get_workflow_path_from_name", return_value=tmp_path), \
            patch("nat.cli.commands.workflow.workflow_commands.subprocess.run", side_effect=uninstall_error):
        result = CliRunner().invoke(delete_command, ["--yes", "nat_workflow_that_is_not_installed"])

    assert result.exit_code == 1
    assert "An error occurred while deleting the workflow" in result.output
    assert "uninstall failed" in result.output


def test_delete_workflow_cancelled_succeeds():
    result = CliRunner().invoke(delete_command, ["test_workflow"], input="n\n")

    assert result.exit_code == 0
    assert "Workflow deletion cancelled." in result.output


@pytest.mark.parametrize("command,args", [(reinstall_command, [" "]), (delete_command, [" "]),
                                          (delete_command, ["--yes", " "])])
def test_reinstall_and_delete_reject_empty_workflow_name(command, args):
    """reinstall and delete reject an empty name before any lookup or prompt, like create."""
    result = CliRunner().invoke(command, args)

    assert result.exit_code == 2
    assert "Workflow name cannot be empty" in result.output
