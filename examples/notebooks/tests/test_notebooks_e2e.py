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

import pytest

_ALL_WORKFLOWS = [
    "getting_started",
    "first_agent_attempt",
    "second_agent_attempt",
    "third_agent_attempt",
    "retail_sales_agent",
    "tmp_workflow"
]

_OTHER_FILES = [
    "nat_embedded.py",
]


@pytest.fixture(name="notebooks_dir", scope='session')
def notebooks_dir_fixture(examples_dir: Path) -> Path:
    return examples_dir / "notebooks"


def _is_installed(package_name: str) -> bool:
    """Check if a package is installed."""
    try:
        subprocess.run(
            ["uv", "pip", "show", "-q", package_name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _delete_all_workflows():
    for workflow in _ALL_WORKFLOWS:
        if _is_installed(workflow):
            cmd = ["nat", "workflow", "delete", "--yes", workflow]
            subprocess.run(cmd, check=False)


def _delete_other_files(notebooks_dir: Path):
    for file in _OTHER_FILES:
        file_path = notebooks_dir / file
        if file_path.exists():
            file_path.unlink()


def _cleanup_all(notebooks_dir: Path):
    _delete_all_workflows()
    _delete_other_files(notebooks_dir)


@pytest.fixture(name="workflow_cleanups", scope='function', autouse=True)
def workflow_cleanups_fixture(notebooks_dir: Path):
    _cleanup_all(notebooks_dir)
    yield
    _cleanup_all(notebooks_dir)


def _run_notebook(notebook_path: Path, expected_packages: list[str], timeout_seconds: int = 120):
    """Run a Jupyter notebook and check for errors."""
    cmd = [
        "jupyter",
        "execute",
        f"--timeout={timeout_seconds}",
        str(notebook_path.absolute()),
    ]

    # Ideally if the notebook times out we want jupyter to catch it and exit gracefully with the most informative error
    # possible. However in the potential situation where jupyter itself hangs, we add a 30s buffer to the timeout here.
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout_seconds + 30)
    assert result.returncode == 0, f"Notebook execution failed:\n{result.stderr}"

    for package in expected_packages:
        assert _is_installed(package), f"Expected package '{package}' is not installed."


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.parametrize(
    "notebook_file_name, expected_packages",
    [
        ("1_getting_started_with_nat.ipynb", ["getting_started"]),
        ("3_adding_tools_to_agents.ipynb", ["retail_sales_agent"]),
        ("4_multi_agent_orchestration.ipynb", ["retail_sales_agent"]),
        ("5_observability_evaluation_and_profiling.ipynb", ["retail_sales_agent"]),
        pytest.param("6_optimize_model_selection.ipynb", ["tmp_workflow"],
                     marks=pytest.mark.skip(reason="Notebook takes over an hour to run completely.")),
    ],
    ids=[f"notebook_{i}" for i in (1, 3, 4, 5, 6)])
def test_notebooks(notebooks_dir: Path, notebook_file_name: str, expected_packages: list[str]):
    _run_notebook(notebooks_dir / notebook_file_name, expected_packages=expected_packages)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "tavily_api_key")
def test_2_bringing_your_own_agent(notebooks_dir: Path):
    # This test is the same as the others but requires a Tavily API key to run
    _run_notebook(notebooks_dir / "2_bringing_your_own_agent.ipynb",
                  expected_packages=["first_agent_attempt", "second_agent_attempt", "third_agent_attempt"],
                  timeout_seconds=180)
