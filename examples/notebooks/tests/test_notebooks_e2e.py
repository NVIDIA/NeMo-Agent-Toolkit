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


@pytest.fixture(name="notebooks_dir", scope='session')
def notebooks_dir_fixture(examples_dir: Path) -> Path:
    return examples_dir / "notebooks"


def _run_notebook(notebook_path: Path, timeout_seconds: int = 120):
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


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.parametrize(
    "notebook_file_name",
    [
        pytest.param("1_getting_started_with_nat.ipynb",
                     marks=pytest.mark.xfail(reason='failing in the nat_embedded.py cell')),
        "3_adding_tools_to_agents.ipynb",
        "4_multi_agent_orchestration.ipynb",
        "5_observability_evaluation_and_profiling.ipynb",
        pytest.param(
            "6_optimize_model_selection.ipynb",
            marks=pytest.mark.skip(reason="failing after about 13 minutes with a key-error in cell 1.2 on line:\n"
                                   "\tbest_trial = trials_df.loc[trials_df['values_0'].idxmax()]")),
    ])
def test_notebooks(notebooks_dir: Path, notebook_file_name: str):
    _run_notebook(notebooks_dir / notebook_file_name)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "tavily_api_key")
def test_2_bringing_your_own_agent(notebooks_dir: Path):
    # This notebook is the same as the others but requires a Tavily API key to run
    _run_notebook(notebooks_dir / "2_bringing_your_own_agent.ipynb")
