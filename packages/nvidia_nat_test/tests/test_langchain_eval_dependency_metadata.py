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

import pathlib
import tomllib

REPO_ROOT = pathlib.Path(__file__).parents[3]
LANGCHAIN_PACKAGE_ROOT = REPO_ROOT / "packages" / "nvidia_nat_langchain"


def _load_pyproject(project_root: pathlib.Path) -> dict:
    with (project_root / "pyproject.toml").open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)


def test_eval_and_langchain_extras_do_not_install_conflicting_aws_dependencies():
    """Keep the generic LangChain extra compatible with the full eval runtime."""
    root_optional_dependencies = _load_pyproject(REPO_ROOT)["tool"]["setuptools_dynamic_dependencies"][
        "optional-dependencies"
    ]
    langchain_optional_dependencies = _load_pyproject(LANGCHAIN_PACKAGE_ROOT)["tool"][
        "setuptools_dynamic_dependencies"
    ]["optional-dependencies"]

    assert root_optional_dependencies["langchain"] == ["nvidia-nat-langchain[common] == {version}"]
    assert "aws" not in langchain_optional_dependencies["common"][0]
    assert langchain_optional_dependencies["all"] == ["nvidia-nat-langchain[aws,common]"]


def test_most_extra_uses_eval_compatible_langchain_dependencies():
    """Keep the aggregate install from reintroducing the same botocore conflict."""
    root_optional_dependencies = _load_pyproject(REPO_ROOT)["tool"]["setuptools_dynamic_dependencies"][
        "optional-dependencies"
    ]

    assert "nvidia-nat-eval[full] == {version}" in root_optional_dependencies["most"]
    assert "nvidia-nat-langchain[common] == {version}" in root_optional_dependencies["most"]
    assert "nvidia-nat-langchain[all] == {version}" not in root_optional_dependencies["most"]
