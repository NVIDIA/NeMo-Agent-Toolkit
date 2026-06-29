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
REDIS_PACKAGE_ROOT = REPO_ROOT / "packages" / "nvidia_nat_redis"


def _load_pyproject(project_root: pathlib.Path) -> dict:
    """Load a project configuration as TOML."""
    with (project_root / "pyproject.toml").open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)


def test_redis_extra_installs_compatibility_package():
    """Verify that the Redis extra preserves the historical package name."""
    pyproject = _load_pyproject(REPO_ROOT)

    optional_dependencies = pyproject["tool"]["setuptools_dynamic_dependencies"]["optional-dependencies"]
    assert optional_dependencies["redis"] == ["nvidia-nat-redis == {version}"]


def test_redis_compatibility_package_has_no_plugin_code():
    """Verify that the historical distribution is a no-code compatibility package."""
    pyproject = _load_pyproject(REDIS_PACKAGE_ROOT)

    assert pyproject["tool"]["setuptools"]["packages"] == []
    assert "entry-points" not in pyproject["project"]
    assert not list((REDIS_PACKAGE_ROOT / "src" / "nat" / "plugins" / "redis").glob("*.py"))


def test_redis_compatibility_package_depends_on_public_plugin_release():
    """Verify that the shim requires the first Redis public-plugin API release."""
    pyproject = _load_pyproject(REDIS_PACKAGE_ROOT)

    assert pyproject["project"]["dependencies"] == ["nemo-agent-toolkit-redis>=0.2.0,<2.0.0"]
