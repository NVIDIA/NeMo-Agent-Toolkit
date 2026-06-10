# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

PACKAGE_ROOT = pathlib.Path(__file__).parents[1]


def _load_pyproject() -> dict:
    with (PACKAGE_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)


def test_shim_has_no_python_packages_or_nat_entry_points():
    pyproject = _load_pyproject()

    assert pyproject["tool"]["setuptools"]["packages"] == []
    assert "entry-points" not in pyproject["project"]
    assert list((PACKAGE_ROOT / "src" / "nat" / "plugins" / "redis").glob("*.py")) == []


def test_shim_depends_on_external_redis_plugin():
    pyproject = _load_pyproject()

    dependencies = pyproject["project"]["dependencies"]
    assert dependencies == ["nemo-agent-toolkit-redis>=0.1.0,<2.0.0"]
