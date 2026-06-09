# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
