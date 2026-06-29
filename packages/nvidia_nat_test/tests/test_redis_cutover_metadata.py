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

import importlib.util
import pathlib
import subprocess
import tomllib

REPO_ROOT = pathlib.Path(__file__).parents[3]
REDIS_PACKAGE_ROOT = REPO_ROOT / "packages" / "nvidia_nat_redis"
EXCLUDED_PROJECTS = {
    "packages/nvidia_nat_redis",
    "examples/memory/redis",
    "examples/object_store/user_report",
}


def _load_pyproject(project_root: pathlib.Path) -> dict:
    """Load a project configuration as TOML."""
    with (project_root / "pyproject.toml").open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)


def _load_run_tests_module():
    """Load the CI test runner without making the CI directory a package."""
    module_path = REPO_ROOT / "ci" / "scripts" / "run_tests.py"
    spec = importlib.util.spec_from_file_location("nat_ci_run_tests", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_redis_third_party_projects_are_excluded_from_test_discovery():
    """Verify that CI test discovery cannot install Redis-maintained code."""
    run_tests = _load_run_tests_module()
    discovered_projects = {project.relative_to(REPO_ROOT).as_posix() for project in run_tests.discover_projects()}

    assert EXCLUDED_PROJECTS.isdisjoint(discovered_projects)


def test_redis_third_party_projects_are_excluded_from_wheel_execution():
    """Verify that shell CI excludes Redis examples and shim wheel installation."""
    shell_check = """
source ci/scripts/common.sh
[[ ! " ${NAT_EXAMPLES[*]} " =~ " ./examples/memory/redis " ]]
[[ ! " ${NAT_EXAMPLES[*]} " =~ " ./examples/object_store/user_report " ]]
[[ " ${NAT_PACKAGES[*]} " =~ " ./packages/nvidia_nat_redis " ]]
is_third_party_dependency_wheel /tmp/nvidia_nat_redis-1.9.0-py3-none-any.whl
"""
    subprocess.run(["bash", "-c", shell_check], cwd=REPO_ROOT, check=True)
