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

import pytest

from aiq.runtime.loader import PluginTypes
from aiq.runtime.loader import discover_and_register_plugins


def pytest_addoption(parser: pytest.Parser):
    """
    Adds command line options for running specfic tests that are disabled by default
    """
    parser.addoption(
        "--run_e2e",
        action="store_true",
        dest="run_e2e",
        help="Run end to end tests that would otherwise be skipped",
    )

    parser.addoption(
        "--run_integration",
        action="store_true",
        dest="run_integration",
        help=("Run integrations tests that would otherwise be skipped. "
              "This will call out to external services instead of using mocks"),
    )

    parser.addoption(
        "--run_slow",
        action="store_true",
        dest="run_slow",
        help="Run end to end tests that would otherwise be skipped",
    )


def pytest_runtest_setup(item):
    if (not item.config.getoption("--run_e2e")):
        if (item.get_closest_marker("e2e") is not None):
            pytest.skip("Skipping end to end tests by default. Use --run_e2e to enable")

    if (not item.config.getoption("--run_integration")):
        if (item.get_closest_marker("integration") is not None):
            pytest.skip("Skipping integration tests by default. Use --run_integration to enable")

    if (not item.config.getoption("--run_slow")):
        if (item.get_closest_marker("slow") is not None):
            pytest.skip("Skipping slow tests by default. Use --run_slow to enable")


@pytest.fixture(name="register_components", scope="session", autouse=True)
def register_components_fixture():

    # Ensure that all components which need to be registered as part of an import are done so. This is necessary
    # because imports will not be reloaded between tests, so we need to ensure that all components are registered
    # before any tests are run.
    discover_and_register_plugins(PluginTypes.ALL)

    # Also import the aiq.test.register module to register test-only components
    import aiq.test.register  # pylint: disable=unused-import # noqa: F401


@pytest.fixture(name="module_registry", scope="module", autouse=True)
def module_registry_fixture():
    """
    Resets and returns the global type registry for testing

    This gets automatically used at the module level to ensure no state is leaked between modules
    """
    from aiq.cli.type_registry import GlobalTypeRegistry

    with GlobalTypeRegistry.push() as registry:
        yield registry


@pytest.fixture(name="registry", scope="function", autouse=True)
def function_registry_fixture():
    """
    Resets and returns the global type registry for testing

    This gets automatically used at the function level to ensure no state is leaked between functions
    """
    from aiq.cli.type_registry import GlobalTypeRegistry

    with GlobalTypeRegistry.push() as registry:
        yield registry
