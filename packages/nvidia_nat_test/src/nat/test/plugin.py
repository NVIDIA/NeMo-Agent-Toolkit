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

import os

import pytest


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

    parser.addoption(
        "--fail_missing",
        action="store_true",
        dest="fail_missing",
        help=("Tests requiring unmet dependencies are normally skipped. "
              "Setting this flag will instead cause them to be reported as a failure"),
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
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins

    # Ensure that all components which need to be registered as part of an import are done so. This is necessary
    # because imports will not be reloaded between tests, so we need to ensure that all components are registered
    # before any tests are run.
    discover_and_register_plugins(PluginTypes.ALL)

    # Also import the nat.test.register module to register test-only components


@pytest.fixture(name="module_registry", scope="module", autouse=True)
def module_registry_fixture():
    """
    Resets and returns the global type registry for testing

    This gets automatically used at the module level to ensure no state is leaked between modules
    """
    from nat.cli.type_registry import GlobalTypeRegistry

    with GlobalTypeRegistry.push() as registry:
        yield registry


@pytest.fixture(name="registry", scope="function", autouse=True)
def function_registry_fixture():
    """
    Resets and returns the global type registry for testing

    This gets automatically used at the function level to ensure no state is leaked between functions
    """
    from nat.cli.type_registry import GlobalTypeRegistry

    with GlobalTypeRegistry.push() as registry:
        yield registry


@pytest.fixture(scope="session", name="fail_missing")
def fail_missing_fixture(pytestconfig: pytest.Config) -> bool:
    """
    Returns the value of the `fail_missing` flag, when false tests requiring unmet dependencies will be skipped, when
    True they will fail.
    """
    yield pytestconfig.getoption("fail_missing")


def require_env_variables(varnames: list[str], reason: str, fail_missing: bool = False) -> dict[str, str]:
    """
    Checks if the given environment variable is set, and returns its value if it is. If the variable is not set, and
    `fail_missing` is False the test will ve skipped, otherwise a `RuntimeError` will be raised.
    """
    env_variables = {}
    try:
        for varname in varnames:
            env_variables[varname] = os.environ[varname]
    except KeyError as e:
        if fail_missing:
            raise RuntimeError(reason) from e

        pytest.skip(reason=reason)

    return env_variables


@pytest.fixture(name="openai_api_key", scope='session')
def openai_api_key_fixture(fail_missing: bool):
    """
    Use for integration tests that require an Openai API key.
    """
    yield require_env_variables(
        varnames=["OPENAI_API_KEY"],
        reason="openai integration tests require the `OPENAI_API_KEY` environment variable to be defined.",
        fail_missing=fail_missing)


@pytest.fixture(name="nvidia_api_key", scope='session')
def nvidia_api_key_fixture(fail_missing: bool):
    """
    Use for integration tests that require an Nvidia API key.
    """
    yield require_env_variables(
        varnames=["NVIDIA_API_KEY"],
        reason="Nvidia integration tests require the `NVIDIA_API_KEY` environment variable to be defined.",
        fail_missing=fail_missing)


@pytest.fixture(name="aws_keys", scope='session')
def aws_keys_fixture(fail_missing: bool):
    """
    Use for integration tests that require AWS credentials.
    """

    yield require_env_variables(
        varnames=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        reason=
        "AWS integration tests require the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables to be "
        "defined.",
        fail_missing=fail_missing)


@pytest.fixture(name="azure_openai_keys", scope='session')
def azure_openai_keys_fixture(fail_missing: bool):
    """
    Use for integration tests that require Azure OpenAI credentials.
    """
    yield require_env_variables(
        varnames=["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
        reason="Azure integration tests require the `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` environment "
        "variable to be defined.",
        fail_missing=fail_missing)


@pytest.fixture(name="restore_environ")
def restore_environ_fixture():
    orig_vars = os.environ.copy()
    yield os.environ

    for key, value in orig_vars.items():
        os.environ[key] = value

    # Delete any new environment variables
    # Iterating over a copy of the keys as we will potentially be deleting keys in the loop
    for key in list(os.environ.keys()):
        if key not in orig_vars:
            del (os.environ[key])
