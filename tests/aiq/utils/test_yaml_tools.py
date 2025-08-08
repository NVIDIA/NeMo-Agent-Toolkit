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
import tempfile
from io import StringIO

import pytest

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.config import Config
from aiq.data_models.config import HashableBaseModel
from aiq.data_models.function import FunctionBaseConfig
from aiq.utils.io.yaml_tools import _interpolate_variables
from aiq.utils.io.yaml_tools import yaml_dump
from aiq.utils.io.yaml_tools import yaml_dumps
from aiq.utils.io.yaml_tools import yaml_load
from aiq.utils.io.yaml_tools import yaml_loads


@pytest.fixture(name="env_vars", scope="function", autouse=True)
def fixture_env_vars():
    """Fixture to set and clean up environment variables for tests."""

    test_vars = {
        "TEST_VAR": "test_value",
        "LIST_VAR": "list_value",
        "NESTED_VAR": "nested_value",
        "BOOL_VAR": "true",
        "FLOAT_VAR": "0.0",
        "INT_VAR": "42",
        "FN_LIST_VAR": "[fn0, fn1, fn2]"
    }

    # Store original environment variables state
    original_env = {}

    # Set test environment variables and store original values
    for var, value in test_vars.items():
        if var in os.environ:
            original_env[var] = os.environ[var]
        os.environ[var] = value

    # Yield the test variables dctionary to the test
    yield test_vars

    # Clean up: restore original environment
    for var in test_vars:
        if var in original_env:
            os.environ[var] = original_env[var]
        else:
            del os.environ[var]


class CustomConfig(FunctionBaseConfig, name="my_test_fn"):
    string_input: str
    int_input: int
    float_input: float
    bool_input: bool
    none_input: None
    list_input: list[str]
    dict_input: dict[str, str]
    fn_list_input: list[str]


@pytest.fixture(scope="module", autouse=True)
async def fixture_register_test_fn():

    @register_function(config_type=CustomConfig)
    async def register(config: CustomConfig, b: Builder):

        async def _inner(some_input: str) -> str:
            return some_input

        yield FunctionInfo.from_fn(_inner)


def test_interpolate_variables(env_vars: dict):
    # Test basic variable interpolation
    assert _interpolate_variables("${TEST_VAR}") == env_vars["TEST_VAR"]

    # Test with default value
    assert _interpolate_variables("${NONEXISTENT_VAR:-default}") == "default"

    # Test with empty default value
    assert _interpolate_variables("${NONEXISTENT_VAR:-}") == ""

    # Test with no default value
    assert _interpolate_variables("${NONEXISTENT_VAR}") == ""

    # Test with non-string input
    assert _interpolate_variables(123) == 123
    assert _interpolate_variables(0.123) == 0.123
    assert _interpolate_variables(None) is None


def test_yaml_load(env_vars: dict):
    # Create a temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_file.write("""
        key1: ${TEST_VAR}
        key2: static_value
        key3:
          nested: ${NESTED_VAR:-default}
        """)
        temp_file_path = temp_file.name

    try:
        config = yaml_load(temp_file_path)
        assert config["key1"] == env_vars["TEST_VAR"]
        assert config["key2"] == "static_value"
        assert config["key3"]["nested"] == env_vars["NESTED_VAR"]
    finally:
        os.unlink(temp_file_path)


def test_yaml_loads(env_vars: dict):
    yaml_str = """
    key1: ${TEST_VAR}
    key2: static_value
    key3:
      nested: ${NESTED_VAR:-default}
    """

    config: dict = yaml_loads(yaml_str)
    assert config["key1"] == env_vars["TEST_VAR"]
    assert config["key2"] == "static_value"
    assert config["key3"]["nested"] == env_vars["NESTED_VAR"]  # type: ignore


def test_yaml_dump():
    config = {"key1": "value1", "key2": "value2", "key3": {"nested": "value3"}}

    # Test dumping to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        yaml_dump(config, temp_file)  # type: ignore
        temp_file_path = temp_file.name

    try:
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "key1: value1" in content
            assert "key2: value2" in content
            assert "nested: value3" in content
    finally:
        os.unlink(temp_file_path)

    # Test dumping to StringIO
    string_io = StringIO()
    yaml_dump(config, string_io)
    content = string_io.getvalue()
    assert "key1: value1" in content
    assert "key2: value2" in content
    assert "nested: value3" in content


def test_yaml_dumps():
    config = {"key1": "value1", "key2": "value2", "key3": {"nested": "value3"}}

    yaml_str = yaml_dumps(config)
    assert "key1: value1" in yaml_str
    assert "key2: value2" in yaml_str
    assert "nested: value3" in yaml_str


def test_yaml_loads_with_function(env_vars: dict):
    yaml_str = """
    workflow:
      _type: my_test_fn
      string_input: ${TEST_VAR}
      int_input: ${INT_VAR}
      float_input: ${FLOAT_VAR}
      bool_input: ${BOOL_VAR}
      none_input: null
      list_input:
        - a
        - ${LIST_VAR}
        - c
      dict_input:
        key1: value1
        key2: ${NESTED_VAR}
      fn_list_input: ${FN_LIST_VAR}
    """

    # Test loading with function
    config_data: dict = yaml_loads(yaml_str)
    # Convert the YAML data to an AIQConfig object
    workflow_config: HashableBaseModel = Config(**config_data)

    assert workflow_config.workflow.type == "my_test_fn"
    assert workflow_config.workflow.string_input == env_vars["TEST_VAR"]  # type: ignore
    assert workflow_config.workflow.int_input == int(env_vars["INT_VAR"])  # type: ignore
    assert workflow_config.workflow.float_input == float(env_vars["FLOAT_VAR"])  # type: ignore
    assert workflow_config.workflow.bool_input is bool(env_vars["BOOL_VAR"])  # type: ignore
    assert workflow_config.workflow.none_input is None  # type: ignore
    assert workflow_config.workflow.list_input == ["a", env_vars["LIST_VAR"], "c"]  # type: ignore
    assert workflow_config.workflow.dict_input == {"key1": "value1", "key2": env_vars["NESTED_VAR"]}  # type: ignore
    assert workflow_config.workflow.fn_list_input == ["fn0", "fn1", "fn2"]  # type: ignore


def test_yaml_load_with_function(env_vars: dict):
    # Create a temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_file.write("""
        workflow:
          _type: my_test_fn
          string_input: ${TEST_VAR}
          int_input: ${INT_VAR}
          float_input: ${FLOAT_VAR}
          bool_input: ${BOOL_VAR}
          none_input: null
          list_input:
            - a
            - ${LIST_VAR}
            - c
          dict_input:
            key1: value1
            key2: ${NESTED_VAR}
          fn_list_input: ${FN_LIST_VAR}
        """)
        temp_file_path = temp_file.name

    try:
        # Test loading with function
        config_data: dict = yaml_load(temp_file_path)
        # Convert the YAML data to an AIQConfig object
        workflow_config: HashableBaseModel = Config(**config_data)

        workflow_config.workflow.type = "my_test_fn"
        assert workflow_config.workflow.type == "my_test_fn"
        assert workflow_config.workflow.string_input == env_vars["TEST_VAR"]  # type: ignore
        assert workflow_config.workflow.int_input == int(env_vars["INT_VAR"])  # type: ignore
        assert workflow_config.workflow.float_input == float(env_vars["FLOAT_VAR"])  # type: ignore
        assert workflow_config.workflow.bool_input is bool(env_vars["BOOL_VAR"])  # type: ignore
        assert workflow_config.workflow.none_input is None  # type: ignore
        assert workflow_config.workflow.list_input == ["a", env_vars["LIST_VAR"], "c"]  # type: ignore
        assert workflow_config.workflow.dict_input == {"key1": "value1", "key2": env_vars["NESTED_VAR"]}  # type: ignore
        assert workflow_config.workflow.fn_list_input == ["fn0", "fn1", "fn2"]  # type: ignore

    finally:
        os.unlink(temp_file_path)


def test_yaml_loads_with_invalid_yaml():
    # Test with invalid YAML syntax
    invalid_yaml = """
    workflow:
      - this is not valid yaml
        indentation is wrong
      key without value
    """

    with pytest.raises(ValueError, match="Error loading YAML"):
        yaml_loads(invalid_yaml)

    # Test with completely malformed content
    malformed_yaml = "{"  # Unclosed bracket
    with pytest.raises(ValueError, match="Error loading YAML"):
        yaml_loads(malformed_yaml)
