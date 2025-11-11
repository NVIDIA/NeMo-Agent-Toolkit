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

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.config import Config
from nat.data_models.config import HashableBaseModel
from nat.data_models.function import FunctionBaseConfig
from nat.utils.io.yaml_tools import _interpolate_variables
from nat.utils.io.yaml_tools import deep_merge
from nat.utils.io.yaml_tools import yaml_dump
from nat.utils.io.yaml_tools import yaml_dumps
from nat.utils.io.yaml_tools import yaml_load
from nat.utils.io.yaml_tools import yaml_loads


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
        with open(temp_file_path, encoding='utf-8') as f:
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
    # Convert the YAML data to an Config object
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
        # Convert the YAML data to an Config object
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


def test_deep_merge():
    # Test basic merge
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    result = deep_merge(base, override)
    assert result == {"a": 1, "b": 3, "c": 4}

    # Test nested merge
    base = {"a": 1, "b": {"c": 2, "d": 3}, "e": 5}
    override = {"b": {"d": 4}, "f": 6}
    result = deep_merge(base, override)
    assert result == {"a": 1, "b": {"c": 2, "d": 4}, "e": 5, "f": 6}

    # Test deep nested merge
    base = {"level1": {"level2": {"level3": {"value": 1, "other": 2}}}}
    override = {"level1": {"level2": {"level3": {"value": 999}}}}
    result = deep_merge(base, override)
    assert result["level1"]["level2"]["level3"]["value"] == 999
    assert result["level1"]["level2"]["level3"]["other"] == 2

    # Test replacing non-dict with dict
    base = {"a": "string"}
    override = {"a": {"b": "dict"}}
    result = deep_merge(base, override)
    assert result == {"a": {"b": "dict"}}

    # Test empty override
    base = {"a": 1, "b": 2}
    override = {}
    result = deep_merge(base, override)
    assert result == {"a": 1, "b": 2}


def test_yaml_load_with_base_inheritance():
    # Create a base config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as base_file:
        base_file.write("""
        llms:
          nim_llm:
            model_name: meta/llama-3.1-70b-instruct
            temperature: 0.0
            max_tokens: 1024
        workflow:
          _type: react_agent
          verbose: true
        """)
        base_file_path = base_file.name

    # Create a variant config that inherits from base
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as variant_file:
        variant_file.write(f"""
        base: {os.path.basename(base_file_path)}
        llms:
          nim_llm:
            temperature: 0.9
        """)
        variant_file_path = variant_file.name

    try:
        # Load variant config with inheritance
        config = yaml_load(variant_file_path)
        # Check overridden value
        assert config["llms"]["nim_llm"]["temperature"] == 0.9
        # Check inherited values
        assert config["llms"]["nim_llm"]["model_name"] == "meta/llama-3.1-70b-instruct"
        assert config["llms"]["nim_llm"]["max_tokens"] == 1024
        assert config["workflow"]["_type"] == "react_agent"
        assert config["workflow"]["verbose"] is True
        # Verify 'base' key is removed from final config
        assert "base" not in config
    finally:
        os.unlink(base_file_path)
        os.unlink(variant_file_path)


def test_yaml_load_without_base():
    # Test that yaml_load works normally when no base key is present
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_file.write("""
        llms:
          nim_llm:
            temperature: 0.5
        workflow:
          verbose: false
        """)
        temp_file_path = temp_file.name

    try:
        config = yaml_load(temp_file_path)
        assert config["llms"]["nim_llm"]["temperature"] == 0.5
        assert config["workflow"]["verbose"] is False
    finally:
        os.unlink(temp_file_path)


def test_yaml_load_chained_inheritance():
    # Test yaml_load with multiple levels of inheritance
    # Create base config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as base_file:
        base_file.write("""
        level1: base
        level2: base
        level3: base
        """)
        base_file_path = base_file.name

    # Create intermediate config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as middle_file:
        middle_file.write(f"""
        base: {os.path.basename(base_file_path)}
        level2: middle
        """)
        middle_file_path = middle_file.name

    # Create final config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as final_file:
        final_file.write(f"""
        base: {os.path.basename(middle_file_path)}
        level3: final
        """)
        final_file_path = final_file.name

    try:
        config = yaml_load(final_file_path)
        assert config["level1"] == "base"  # From base
        assert config["level2"] == "middle"  # From intermediate
        assert config["level3"] == "final"  # From final
    finally:
        os.unlink(base_file_path)
        os.unlink(middle_file_path)
        os.unlink(final_file_path)


def test_yaml_load_base_type_validation():
    # Test that base key must be a string
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
        config_file.write("""
        base: 123
        key: value
        """)
        config_file_path = config_file.name
    try:
        with pytest.raises(TypeError, match="Configuration 'base' key must be a string"):
            yaml_load(config_file_path)
    finally:
        os.unlink(config_file_path)


def test_yaml_load_base_file_not_found():
    # Test that missing base file raises FileNotFoundError
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
        config_file.write("""
        base: nonexistent_file.yml
        key: value
        """)
        config_file_path = config_file.name
    try:
        with pytest.raises(FileNotFoundError, match="Base configuration file not found"):
            yaml_load(config_file_path)
    finally:
        os.unlink(config_file_path)


def test_yaml_load_circular_dependency():
    # Test that circular dependencies are detected
    # Create config A that inherits from B
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as file_a:
        file_a_path = file_a.name
    # Create config B that inherits from A
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as file_b:
        file_b_path = file_b.name
    try:
        # Write config A (inherits from B)
        with open(file_a_path, 'w') as f:
            f.write(f"""
            base: {os.path.basename(file_b_path)}
            key_a: value_a
            """)
        # Write config B (inherits from A - creates cycle)
        with open(file_b_path, 'w') as f:
            f.write(f"""
            base: {os.path.basename(file_a_path)}
            key_b: value_b
            """)
        with pytest.raises(ValueError, match="Circular dependency detected"):
            yaml_load(file_a_path)
    finally:
        os.unlink(file_a_path)
        os.unlink(file_b_path)
