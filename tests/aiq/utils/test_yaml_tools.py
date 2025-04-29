import os
import tempfile
from io import StringIO

import pytest

from aiq.utils.io.yaml_tools import _interpolate_variables
from aiq.utils.io.yaml_tools import _process_config
from aiq.utils.io.yaml_tools import yaml_dump
from aiq.utils.io.yaml_tools import yaml_dumps
from aiq.utils.io.yaml_tools import yaml_load
from aiq.utils.io.yaml_tools import yaml_loads

TEST_VAR = "TEST_VAR"
NESTED_VAR = "NESTED_VAR"


def test_interpolate_variables():
    # Test basic variable interpolation
    try:
        os.environ[TEST_VAR] = "test_value"
        assert _interpolate_variables("${TEST_VAR}") == "test_value"

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
    finally:
        if (TEST_VAR in os.environ):
            os.environ.pop(TEST_VAR)


def test_process_config_with_basic_types():
    # Test with unsupported type
    with pytest.raises(ValueError, match="Unsupported type"):
        _process_config(complex(1, 2))  # type: ignore

    # Test with boolean values
    assert _process_config(True) is True
    assert _process_config(False) is False

    # Test with None value
    assert _process_config(None) is None

    # Test with integer
    assert _process_config(42) == 42

    # Test with float
    assert _process_config(3.14) == 3.14

    # Test with string
    assert _process_config("hello") == "hello"

    # Test with empty containers
    assert _process_config({}) == {}
    assert _process_config([]) == []


def test_process_config_with_nested_containers():
    # Test with nested containers containing all data types
    nested_config = {
        "string":
            "plain_text",
        "string_with_var":
            "${TEST_VAR}",
        "integer":
            42,
        "float":
            3.14,
        "boolean_true":
            True,
        "boolean_false":
            False,
        "none_value":
            None,
        "nested_dict": {
            "inner_string": "nested_text",
            "inner_var": "${NESTED_VAR:-default}",
            "inner_int": 100,
            "inner_float": 2.718,
            "inner_bool": False,
            "inner_none": None
        },
        "nested_list": [
            "list_string", "${TEST_VAR}", 123, 4.56, True, None, ["deeply_nested", "${NESTED_VAR:-fallback}", 999]
        ]
    }

    try:
        os.environ["TEST_VAR"] = "test_value"
        os.environ["NESTED_VAR"] = "nested_value"

        processed_nested = _process_config(nested_config)
        assert isinstance(processed_nested, dict)
        nested_dict = processed_nested["nested_dict"]
        assert isinstance(nested_dict, dict)

        # Verify string values
        assert processed_nested["string"] == "plain_text"
        assert processed_nested["string_with_var"] == "test_value"

        # Verify numeric values
        assert processed_nested["integer"] == 42
        assert processed_nested["float"] == 3.14

        # Verify boolean values
        assert processed_nested["boolean_true"] is True
        assert processed_nested["boolean_false"] is False

        # Verify None value
        assert processed_nested["none_value"] is None

        # Verify nested dictionary
        assert nested_dict["inner_string"] == "nested_text"
        assert nested_dict["inner_var"] == "nested_value"
        assert nested_dict["inner_int"] == 100
        assert nested_dict["inner_float"] == 2.718
        assert nested_dict["inner_bool"] is False
        assert nested_dict["inner_none"] is None

        # Verify nested lists
        nested_list = processed_nested["nested_list"]
        assert isinstance(nested_list, list)
        assert nested_list[0] == "list_string"
        assert nested_list[1] == "test_value"
        assert nested_list[2] == 123
        assert nested_list[3] == 4.56
        assert nested_list[4] is True
        assert nested_list[5] is None

        deep_list = nested_list[6]
        assert isinstance(deep_list, list)
        assert deep_list[0] == "deeply_nested"
        assert deep_list[1] == "nested_value"
        assert deep_list[2] == 999

    finally:
        if (TEST_VAR in os.environ):
            os.environ.pop(TEST_VAR)
        if (NESTED_VAR in os.environ):
            os.environ.pop(NESTED_VAR)


def test_yaml_load():
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
        os.environ["TEST_VAR"] = "test_value"
        os.environ["NESTED_VAR"] = "nested_value"

        config = yaml_load(temp_file_path)
        assert config["key1"] == "test_value"
        assert config["key2"] == "static_value"
        assert config["key3"]["nested"] == "nested_value"
    finally:
        os.unlink(temp_file_path)
        if (TEST_VAR in os.environ):
            os.environ.pop(TEST_VAR)
        if (NESTED_VAR in os.environ):
            os.environ.pop(NESTED_VAR)


def test_yaml_loads():
    yaml_str = """
    key1: ${TEST_VAR}
    key2: static_value
    key3:
      nested: ${NESTED_VAR:-default}
    """

    try:
        os.environ["TEST_VAR"] = "test_value"
        os.environ["NESTED_VAR"] = "nested_value"

        config = yaml_loads(yaml_str)
        assert config["key1"] == "test_value"
        assert config["key2"] == "static_value"
        assert config["key3"]["nested"] == "nested_value"
    finally:
        if (TEST_VAR in os.environ):
            os.environ.pop(TEST_VAR)
        if (NESTED_VAR in os.environ):
            os.environ.pop(NESTED_VAR)


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
