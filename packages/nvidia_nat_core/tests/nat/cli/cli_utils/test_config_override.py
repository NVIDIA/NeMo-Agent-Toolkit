# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import click
import pytest
from pydantic import ValidationError

from nat.cli.cli_utils import config_override
from nat.data_models.function import FunctionBaseConfig
from nat.llm.nim_llm import NIMModelConfig


@pytest.fixture(name="base_config")
def fixture_base_config() -> dict:
    return {"a": {"b": 1, "c": 2}, "d": 3, "bool_val": True}


def test_layered_config_set_override(base_config: dict):
    layered_config = config_override.LayeredConfig(base_config)

    # Override a value that already exists
    layered_config.set_override("a.b", '10')

    # Override a value that doesn't exist
    layered_config.set_override("a.e", '20')

    # override a nested value
    layered_config.set_override("f.g", '30')

    layered_config.set_override("bool_val", '\tfALse ')

    assert layered_config.get_effective_config() == {
        "a": {
            "b": 10, "c": 2, "e": '20'
        }, "d": 3, "f": {
            "g": '30'
        }, "bool_val": False
    }


def test_layered_config_set_override_error(base_config: dict):
    layered_config = config_override.LayeredConfig(base_config)

    # Attempt to set an override with an invalid path
    with pytest.raises(click.BadParameter, match="Cannot navigate through non-dictionary value at 'a.b'"):
        layered_config.set_override("a.b.c", '10')

    # Attempt to set an override a boolean value with an invalid string
    with pytest.raises(click.BadParameter, match="Boolean value must be 'true' or 'false', got 'not_a_bool'"):
        layered_config.set_override("bool_val", 'not_a_bool')

    # Attempt to set a value with a type that doesn't match the original
    with pytest.raises(click.BadParameter, match=r"Type mismatch for 'a\.b'"):
        layered_config.set_override("a.b", 'not_a_number')


def test_layered_config_constructor_error(base_config: dict):
    # Attempt to set an override with an invalid base config
    with pytest.raises(ValueError, match="Base config must be a dictionary"):
        config_override.LayeredConfig("invalid_base_config")


def test_config_casting():
    """
    Test to verify that pydantic's casting works as expected in situations where LayeredConfig
    is unable to determine the type of the value being set.
    """

    class TestConfig(FunctionBaseConfig, name="TestConfig"):
        a: bool
        b: int
        c: float

    layered_config = config_override.LayeredConfig({})
    for (field, value) in (
        ("a", "false"),
        ("b", "45"),
        ("c", "5.6"),
    ):
        layered_config.set_override(field, value)

    effective_config = layered_config.get_effective_config()
    config = TestConfig(**effective_config)
    assert config.a is False
    assert config.b == 45
    assert config.c == 5.6


def test_float_override_of_integer_valued_float_field():
    """YAML loads `temperature: 0` as int, but `--override ... 0.7` is the documented example."""
    layered_config = config_override.LayeredConfig({"llms": {"nim_llm": {"_type": "nim", "temperature": 0}}})

    layered_config.set_override("llms.nim_llm.temperature", "0.7")

    assert layered_config.get_effective_config()["llms"]["nim_llm"]["temperature"] == 0.7


def test_integer_override_of_integer_field_stays_int():
    layered_config = config_override.LayeredConfig({"llms": {"nim_llm": {"max_tokens": 1024}}})

    layered_config.set_override("llms.nim_llm.max_tokens", "2048")

    value = layered_config.get_effective_config()["llms"]["nim_llm"]["max_tokens"]
    assert value == 2048 and isinstance(value, int)


def test_fractional_override_of_integer_field_rejected_by_model():
    """The override itself falls back to float, and the provider model still rejects it for an int field."""
    layered_config = config_override.LayeredConfig({"llms": {"nim_llm": {"model_name": "m", "max_tokens": 1024}}})

    layered_config.set_override("llms.nim_llm.max_tokens", "2.5")

    with pytest.raises(ValidationError, match="max_tokens"):
        NIMModelConfig(**layered_config.get_effective_config()["llms"]["nim_llm"])


def test_non_numeric_override_of_integer_field_still_rejected():
    layered_config = config_override.LayeredConfig({"llms": {"nim_llm": {"temperature": 0}}})

    with pytest.raises(click.BadParameter, match="Type mismatch"):
        layered_config.set_override("llms.nim_llm.temperature", "hot")
