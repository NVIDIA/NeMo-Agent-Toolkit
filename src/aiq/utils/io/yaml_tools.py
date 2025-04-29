# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import io
import os
import re
import typing

import yaml

from aiq.utils.type_utils import StrPath


def _interpolate_variables(value: str | int | float | bool | None) -> str | int | float | bool | None:
    """
    Interpolate variables in a string with the format ${VAR:-default_value}.
    If the variable is not set, the default value will be used.
    If no default value is provided, an empty string will be used.

    Args:
        value (str | int | float | bool | None): The value to interpolate variables in.

    Returns:
        str | int | float | bool | None: The value with variables interpolated.
    """

    if not isinstance(value, str):
        return value

    def replace_var(match):
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) else ""
        return os.environ.get(var_name, default_value)

    pattern = r'\${([^:}]+)(?::-([^}]*))?}'
    return re.sub(pattern, replace_var, value)


def _process_config(
        config_data: dict | list | str | int | float | bool | None) -> dict | list | str | int | float | bool | None:
    """
    Recursively process a configuration dictionary to interpolate variables.

    Args:
        config_data (dict): The configuration dictionary to process.

    Returns:
        dict: The processed configuration dictionary.
    """

    if isinstance(config_data, dict):
        return {k: _process_config(v) for k, v in config_data.items()}
    if isinstance(config_data, list):
        return [_process_config(item) for item in config_data]
    if isinstance(config_data, (str, int, float, bool, type(None))):
        return _interpolate_variables(config_data)

    raise ValueError(f"Unsupported type: {type(config_data)}")


def yaml_load(config_path: StrPath) -> dict:
    """
    Load a YAML file and interpolate variables in the format
    ${VAR:-default_value}.

    Args:
        config_path (StrPath): The path to the YAML file to load.

    Returns:
        dict: The processed configuration dictionary.
    """

    # Read YAML file
    with open(config_path, 'r', encoding="utf-8") as stream:
        config_data = yaml.safe_load(stream)

    # Process the configuration to interpolate variables
    config_data = _process_config(config_data)
    assert isinstance(config_data, dict)

    return config_data


def yaml_loads(config: str) -> dict:
    """
    Load a YAML string and interpolate variables in the format
    ${VAR:-default_value}.

    Args:
        config (str): The YAML string to load.

    Returns:
        dict: The processed configuration dictionary.
    """

    stream = io.StringIO(config)
    stream.seek(0)

    config_data = yaml.safe_load(stream)
    return _process_config(config_data)


def yaml_dump(config: dict, fp: typing.TextIO) -> None:
    """
    Dump a configuration dictionary to a YAML file.

    Args:
        config (dict): The configuration dictionary to dump.
        fp (typing.TextIO): The file pointer to write the YAML to.
    """
    yaml.dump(config, stream=fp, indent=2, sort_keys=False)
    fp.flush()


def yaml_dumps(config: dict) -> str:
    """
    Dump a configuration dictionary to a YAML string.

    Args:
        config (dict): The configuration dictionary to dump.

    Returns:
        str: The YAML string.
    """

    return yaml.dump(config, indent=2)
