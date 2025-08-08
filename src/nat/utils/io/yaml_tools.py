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
import logging
import typing

import expandvars
import yaml

from nat.utils.type_utils import StrPath

logger = logging.getLogger(__name__)


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

    return expandvars.expandvars(value)


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
    with open(config_path, "r", encoding="utf-8") as stream:
        config_str = stream.read()

    return yaml_loads(config_str)


def yaml_loads(config: str) -> dict:
    """
    Load a YAML string and interpolate variables in the format
    ${VAR:-default_value}.

    Args:
        config (str): The YAML string to load.

    Returns:
        dict: The processed configuration dictionary.
    """

    interpolated_config_str = _interpolate_variables(config)
    assert isinstance(interpolated_config_str, str), "Config must be a string"

    stream = io.StringIO(interpolated_config_str)
    stream.seek(0)

    # Load the YAML data
    try:
        config_data = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        logger.error("Error loading YAML: %s", interpolated_config_str, exc_info=True)
        raise ValueError(f"Error loading YAML: {e}") from e

    assert isinstance(config_data, dict)

    return config_data


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
