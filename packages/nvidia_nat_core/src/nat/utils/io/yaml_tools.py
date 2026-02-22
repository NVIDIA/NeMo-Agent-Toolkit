# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path

import expandvars
import yaml

from nat.utils.type_utils import StrPath

logger = logging.getLogger(__name__)

# Prefix for file protocol references to indicate content should be loaded from a file
FILE_PROTOCOL_PREFIX = "file://"

# Allowed file extensions for file references (security: prevent loading code files)
ALLOWED_FILE_EXTENSIONS = frozenset({".txt", ".md", ".j2", ".jinja2", ".jinja", ".prompt", ".tpl", ".template"})


def _load_file_content(file_path: StrPath) -> str:
    """
    Load content from a file.

    Args:
        file_path: Path to the file to load.

    Returns:
        The file content as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Referenced file not found: {file_path}")

    with open(path, encoding="utf-8") as f:
        return f.read()


def _validate_file_extension(file_path: Path) -> None:
    """
    Validate that a referenced file has an allowed extension.

    Args:
        file_path: Path to the file.

    Raises:
        ValueError: If the file extension is not in the allowed list.
    """
    ext = file_path.suffix.lower()
    if ext not in ALLOWED_FILE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_FILE_EXTENSIONS))
        raise ValueError(f"Unsupported file extension '{ext}' for file reference: {file_path}. "
                         f"Allowed extensions: {allowed}")


def _resolve_file_references(config: dict, base_path: Path) -> dict:
    """
    Recursively resolve file:// references in configuration fields.

    Resolves any string value starting with "file://" by loading
    the referenced file's contents as a string.

    Args:
        config: The configuration dictionary to process.
        base_path: The base path for resolving relative file paths.

    Returns:
        A new dictionary with file:// references resolved to file contents.
    """
    result = {}

    for key, value in config.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            result[key] = _resolve_file_references(value, base_path)
        elif isinstance(value, str) and value.startswith(FILE_PROTOCOL_PREFIX):
            # Load file content for any string field with file:// prefix
            file_path_str = value[len(FILE_PROTOCOL_PREFIX):]

            # Resolve relative paths from base_path
            resolved_path = Path(file_path_str)
            if not resolved_path.is_absolute():
                resolved_path = base_path / file_path_str

            # Resolve symlinks to validate the actual file extension
            resolved_path = resolved_path.resolve()

            # Validate file extension before loading
            _validate_file_extension(resolved_path)

            result[key] = _load_file_content(resolved_path)
        else:
            # Keep other values unchanged
            result[key] = value

    return result


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


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override dictionary into base dictionary.

    Args:
        base (dict): The base configuration dictionary.
        override (dict): The override configuration dictionary.

    Returns:
        dict: The merged configuration dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def yaml_load(config_path: StrPath, _visited: set[Path] | None = None) -> dict:
    """
    Load a YAML file and interpolate variables in the format
    ${VAR:-default_value}.

    If the YAML file contains a "base" key, the file at that path will be
    loaded first, and the current config will be merged on top of it. This enables
    config inheritance to reduce duplication across similar configuration files.

    Args:
        config_path (StrPath): The path to the YAML file to load.
        _visited (set[Path] | None): Internal parameter for circular dependency detection.

    Returns:
        dict: The processed configuration dictionary.

    Raises:
        TypeError: If the "base" key is not a string.
        FileNotFoundError: If the base configuration file does not exist.
        ValueError: If a circular dependency is detected in configuration inheritance.
    """
    # Normalize the config path and detect circular dependencies
    config_path_obj = Path(config_path).resolve()

    if _visited is None:
        _visited = set()

    if config_path_obj in _visited:
        raise ValueError(f"Circular dependency detected in configuration inheritance: {config_path_obj} "
                         f"is already in the inheritance chain")

    _visited.add(config_path_obj)

    # Read YAML file
    with open(config_path_obj, encoding="utf-8") as stream:
        config_str = stream.read()

    base_path = config_path_obj.parent
    config = yaml_loads(config_str, base_path)

    # Check if config specifies a base for inheritance
    if "base" in config:
        base_path_str = config["base"]

        # Validate that base is a string
        if not isinstance(base_path_str, str):
            raise TypeError(f"Configuration 'base' key must be a string, got {type(base_path_str).__name__}")

        # Resolve base path relative to current config
        if not Path(base_path_str).is_absolute():
            base_path = base_path / base_path_str
        else:
            base_path = Path(base_path_str)

        # Normalize and check if base file exists
        base_path = base_path.resolve()
        if not base_path.exists():
            raise FileNotFoundError(f"Base configuration file not found: {base_path}")

        # Load base config (recursively, so bases can have bases)
        base_config = yaml_load(base_path, _visited=_visited)

        # Perform deep merge and remove 'base' key from result
        config = deep_merge(base_config, config)
        config.pop("base", None)

    return config


def yaml_loads(config: str, base_path: Path) -> dict:
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
        logger.error("Error loading YAML: %s", interpolated_config_str)
        raise ValueError(f"Error loading YAML: {e}") from e

    assert isinstance(config_data, dict)

    config_data = _resolve_file_references(config_data, base_path)

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
