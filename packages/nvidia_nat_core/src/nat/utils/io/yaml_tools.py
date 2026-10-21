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
import os
import re
import typing
from pathlib import Path

import yaml
from expandvars import expand

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

    return expand(value, surrounded_vars_only=True)


# A plain YAML value that is exactly one reference, such as `${VAR}` or `${VAR:-default}`.
_WHOLE_REFERENCE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:(?P<op>:?-)(?P<default>[^${}]*))?\}")


def _interpolate_whole_reference(match: re.Match[str]) -> typing.Any:
    """
    Resolve a plain YAML value that is exactly one `${...}` reference.

    The value keeps the YAML type it had when variables were expanded on the raw text: numbers,
    booleans, null and `[...]` / `{...}` collections are parsed. Any other value is used verbatim,
    so YAML comment and indicator characters in a value are not interpreted.

    Args:
        match: The `_WHOLE_REFERENCE` match for the value.

    Returns:
        The resolved value.
    """
    default = match["default"]
    if default is not None and len(default) >= 2 and default[0] == default[-1] and default[0] in "\"'":
        env_value = os.environ.get(match["name"])
        if env_value is None or (match["op"] == ":-" and env_value == ""):
            # A quoted default, as in `${VAR:-"http://default.com"}`, is a string without its quotes.
            return default[1:-1]

    text: str = expand(match.group(0), surrounded_vars_only=True)
    if text.strip() == "":
        # Matches loading the expanded text: an empty plain value is null.
        return None

    try:
        node = yaml.compose(text, Loader=yaml.SafeLoader)
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        return text

    if node is None or node.end_mark.index < len(text.rstrip()):
        # The YAML value ends before the text does, e.g. `123 #42` parses as `123`. Keep the text.
        return text

    if isinstance(parsed, bool | int | float):
        return parsed
    if parsed is None and text.strip() in ("~", "null", "Null", "NULL"):
        return None
    if isinstance(parsed, list | dict) and text.lstrip()[:1] in ("[", "{"):
        return parsed
    return text


# Tag the loader resolves an untagged plain `${VAR}` value to. Quoted values and values with an explicit
# tag such as `!!str` are not resolved, so they stay strings.
_WHOLE_REFERENCE_TAG = "tag:nvidia.com,2026:nat-env-reference"

# A `${...}` reference on one line. Its braces are YAML flow indicators, so each reference is swapped
# for a placeholder of letters, digits and `_` before parsing and swapped back in each scalar.
_REFERENCE = re.compile(r"\$\{[^{}\n]*\}")
_PLACEHOLDER = re.compile(r"NATENVREF[0-9a-f]{12}N[0-9]+N")


def _protect_references(config: str) -> tuple[str, dict[str, str]]:
    """
    Replace each `${...}` reference in the raw YAML text with a placeholder that is safe in any YAML context.

    Args:
        config (str): The raw YAML text.

    Returns:
        tuple[str, dict[str, str]]: The text with placeholders, and each placeholder's original reference.
    """
    nonce = os.urandom(6).hex()
    references: dict[str, str] = {}

    def _swap(match: re.Match[str]) -> str:
        placeholder = f"NATENVREF{nonce}N{len(references)}N"
        references[placeholder] = match.group(0)
        return placeholder

    return _REFERENCE.sub(_swap, config), references


class _EnvVarLoader(yaml.SafeLoader):
    """Safe YAML loader that expands `${VAR}` references in scalar values after parsing."""

    references: dict[str, str] = {}

    def _restore(self, value: str) -> str:
        return _PLACEHOLDER.sub(lambda m: self.references.get(m.group(0), m.group(0)), value)

    def resolve(self, kind: typing.Any, value: typing.Any, implicit: typing.Any) -> str:
        if kind is yaml.ScalarNode and implicit[0] and _WHOLE_REFERENCE.fullmatch(self._restore(value)):
            return _WHOLE_REFERENCE_TAG
        return super().resolve(kind, value, implicit)

    def _compose_reference(self, node: yaml.Node) -> yaml.Node:
        """Turn a whole-reference scalar into the node its expanded text composes to, for use as a merge value."""
        if not isinstance(node, yaml.ScalarNode) or node.tag != _WHOLE_REFERENCE_TAG:
            return node
        text = expand(self._restore(yaml.SafeLoader.construct_scalar(self, node)), surrounded_vars_only=True)
        try:
            composed = yaml.compose(text, Loader=yaml.SafeLoader)
        except yaml.YAMLError:
            return node
        return composed if isinstance(composed, yaml.MappingNode) else node

    def flatten_mapping(self, node: yaml.MappingNode) -> None:
        # PyYAML checks `<<` merge values by node type before any scalar is constructed, so a merge value
        # that comes from a reference has to be composed into a mapping node first.
        for index, (key_node, value_node) in enumerate(node.value):
            if key_node.tag != "tag:yaml.org,2002:merge":
                continue
            if isinstance(value_node, yaml.SequenceNode):
                value_node.value = [self._compose_reference(item) for item in value_node.value]
            else:
                node.value[index] = (key_node, self._compose_reference(value_node))
        super().flatten_mapping(node)

    def construct_scalar(self, node: yaml.ScalarNode) -> str:
        # Every scalar, including keys and explicitly tagged values, is expanded exactly once here.
        return expand(self._restore(super().construct_scalar(node)), surrounded_vars_only=True)


def _env_var_loader(references: dict[str, str]) -> type[_EnvVarLoader]:
    """Create an `_EnvVarLoader` subclass that restores the given placeholders."""
    return typing.cast(type[_EnvVarLoader],
                       type("_EnvVarLoaderForConfig", (_EnvVarLoader, ), {"references": references}))


def _construct_whole_reference(loader: _EnvVarLoader, node: yaml.ScalarNode) -> typing.Any:
    match = _WHOLE_REFERENCE.fullmatch(loader._restore(yaml.SafeLoader.construct_scalar(loader, node)))
    assert match is not None  # the resolver only assigns this tag to a full match
    return _interpolate_whole_reference(match)


_EnvVarLoader.add_constructor(_WHOLE_REFERENCE_TAG, _construct_whole_reference)


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
        base_path (Path): The base directory for resolving relative file references.

    Returns:
        dict: The processed configuration dictionary.

    Raises:
        ValueError: If the YAML is invalid or its top level is not a mapping.
    """
    # Variables are expanded in the parsed string values, so a value is never read again as YAML.
    config_text, references = _protect_references(config)

    stream = io.StringIO(config_text)
    stream.seek(0)

    # Load the YAML data
    try:
        config_data = yaml.load(stream, Loader=_env_var_loader(references))  # noqa: S506  # a SafeLoader
    except yaml.YAMLError as e:
        # Parser messages and source snippets can contain interpolated credentials.
        # Report only the error type and numeric location, including in tracebacks.
        error_detail = type(e).__name__
        if isinstance(e, yaml.MarkedYAMLError) and e.problem_mark is not None:
            mark = e.problem_mark
            error_detail += f" at line {mark.line + 1}, column {mark.column + 1}"
        logger.error("Error loading YAML: %s", error_detail)
        raise ValueError(f"Error loading YAML: {error_detail}") from None

    if not isinstance(config_data, dict):
        raise ValueError(f"Error loading YAML: the top level must be a mapping, got {type(config_data).__name__}")

    # Each string value was expanded once while loading. Expanding again would also expand any
    # "${...}" that came from an environment variable's own value.
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
