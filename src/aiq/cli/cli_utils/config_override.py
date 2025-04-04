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

import logging
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import click
import yaml

from aiq.utils.data_models.schema_validator import validate_yaml

logger = logging.getLogger(__name__)


class _Placeholder:
    """Placeholder class to represent a value that is not set yet."""
    pass


class LayeredConfig:

    def __init__(self, base_config: dict[str, Any]):
        if not isinstance(base_config, dict):
            raise ValueError("Base config must be a dictionary")
        self.base_config = deepcopy(base_config)
        self.overrides: dict[str, Any] = {}
        self._effective_config: dict[str, Any] | None = None

    def validate_path(self, path: str) -> None:
        """Validate if a path exists in base config"""
        parts = path.split('.')
        current = self.base_config

        for i, part in enumerate(parts):
            if not isinstance(current, dict):
                current_path = '.'.join(parts[:i])
                raise click.BadParameter(f"Cannot navigate through non-dictionary value at '{current_path}'")
            if part not in current:
                if i == len(parts) - 1:
                    current[part] = _Placeholder()
                else:
                    current[part] = {}

            current = current[part]

    def set_override(self, path: str, value: str) -> None:
        """Set an override value with type conversion based on original config value.

        Args:
            path: Configuration path in dot notation (e.g., "llms.nim_llm.temperature")
            value: String value from CLI to override with

        Raises:
            click.BadParameter: If path doesn't exist or type conversion fails
            Exception: For other unexpected errors
        """
        try:
            # Validate path exists in config
            self.validate_path(path)

            # Get original value to determine type
            original_value = self.get_value(path)

            # Convert string value to appropriate type
            try:
                if isinstance(original_value, bool):
                    lower_value = value.lower().strip()
                    if lower_value not in ['true', 'false']:
                        raise ValueError(f"Boolean value must be 'true' or 'false', got '{value}'")
                    value = lower_value == 'true'
                elif isinstance(original_value, (int, float)):
                    value = type(original_value)(value)
                elif isinstance(original_value, list):
                    value = [v.strip() for v in value.split(',')]
                elif isinstance(original_value, Path):
                    value = Path(value)
            except (ValueError, TypeError) as e:
                raise click.BadParameter(f"Type mismatch for '{path}': expected {type(original_value).__name__}, "
                                         f"got '{value}' ({type(value).__name__}). Error: {str(e)}")

            # Store converted value
            self.overrides[path] = value
            self._effective_config = None

            log_msg = f"Successfully set override for {path} with value: {value}"
            if not isinstance(original_value, _Placeholder):
                log_msg += f" with type {type(value)})"

            logger.info(log_msg)

        except Exception as e:
            logger.error("Failed to set override for %s: %s", path, str(e))
            raise

    def get_value(self, path: str) -> Any:
        """Get value with better error messages"""
        try:
            if path in self.overrides:
                return self.overrides[path]

            parts = path.split('.')
            current = self.base_config

            for i, part in enumerate(parts):
                if not isinstance(current, dict):
                    current_path = '.'.join(parts[:i])
                    raise click.BadParameter(f"Cannot access '{path}': '{current_path}' is not a dictionary")
                if part not in current:
                    raise click.BadParameter(f"Path '{path}' not found: '{part}' does not exist")
                current = current[part]

            return current

        except Exception as e:
            logger.error("Error accessing path %s: %s", path, e)
            raise

    def _update_config_value(self, config: dict[str, Any], path: str, value: Any) -> None:
        """Update a single value in the config dictionary at the specified path.

        Args:
            config: The configuration dictionary to update
            path: String representing the path to the value using dot notation (e.g. "llms.nim_llm.temperature")
            value: The new value to set at the specified path

        Example:
            If config is {"llms": {"nim_llm": {"temperature": 0.5}}}
            and path is "llms.nim_llm.temperature" with value 0.7,
            this will update config to {"llms": {"nim_llm": {"temperature": 0.7}}}
        """
        parts = path.split('.')
        current = config
        # Navigate through nested dictionaries until reaching the parent of target
        for part in parts[:-1]:
            current = current[part]
        # Update the value at the target location
        current[parts[-1]] = value

    def get_effective_config(self) -> dict[str, Any]:
        """Get the configuration with all overrides applied.

        Creates a new configuration dictionary by applying all stored overrides
        to a deep copy of the base configuration. Caches the result to avoid
        recomputing unless overrides change.

        Returns:
            Dict containing the full configuration with all overrides applied

        Note:
            The configuration is cached in self._effective_config and only
            recomputed when new overrides are added via set_override()
        """
        # Return cached config if available
        if self._effective_config is not None:
            return self._effective_config

        # Create deep copy to avoid modifying base config
        config = deepcopy(self.base_config)

        # Apply each override to the config copy
        for path, value in self.overrides.items():
            self._update_config_value(config, path, value)

        # Return the result
        self._effective_config = config
        return config


def load_and_override_config(config_file: Path, overrides: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    """Load config file and apply any overrides"""
    # First validate the original config file
    validate_yaml(None, None, config_file)

    # Load the base config
    with open(config_file, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    # Create layered config
    config = LayeredConfig(base_config)

    # Apply overrides if any
    if overrides:
        for param_path, value in overrides:
            config.set_override(param_path, value)

        effective_config = config.get_effective_config()

        # Second validation is necessary to ensure overrides haven't created an invalid config
        # For example, overrides might break required relationships between fields
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as tmp:
            yaml.safe_dump(effective_config, tmp)
            tmp_path = Path(tmp.name)

        try:
            # Validate using the temporary file
            validate_yaml(None, None, tmp_path)
            # If validation succeeds, print the config
            logger.info(
                "\n\nConfiguration after overrides:\n\n%s",
                yaml.dump(effective_config, default_flow_style=False),
            )
        except Exception as e:
            logger.exception("Modified configuration failed validation: %s", e, exc_info=True)
            raise click.BadParameter(f"Modified configuration failed validation: {str(e)}")
        finally:
            # Clean up the temporary file
            tmp_path.unlink()

    return config.get_effective_config()


def add_override_option(command):
    """Decorator to add override option to a command"""
    return click.option(
        '--override',
        type=(str, str),
        multiple=True,
        help="Override config values using dot notation (e.g., --override llms.nim_llm.temperature 0.7)")(command)
