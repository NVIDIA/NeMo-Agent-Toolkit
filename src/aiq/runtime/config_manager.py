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
"""
Configuration manager for AIQ Toolkit hot-reloading functionality.

This module provides configuration lifecycle management including loading,
reloading, validation, and rollback capabilities for development workflows.
"""

from __future__ import annotations

import logging
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from aiq.cli.cli_utils.config_override import LayeredConfig
from aiq.data_models.config import AIQConfig
from aiq.runtime.events import ConfigChangeEvent
from aiq.runtime.events import ConfigEventType
from aiq.runtime.events import get_event_manager
from aiq.runtime.loader import load_config
from aiq.utils.type_utils import StrPath

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails during reload."""
    pass


class ConfigReloadError(Exception):
    """Raised when configuration reload fails."""
    pass


class ConfigSnapshot:
    """
    Represents a snapshot of configuration state for rollback purposes.
    """

    def __init__(self, config: AIQConfig, overrides: dict[str, Any] | None = None, timestamp: datetime | None = None):
        self.config = deepcopy(config)
        self.overrides = deepcopy(overrides) if overrides else {}
        self.timestamp = timestamp or datetime.now()

    def __repr__(self) -> str:
        return f"ConfigSnapshot(timestamp={self.timestamp.isoformat()})"


class ConfigManager:
    """
    Manages configuration lifecycle for AIQ Toolkit applications.

    This class provides centralized configuration management with support for:
    - Hot-reloading configuration files
    - Configuration validation before applying changes
    - Rollback capabilities on validation failures
    - Preservation of configuration overrides
    - Event-driven reload notifications
    """

    def __init__(self, config_file: StrPath, config: AIQConfig | None = None):
        """
        Initialize the configuration manager.

        Parameters
        ----------
        config_file : StrPath
            Path to the configuration file
        config : AIQConfig, optional
            Pre-loaded configuration. If None, will load from file.
        """
        self.config_file = Path(config_file).resolve()
        self._current_config = config or load_config(self.config_file)
        self._layered_config: LayeredConfig | None = None
        self._snapshots: list[ConfigSnapshot] = []
        self._max_snapshots = 10
        self._lock = threading.RLock()
        self._reload_count = 0

        # Create initial snapshot
        self._create_snapshot()

        # Register for configuration change events
        get_event_manager().register_handler(self._on_config_file_changed, ConfigEventType.FILE_MODIFIED)

        logger.info("Configuration manager initialized for %s", self.config_file)

    @property
    def current_config(self) -> AIQConfig:
        """Get the current configuration."""
        with self._lock:
            return deepcopy(self._current_config)

    @property
    def config_overrides(self) -> dict[str, Any]:
        """Get the current configuration overrides."""
        with self._lock:
            return self._layered_config.overrides if self._layered_config else {}

    @property
    def reload_count(self) -> int:
        """Get the number of times configuration has been reloaded."""
        with self._lock:
            return self._reload_count

    def set_overrides(self, overrides: dict[str, Any]) -> None:
        """
        Set configuration overrides that will be preserved during reloads.

        Parameters
        ----------
        overrides : dict[str, Any]
            Dictionary of configuration overrides in dot notation
        """
        with self._lock:
            config_dict = self._current_config.model_dump()
            self._layered_config = LayeredConfig(config_dict)

            for path, value in overrides.items():
                try:
                    self._layered_config.set_override(path, str(value))
                    logger.debug("Applied configuration override: %s = %s", path, value)
                except Exception as e:
                    logger.warning("Failed to apply override %s = %s: %s", path, value, e)

            logger.info("Applied %d configuration overrides", len(overrides))

    def reload_config(self, validate_only: bool = False) -> Path:
        """
        Reload configuration from file with validation and rollback support.

        Parameters
        ----------
        validate_only : bool, optional
            If True, only validate the new configuration without applying it

        Returns
        -------
        Path
            The config file path if validation succeeds

        Raises
        ------
        ConfigValidationError
            If the new configuration is invalid
        ConfigReloadError
            If reload fails for other reasons
        """
        with self._lock:
            logger.info("Starting configuration reload from %s", self.config_file)

            try:
                # Create snapshot before attempting reload
                if not validate_only:
                    self._create_snapshot()

                # Load and validate new configuration
                new_config = self._load_and_validate_config()

                if validate_only:
                    logger.info("Configuration validation successful")
                    return self.config_file

                # Apply the new configuration
                self._current_config = new_config
                self._reload_count += 1

                # Reapply overrides if they exist
                if self._layered_config:
                    self._reapply_overrides()

                logger.info("Configuration reload successful (reload #%d)", self._reload_count)

                return self.config_file

            except Exception as e:
                if validate_only:
                    logger.error("Configuration validation failed: %s", e)
                    raise ConfigValidationError(f"Configuration validation failed: {e}") from e
                else:
                    logger.error("Configuration reload failed: %s", e)
                    raise ConfigReloadError(f"Configuration reload failed: {e}") from e

    def rollback_config(self, steps: int = 1) -> Path:
        """
        Rollback configuration to a previous snapshot.

        Parameters
        ----------
        steps : int, optional
            Number of steps to rollback, by default 1

        Returns
        -------
        Path
            The config file path after rollback

        Raises
        ------
        ConfigReloadError
            If rollback is not possible
        """
        with self._lock:
            if len(self._snapshots) <= steps:
                raise ConfigReloadError(
                    f"Cannot rollback {steps} steps, only {len(self._snapshots)} snapshots available")

            # Get the target snapshot (excluding the current one)
            target_snapshot = self._snapshots[-(steps + 1)]

            logger.info("Rolling back configuration %d steps to snapshot from %s",
                        steps,
                        target_snapshot.timestamp.isoformat())

            # Restore configuration and overrides
            self._current_config = deepcopy(target_snapshot.config)

            if target_snapshot.overrides:
                config_dict = self._current_config.model_dump()
                self._layered_config = LayeredConfig(config_dict)
                for path, value in target_snapshot.overrides.items():
                    try:
                        self._layered_config.set_override(path, str(value))
                    except Exception as e:
                        logger.warning("Failed to restore override %s = %s: %s", path, value, e)
            else:
                self._layered_config = None

            self._reload_count += 1

            # Remove snapshots after the rollback point
            self._snapshots = self._snapshots[:-(steps)]

            logger.info("Configuration rollback completed")
            return self.config_file

    def get_snapshots(self) -> list[ConfigSnapshot]:
        """
        Get all configuration snapshots.

        Returns
        -------
        list[ConfigSnapshot]
            List of configuration snapshots, most recent first
        """
        with self._lock:
            return list(reversed(self._snapshots))

    def clear_snapshots(self) -> None:
        """Clear all configuration snapshots except the current one."""
        with self._lock:
            if self._snapshots:
                current_snapshot = self._snapshots[-1]
                self._snapshots = [current_snapshot]
            logger.info("Cleared configuration snapshots")

    def _load_and_validate_config(self) -> AIQConfig:
        """Load and validate configuration from file."""
        if not self.config_file.exists():
            raise ConfigValidationError(f"Configuration file not found: {self.config_file}")

        try:
            # Load new configuration using the standard loader
            new_config = load_config(self.config_file)
            logger.debug("Successfully loaded and validated configuration from %s", self.config_file)
            return new_config

        except Exception as e:
            raise ConfigValidationError(f"Failed to load configuration: {e}") from e

    def _reapply_overrides(self) -> None:
        """Reapply configuration overrides to the current configuration."""
        if not self._layered_config:
            return

        old_overrides = deepcopy(self._layered_config.overrides)

        # Create new layered config with current configuration
        config_dict = self._current_config.model_dump()
        self._layered_config = LayeredConfig(config_dict)

        # Reapply overrides
        applied_count = 0
        for path, value in old_overrides.items():
            try:
                self._layered_config.set_override(path, str(value))
                applied_count += 1
                logger.debug("Reapplied configuration override: %s = %s", path, value)
            except Exception as e:
                logger.warning("Failed to reapply override %s = %s: %s", path, value, e)

        logger.debug("Reapplied %d/%d configuration overrides", applied_count, len(old_overrides))

    def _create_snapshot(self) -> None:
        """Create a snapshot of the current configuration state."""
        overrides = self.config_overrides if self._layered_config else None
        snapshot = ConfigSnapshot(self._current_config, overrides)

        self._snapshots.append(snapshot)

        # Limit number of snapshots
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)

        logger.debug("Created configuration snapshot: %s", snapshot)

    def _on_config_file_changed(self, event: ConfigChangeEvent) -> None:
        """Handle configuration file change events."""
        if event.file_path.resolve() == self.config_file:
            logger.debug("Configuration file change detected: %s", event.file_path)
            # Note: This is just for logging/monitoring in manual trigger mode
            # Automatic reloading will be implemented in Step 3

    def __enter__(self) -> ConfigManager:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        # Unregister event handler
        try:
            get_event_manager().unregister_handler(self._on_config_file_changed, ConfigEventType.FILE_MODIFIED)
        except Exception as e:
            logger.warning("Failed to unregister config change handler: %s", e)
