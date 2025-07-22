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
Development mode manager for AIQ Toolkit.

This module provides automatic configuration reloading during development,
integrating file watching and configuration management for improved
developer experience.
"""

import asyncio
import logging
import signal
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

from aiq.data_models.config import AIQConfig
from aiq.runtime.config_manager import ConfigManager
from aiq.runtime.config_watcher import ConfigWatcher
from aiq.runtime.events import ConfigChangeEvent
from aiq.runtime.events import ConfigEventType

logger = logging.getLogger(__name__)


class DevModeManager:
    """
    Development mode manager that provides automatic configuration reloading.

    Integrates file watching and configuration management to automatically
    reload workflows when configuration files change during development.
    """

    def __init__(self,
                 config_file: Path,
                 override: tuple[tuple[str, str], ...] = (),
                 watch_additional_files: set[Path] | None = None,
                 reload_delay: float = 0.1,
                 max_reload_attempts: int = 3):
        """
        Initialize the development mode manager.

        Args:
            config_file: Primary configuration file to watch and reload
            override: Configuration overrides to preserve across reloads
            watch_additional_files: Additional files to watch for changes
            reload_delay: Delay in seconds before triggering reload after file change
            max_reload_attempts: Maximum number of consecutive reload attempts before giving up
        """
        self.config_file = config_file.resolve()
        self.override = override
        self.watch_additional_files = watch_additional_files or set()
        self.reload_delay = reload_delay
        self.max_reload_attempts = max_reload_attempts

        # Core components
        self.config_manager: ConfigManager | None = None
        self.config_watcher: ConfigWatcher | None = None

        # State management
        self._is_running = False
        self._reload_lock = threading.Lock()
        self._reload_attempts = 0
        self._last_successful_config: AIQConfig | None = None

        # Event handling
        self._reload_callbacks: list[Callable[[Path], None]] = []
        self._error_callbacks: list[Callable[[Exception], None]] = []

        # Signal handling for graceful shutdown
        self._original_sigint_handler = None
        self._original_sigterm_handler = None

    def add_reload_callback(self, callback: Callable[[Path], None]) -> None:
        """Add a callback to be called when configuration is successfully reloaded."""
        self._reload_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add a callback to be called when a reload error occurs."""
        self._error_callbacks.append(callback)

    async def start(self) -> AIQConfig:
        """
        Start development mode and return the initial configuration.

        Returns:
            Initial validated configuration

        Raises:
            ConfigValidationError: If initial configuration is invalid
            FileNotFoundError: If configuration file doesn't exist
        """
        if self._is_running:
            raise RuntimeError("Development mode is already running")

        logger.info("Starting development mode for config: %s", self.config_file)

        try:
            # Create and initialize config manager
            from aiq.runtime.loader import create_config_manager
            self.config_manager = create_config_manager(self.config_file)

            # Apply overrides if provided
            if self.override:
                override_dict = {key: value for key, value in self.override}
                self.config_manager.set_overrides(override_dict)

            # Get initial configuration
            initial_config = self.config_manager.current_config
            self._last_successful_config = initial_config

            # Set up file watching
            watched_files = {self.config_file} | self.watch_additional_files
            self.config_watcher = ConfigWatcher()

            for file_path in watched_files:
                if file_path.exists():
                    self.config_watcher.add_file(file_path)
                    logger.debug("Watching file: %s", file_path)

            # Register event handler with global event manager
            from aiq.runtime.events import get_event_manager
            get_event_manager().register_handler(self._handle_file_change)

            # Start watching
            self.config_watcher.start()

            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()

            self._is_running = True
            logger.info("Development mode started successfully")

            # Show helpful information
            self._show_dev_mode_info()

            return initial_config

        except Exception as e:
            logger.error("Failed to start development mode: %s", e)
            await self._cleanup()
            raise

    async def stop(self) -> None:
        """Stop development mode and clean up resources."""
        if not self._is_running:
            return

        logger.info("Stopping development mode...")

        await self._cleanup()
        self._is_running = False

        logger.info("Development mode stopped")

    def get_current_config(self) -> AIQConfig | None:
        """Get the current configuration."""
        if self.config_manager:
            return self.config_manager.current_config
        return self._last_successful_config

    def _handle_file_change(self, event: ConfigChangeEvent) -> None:
        """Handle file change events from the watcher."""
        if not self._is_running:
            return

        # Only process file modifications and creations
        if event.event_type not in (ConfigEventType.FILE_MODIFIED, ConfigEventType.FILE_CREATED):
            return

        logger.info("Configuration file changed: %s", event.file_path)

        # Use a lock to prevent concurrent reloads
        if not self._reload_lock.acquire(blocking=False):
            logger.debug("Skipping reload (already in progress)")
            return

        try:
            # Use threading to avoid blocking the file watcher
            threading.Thread(target=self._delayed_reload, name="config-reload", daemon=True).start()
        except Exception as e:
            logger.error("Failed to start reload thread: %s", e)
            self._reload_lock.release()

    def _delayed_reload(self) -> None:
        """Perform delayed configuration reload in a separate thread."""
        try:
            # Wait for file system to settle
            threading.Event().wait(self.reload_delay)

            # Perform the reload
            self._perform_reload()

        finally:
            self._reload_lock.release()

    def _perform_reload(self) -> None:
        """Perform the actual configuration reload."""
        if not self.config_manager or not self._is_running:
            return

        try:
            logger.info("Reloading configuration...")

            # Attempt to reload the configuration
            config_file_path = self.config_manager.reload_config()

            # Reset reload attempts on success
            self._reload_attempts = 0
            self._last_successful_config = self.config_manager.current_config

            logger.info("Configuration reloaded successfully")

            # Notify callbacks
            for callback in self._reload_callbacks:
                try:
                    callback(config_file_path)
                except Exception as e:
                    logger.error("Error in reload callback: %s", e)

        except Exception as e:
            self._reload_attempts += 1
            logger.error("Configuration reload failed (attempt %d/%d): %s",
                         self._reload_attempts,
                         self.max_reload_attempts,
                         e)

            # Notify error callbacks
            for callback in self._error_callbacks:
                try:
                    callback(e)
                except Exception as callback_error:
                    logger.error("Error in error callback: %s", callback_error)

            # If we've exceeded max attempts, try to rollback
            if self._reload_attempts >= self.max_reload_attempts:
                logger.warning("Maximum reload attempts exceeded, attempting rollback...")
                try:
                    rollback_config_path = self.config_manager.rollback_config()
                    self._reload_attempts = 0  # Reset after successful rollback
                    logger.info("Configuration rolled back successfully")

                    # Notify callbacks about rollback
                    for callback in self._reload_callbacks:
                        try:
                            callback(rollback_config_path)
                        except Exception as callback_error:
                            logger.error("Error in rollback callback: %s", callback_error)

                except Exception as rollback_error:
                    logger.error("Rollback failed: %s", rollback_error)
                    logger.warning("Configuration may be in an inconsistent state")

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info("Received signal %d, shutting down development mode...", signum)
            asyncio.create_task(self.stop())

        # Store original handlers
        self._original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, signal_handler)

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
        if self._original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)

    def _show_dev_mode_info(self) -> None:
        """Show helpful development mode information."""
        click.echo(click.style("\nDevelopment Mode Active", fg="green", bold=True))
        click.echo(click.style("=" * 50, fg="green"))
        click.echo(f"Watching config: {self.config_file}")

        if self.watch_additional_files:
            click.echo("Additional files:")
            for file_path in sorted(self.watch_additional_files):
                click.echo(f"   • {file_path}")

        click.echo(f"Reload delay: {self.reload_delay}s")
        click.echo("\nConfiguration changes will be automatically reloaded")
        click.echo("Press Ctrl+C to stop development mode")
        click.echo(click.style("=" * 50 + "\n", fg="green"))

    async def _cleanup(self) -> None:
        """Clean up all resources."""
        # Stop file watcher
        if self.config_watcher:
            try:
                self.config_watcher.stop()
                self.config_watcher = None
            except Exception as e:
                logger.error("Error stopping config watcher: %s", e)

        # Clean up config manager
        if self.config_manager:
            try:
                # Config manager cleanup happens automatically via context manager
                self.config_manager = None
            except Exception as e:
                logger.error("Error cleaning up config manager: %s", e)

        # Restore signal handlers
        self._restore_signal_handlers()

    async def __aenter__(self):
        """Async context manager entry."""
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


async def run_with_dev_mode(config_file: Path,
                            override: tuple[tuple[str, str], ...],
                            workflow_runner: Callable[[], Any],
                            watch_additional_files: set[Path] | None = None) -> None:
    """
    Run a workflow with development mode enabled.

    Args:
        config_file: Configuration file to watch
        override: Configuration overrides
        workflow_runner: Async function that runs the workflow with given config
        watch_additional_files: Additional files to watch for changes
    """
    dev_manager = DevModeManager(config_file=config_file,
                                 override=override,
                                 watch_additional_files=watch_additional_files)

    # Track the current workflow task
    current_workflow_task: asyncio.Task | None = None
    workflow_shutdown_event = asyncio.Event()

    # Get the current event loop for thread-safe scheduling
    main_loop = asyncio.get_running_loop()

    async def start_workflow(config_file_path: Path) -> None:
        """Start or restart the workflow with the given configuration file path."""
        nonlocal current_workflow_task

        # Stop existing workflow if running
        if current_workflow_task and not current_workflow_task.done():
            logger.info("Stopping current workflow for reload...")
            current_workflow_task.cancel()
            try:
                await current_workflow_task
            except asyncio.CancelledError:
                pass

        # Start new workflow
        logger.info("Starting workflow with updated configuration...")
        workflow_shutdown_event.clear()
        current_workflow_task = asyncio.create_task(workflow_runner())

    async def handle_reload(config_file_path: Path) -> None:
        """Handle configuration reload."""
        await start_workflow(config_file_path)

    def handle_error(error: Exception) -> None:
        """Handle reload errors."""
        logger.warning("Workflow continues with previous configuration due to reload error")

    # Set up callbacks with proper async handling
    def sync_reload_callback(config_file_path: Path) -> None:
        """Synchronous wrapper for async reload callback."""
        try:
            # Schedule the coroutine to run in the main event loop from background thread
            asyncio.run_coroutine_threadsafe(handle_reload(config_file_path), main_loop)
        except Exception as e:
            logger.error("Failed to schedule reload callback: %s", e)

    dev_manager.add_reload_callback(sync_reload_callback)
    dev_manager.add_error_callback(handle_error)

    try:
        # Start development mode and get initial config
        async with dev_manager as _:
            # Start the initial workflow
            await start_workflow(config_file)

            # Keep running and waiting for workflow tasks (including reloaded ones)
            while True:
                if current_workflow_task:
                    try:
                        await current_workflow_task
                        # If workflow completed normally, break out of the loop
                        break
                    except asyncio.CancelledError:
                        logger.info("Workflow stopped for reload")
                        # Wait a bit for the reload callback to create a new task
                        await asyncio.sleep(0.1)
                        continue
                else:
                    # No current task, wait a bit and check again
                    await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    except Exception as e:
        logger.error("Development mode error: %s", e)
        raise
    finally:
        # Clean up
        if current_workflow_task and not current_workflow_task.done():
            current_workflow_task.cancel()
            try:
                await current_workflow_task
            except asyncio.CancelledError:
                pass
