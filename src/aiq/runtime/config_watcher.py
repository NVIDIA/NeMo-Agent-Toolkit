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
Configuration file watcher for AIQ Toolkit.

This module provides file system monitoring capabilities for configuration files,
enabling hot-reloading functionality by detecting file changes and dispatching
appropriate events.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from aiq.runtime.events import ConfigChangeEvent
from aiq.runtime.events import ConfigEventType
from aiq.runtime.events import get_event_manager
from aiq.utils.type_utils import StrPath

logger = logging.getLogger(__name__)


class ConfigFileHandler(FileSystemEventHandler):
    """
    File system event handler for configuration files.

    This handler processes file system events and converts them to configuration
    change events that are dispatched through the event system.
    """

    def __init__(self, watched_files: set[Path], debounce_delay: float = 0.1) -> None:
        """
        Initialize the configuration file handler.

        Parameters
        ----------
        watched_files : set[Path]
            Set of file paths to monitor for changes
        debounce_delay : float, optional
            Delay in seconds to debounce rapid file changes, by default 0.1
        """
        super().__init__()
        self.watched_files = watched_files
        self.debounce_delay = debounce_delay
        self._pending_events: dict[Path, float] = {}
        self._file_checksums: dict[Path, str] = {}
        self._lock = threading.Lock()
        self._debounce_timer: threading.Timer | None = None

        # Initialize file checksums
        self._update_file_checksums()

    def _update_file_checksums(self) -> None:
        """Update checksums for all watched files."""
        with self._lock:
            for file_path in self.watched_files:
                if file_path.exists():
                    self._file_checksums[file_path] = self._calculate_file_checksum(file_path)

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """
        Calculate SHA256 checksum of a file.

        Parameters
        ----------
        file_path : Path
            Path to the file

        Returns
        -------
        str
            SHA256 checksum of the file
        """
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except (OSError, IOError) as e:
            logger.warning("Failed to calculate checksum for %s: %s", file_path, e)
            return ""

    def _should_process_event(self, event: FileSystemEvent) -> bool:
        """
        Check if the event should be processed.

        Parameters
        ----------
        event : FileSystemEvent
            The file system event to check

        Returns
        -------
        bool
            True if the event should be processed, False otherwise
        """
        if event.is_directory:
            return False

        event_path = Path(event.src_path)

        # Check if the file is in our watched files
        if event_path not in self.watched_files:
            return False

        # For modification events, check if the file actually changed
        if event.event_type == 'modified':
            new_checksum = self._calculate_file_checksum(event_path)
            with self._lock:
                old_checksum = self._file_checksums.get(event_path, "")
                if new_checksum == old_checksum:
                    return False
                self._file_checksums[event_path] = new_checksum

        return True

    def _process_pending_events(self) -> None:
        """Process all pending events after debounce delay."""
        with self._lock:
            current_time = time.time()
            events_to_process = []

            for file_path, event_time in list(self._pending_events.items()):
                if current_time - event_time >= self.debounce_delay:
                    events_to_process.append(file_path)
                    del self._pending_events[file_path]

            for file_path in events_to_process:
                self._dispatch_config_event(file_path)

            # Schedule next processing if there are still pending events
            if self._pending_events:
                self._debounce_timer = threading.Timer(self.debounce_delay, self._process_pending_events)
                self._debounce_timer.start()
            else:
                self._debounce_timer = None

    def _dispatch_config_event(self, file_path: Path) -> None:
        """
        Dispatch a configuration change event.

        Parameters
        ----------
        file_path : Path
            Path to the changed file
        """
        event_type = ConfigEventType.FILE_MODIFIED
        checksum = self._file_checksums.get(file_path)

        event = ConfigChangeEvent(event_type=event_type, file_path=file_path, checksum=checksum)

        get_event_manager().dispatch_event(event)

    def _schedule_event_processing(self, file_path: Path) -> None:
        """
        Schedule event processing with debouncing.

        Parameters
        ----------
        file_path : Path
            Path to the file that changed
        """
        with self._lock:
            self._pending_events[file_path] = time.time()

            # Start debounce timer if not already running
            if self._debounce_timer is None:
                self._debounce_timer = threading.Timer(self.debounce_delay, self._process_pending_events)
                self._debounce_timer.start()

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if self._should_process_event(event):
            logger.debug("Configuration file modified: %s", event.src_path)
            self._schedule_event_processing(Path(event.src_path))

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if self._should_process_event(event):
            logger.debug("Configuration file created: %s", event.src_path)
            self._schedule_event_processing(Path(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events."""
        if not event.is_directory:
            event_path = Path(event.src_path)
            if event_path in self.watched_files:
                logger.debug("Configuration file deleted: %s", event.src_path)

                # Remove from checksums
                with self._lock:
                    self._file_checksums.pop(event_path, None)

                # Dispatch delete event immediately (no debouncing needed)
                delete_event = ConfigChangeEvent(event_type=ConfigEventType.FILE_DELETED, file_path=event_path)
                get_event_manager().dispatch_event(delete_event)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move events."""
        if not event.is_directory:
            src_path = Path(event.src_path)
            dest_path = Path(event.dest_path)

            if src_path in self.watched_files or dest_path in self.watched_files:
                logger.debug("Configuration file moved: %s -> %s", event.src_path, event.dest_path)

                # Update checksums
                with self._lock:
                    if src_path in self._file_checksums:
                        checksum = self._file_checksums.pop(src_path)
                        if dest_path in self.watched_files:
                            self._file_checksums[dest_path] = checksum

                # Dispatch move event immediately
                move_event = ConfigChangeEvent(event_type=ConfigEventType.FILE_MOVED,
                                               file_path=dest_path,
                                               old_path=src_path)
                get_event_manager().dispatch_event(move_event)


class ConfigWatcher:
    """
    Configuration file watcher for AIQ Toolkit.

    This class provides file system monitoring capabilities for configuration files,
    enabling hot-reloading functionality by detecting file changes and dispatching
    appropriate events through the event system.
    """

    def __init__(self, debounce_delay: float = 0.1) -> None:
        """
        Initialize the configuration watcher.

        Parameters
        ----------
        debounce_delay : float, optional
            Delay in seconds to debounce rapid file changes, by default 0.1
        """
        self.debounce_delay = debounce_delay
        self._observer: Observer | None = None
        self._watched_files: set[Path] = set()
        self._watched_directories: dict[Path, ConfigFileHandler] = {}
        self._is_running = False
        self._lock = threading.Lock()

    def add_file(self, file_path: StrPath) -> None:
        """
        Add a file to be watched for changes.

        Parameters
        ----------
        file_path : StrPath
            Path to the file to watch
        """
        file_path = Path(file_path).resolve()

        with self._lock:
            if file_path in self._watched_files:
                logger.debug("File %s is already being watched", file_path)
                return

            if not file_path.exists():
                logger.warning("File %s does not exist, cannot watch", file_path)
                return

            self._watched_files.add(file_path)
            logger.info("Added file %s to configuration watcher", file_path)

            # If watcher is running, set up monitoring for the new file
            if self._is_running:
                self._setup_file_monitoring(file_path)

    def remove_file(self, file_path: StrPath) -> None:
        """
        Remove a file from being watched.

        Parameters
        ----------
        file_path : StrPath
            Path to the file to stop watching
        """
        file_path = Path(file_path).resolve()

        with self._lock:
            if file_path not in self._watched_files:
                logger.debug("File %s is not being watched", file_path)
                return

            self._watched_files.remove(file_path)
            logger.info("Removed file %s from configuration watcher", file_path)

            # Clean up monitoring for directories that no longer have watched files
            if self._is_running:
                self._cleanup_unused_directories()

    def _setup_file_monitoring(self, file_path: Path) -> None:
        """Set up monitoring for a specific file."""
        if not self._observer:
            return

        directory = file_path.parent

        # Check if we're already monitoring this directory
        if directory in self._watched_directories:
            # Update the handler's watched files
            handler = self._watched_directories[directory]
            handler.watched_files.add(file_path)
            handler._update_file_checksums()
        else:
            # Create new handler for this directory
            files_in_dir = {f for f in self._watched_files if f.parent == directory}
            handler = ConfigFileHandler(files_in_dir, self.debounce_delay)

            self._watched_directories[directory] = handler
            self._observer.schedule(handler, str(directory), recursive=False)
            logger.debug("Started monitoring directory %s", directory)

    def _cleanup_unused_directories(self) -> None:
        """Clean up monitoring for directories that no longer have watched files."""
        directories_to_remove = []

        for directory, handler in self._watched_directories.items():
            # Find files in this directory that are still being watched
            files_in_dir = {f for f in self._watched_files if f.parent == directory}

            if not files_in_dir:
                # No more files in this directory, remove monitoring
                directories_to_remove.append(directory)
            else:
                # Update handler's watched files
                handler.watched_files = files_in_dir
                handler._update_file_checksums()

        # Remove unused directories
        for directory in directories_to_remove:
            del self._watched_directories[directory]
            logger.debug("Stopped monitoring directory %s", directory)

    def start(self) -> None:
        """Start the configuration file watcher."""
        with self._lock:
            if self._is_running:
                logger.warning("Configuration watcher is already running")
                return

            if not self._watched_files:
                logger.warning("No files to watch, configuration watcher not started")
                return

            self._observer = Observer()

            # Group files by directory and set up monitoring
            directories: dict[Path, set[Path]] = {}
            for file_path in self._watched_files:
                directory = file_path.parent
                if directory not in directories:
                    directories[directory] = set()
                directories[directory].add(file_path)

            # Create handlers for each directory
            for directory, files in directories.items():
                handler = ConfigFileHandler(files, self.debounce_delay)
                self._watched_directories[directory] = handler
                self._observer.schedule(handler, str(directory), recursive=False)
                logger.debug("Started monitoring directory %s with %d files", directory, len(files))

            self._observer.start()
            self._is_running = True
            logger.info("Configuration watcher started monitoring %d files", len(self._watched_files))

    def stop(self) -> None:
        """Stop the configuration file watcher."""
        with self._lock:
            if not self._is_running:
                logger.debug("Configuration watcher is not running")
                return

            if self._observer:
                self._observer.stop()
                self._observer.join()
                self._observer = None

            # Clean up handlers
            for handler in self._watched_directories.values():
                if handler._debounce_timer:
                    handler._debounce_timer.cancel()

            self._watched_directories.clear()
            self._is_running = False
            logger.info("Configuration watcher stopped")

    def is_running(self) -> bool:
        """
        Check if the watcher is currently running.

        Returns
        -------
        bool
            True if the watcher is running, False otherwise
        """
        with self._lock:
            return self._is_running

    def get_watched_files(self) -> list[Path]:
        """
        Get the list of files being watched.

        Returns
        -------
        List[Path]
            List of file paths being watched
        """
        with self._lock:
            return list(self._watched_files)

    def __enter__(self) -> ConfigWatcher:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
