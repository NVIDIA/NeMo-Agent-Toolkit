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

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from nat.observability.mixin.file_mode import FileMode
from nat.observability.mixin.resource_conflict_mixin import ResourceConflictMixin

logger = logging.getLogger(__name__)


class FileExportMixin(ResourceConflictMixin):
    """Mixin for file-based exporters.

    This mixin provides file I/O functionality for exporters that need to write
    serialized data to local files, with support for file overwriting and rolling logs.

    Automatically detects and prevents file path conflicts between multiple instances
    by raising ResourceConflictError during initialization.
    """

    def __init__(
            self,
            *args,
            output_path,
            project,
            mode: FileMode = FileMode.APPEND,
            enable_rolling: bool = False,
            max_file_size: int = 10 * 1024 * 1024,  # 10MB default
            max_files: int = 5,
            cleanup_on_init: bool = False,
            **kwargs):
        """Initialize the file exporter with the specified output_path and project.

        Args:
            output_path (str): The path to the output file or directory (if rolling enabled).
            project (str): The project name for metadata.
            mode (str): Either "append" or "overwrite". Defaults to "append".
            enable_rolling (bool): Enable rolling log files. Defaults to False.
            max_file_size (int): Maximum file size in bytes before rolling. Defaults to 10MB.
            max_files (int): Maximum number of rolled files to keep. Defaults to 5.
            cleanup_on_init (bool): Clean up old files during initialization. Defaults to False.

        Raises:
            ResourceConflictError: If another FileExportMixin instance is already using
                                 the same file path or would create conflicting files.
        """
        self._filepath = Path(output_path)
        self._project = project
        self._mode = mode
        self._enable_rolling = enable_rolling
        self._max_file_size = max_file_size
        self._max_files = max_files
        self._cleanup_on_init = cleanup_on_init
        self._lock = asyncio.Lock()
        self._first_write = True

        # Initialize file paths first, then check for conflicts via ResourceConflictMixin
        self._setup_file_paths()

        # This calls _register_resources() which will check for conflicts
        super().__init__(*args, **kwargs)

    def _setup_file_paths(self):
        """Setup file paths using the project name."""

        if self._enable_rolling:
            # If rolling is enabled, output_path should be a directory
            self._base_dir = self._filepath if self._filepath.is_dir(
            ) or not self._filepath.suffix else self._filepath.parent
            self._base_filename = self._filepath.stem if self._filepath.suffix else f"{self._project}_export"
            self._file_extension = self._filepath.suffix or ".log"
            self._base_dir.mkdir(parents=True, exist_ok=True)
            self._current_file_path = self._base_dir / f"{self._base_filename}{self._file_extension}"

            # Perform initial cleanup if requested
            if self._cleanup_on_init:
                self._cleanup_old_files_sync()
        else:
            # Traditional single file mode
            self._filepath.parent.mkdir(parents=True, exist_ok=True)
            self._current_file_path = self._filepath

            # For single file mode with overwrite, remove existing file
            if self._mode == FileMode.OVERWRITE and self._cleanup_on_init and self._current_file_path.exists():
                try:
                    self._current_file_path.unlink()
                    logger.info("Cleaned up existing file: %s", self._current_file_path)
                except OSError as e:
                    logger.error("Error removing existing file %s: %s", self._current_file_path, e)

    def _get_resource_identifiers(self) -> dict[str, Any]:
        """Return the file resources this instance will use.

        Returns:
            dict with file_path and optionally cleanup_pattern for rolling files.
        """
        identifiers = {"file_path": str(self._current_file_path.resolve())}

        # Add cleanup pattern for rolling files
        if self._enable_rolling:
            cleanup_pattern = f"{self._base_filename}_*{self._file_extension}"
            pattern_key = f"{self._base_dir.resolve()}:{cleanup_pattern}"
            identifiers["cleanup_pattern"] = pattern_key

        return identifiers

    def _format_conflict_error(self, resource_type: str, identifier: Any, existing_instance: Any) -> str:
        """Format user-friendly error messages for file conflicts."""
        match resource_type:
            case "file_path":
                return (f"File path conflict detected: '{self._current_file_path}' is already in use by another "
                        f"FileExportMixin instance (project: '{existing_instance._project}'). "
                        f"Use different project names or output paths to avoid conflicts.")
            case "cleanup_pattern":
                return (f"Rolling file cleanup conflict detected: Both instances would use pattern "
                        f"'{self._base_filename}_*{self._file_extension}' in directory '{self._base_dir}', "
                        f"causing one to delete the other's files. "
                        f"Current instance (project: '{self._project}'), "
                        f"existing instance (project: '{existing_instance._project}'). "
                        f"Use different project names or directories to avoid conflicts.")
            case _:
                return f"Unknown file resource conflict: {resource_type} = {identifier}"

    def _cleanup_old_files_sync(self) -> None:
        """Synchronous version of cleanup for use during initialization."""
        try:
            # Find all rolled files matching our pattern
            pattern = f"{self._base_filename}_*{self._file_extension}"
            rolled_files = list(self._base_dir.glob(pattern))

            # Sort by modification time (newest first)
            rolled_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            # Remove files beyond max_files limit
            for old_file in rolled_files[self._max_files:]:
                try:
                    old_file.unlink()
                    logger.info("Cleaned up old log file during init: %s", old_file)
                except OSError as e:
                    logger.error("Error removing old file %s: %s", old_file, e)

        except Exception as e:
            logger.error("Error during initialization cleanup: %s", e)

    async def _should_roll_file(self) -> bool:
        """Check if the current file should be rolled based on size."""
        if not self._enable_rolling:
            return False

        try:
            if self._current_file_path.exists():
                stat = self._current_file_path.stat()
                return stat.st_size >= self._max_file_size
        except OSError:
            pass
        return False

    async def _roll_file(self) -> None:
        """Roll the current file by renaming it with a timestamp and cleaning up old files."""
        if not self._current_file_path.exists():
            return

        # Generate timestamped filename with microsecond precision
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        rolled_filename = f"{self._base_filename}_{timestamp}{self._file_extension}"
        rolled_path = self._base_dir / rolled_filename

        try:
            # Rename current file
            self._current_file_path.rename(rolled_path)
            logger.info("Rolled log file to: %s", rolled_path)

            # Clean up old files
            await self._cleanup_old_files()

        except OSError as e:
            logger.error("Error rolling file %s: %s", self._current_file_path, e)

    async def _cleanup_old_files(self) -> None:
        """Remove old rolled files beyond the maximum count."""
        try:
            # Find all rolled files matching our pattern
            pattern = f"{self._base_filename}_*{self._file_extension}"
            rolled_files = list(self._base_dir.glob(pattern))

            # Sort by modification time (newest first)
            rolled_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            # Remove files beyond max_files limit
            for old_file in rolled_files[self._max_files:]:
                try:
                    old_file.unlink()
                    logger.info("Cleaned up old log file: %s", old_file)
                except OSError as e:
                    logger.error("Error removing old file %s: %s", old_file, e)

        except Exception as e:
            logger.error("Error during cleanup: %s", e)

    async def export_processed(self, item: str | list[str]) -> None:
        """Export a processed string or list of strings.

        Args:
            item (str | list[str]): The string or list of strings to export.
        """
        try:
            # Lazy import to avoid slow startup times
            import aiofiles

            async with self._lock:
                # Check if we need to roll the file
                if await self._should_roll_file():
                    await self._roll_file()

                # Determine file mode
                if self._first_write and self._mode == FileMode.OVERWRITE:
                    file_mode = "w"
                    self._first_write = False
                else:
                    file_mode = "a"

                async with aiofiles.open(self._current_file_path, mode=file_mode) as f:
                    if isinstance(item, list):
                        # Handle list of strings
                        for single_item in item:
                            await f.write(single_item)
                            await f.write("\n")
                    else:
                        # Handle single string
                        await f.write(item)
                        await f.write("\n")

        except Exception as e:
            logger.error("Error exporting event: %s", e, exc_info=True)

    def get_current_file_path(self) -> Path:
        """Get the current file path being written to.

        Returns:
            Path: The current file path being written to.
        """
        return self._current_file_path

    def get_file_info(self) -> dict:
        """Get information about the current file and rolling configuration.

        Returns:
            dict: A dictionary containing the current file path, mode, rolling enabled, cleanup on init,
                  effective project name, and additional rolling configuration if enabled.
        """
        info = {
            "current_file": str(self._current_file_path),
            "mode": self._mode,
            "rolling_enabled": self._enable_rolling,
            "cleanup_on_init": self._cleanup_on_init,
            "project": self._project,
            "effective_project": self._project,
        }

        if self._enable_rolling:
            info.update({
                "max_file_size": self._max_file_size,
                "max_files": self._max_files,
                "base_directory": str(self._base_dir),
            })

            # Add current file size if it exists
            if self._current_file_path.exists():
                info["current_file_size"] = self._current_file_path.stat().st_size

        return info
