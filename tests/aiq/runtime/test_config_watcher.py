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
Unit tests for configuration file watcher functionality.
"""

import tempfile
import time
from pathlib import Path

from aiq.runtime.config_watcher import ConfigFileHandler
from aiq.runtime.config_watcher import ConfigWatcher
from aiq.runtime.events import ConfigChangeEvent
from aiq.runtime.events import ConfigEventType
from aiq.runtime.events import get_event_manager
from aiq.runtime.events import reset_event_manager


class TestConfigFileHandler:
    """Test cases for ConfigFileHandler class."""

    def setup_method(self):
        """Set up test fixtures."""
        reset_event_manager()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create test files
        self.test_file1 = self.temp_path / "config1.yml"
        self.test_file2 = self.temp_path / "config2.yml"

        self.test_file1.write_text("config1: value1\n")
        self.test_file2.write_text("config2: value2\n")

        self.watched_files = {self.test_file1, self.test_file2}
        self.handler = ConfigFileHandler(self.watched_files, debounce_delay=0.05)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        reset_event_manager()

    def test_init(self):
        """Test ConfigFileHandler initialization."""
        assert self.handler.watched_files == self.watched_files
        assert self.handler.debounce_delay == 0.05
        assert len(self.handler._file_checksums) == 2
        assert self.test_file1 in self.handler._file_checksums
        assert self.test_file2 in self.handler._file_checksums

    def test_calculate_file_checksum(self):
        """Test file checksum calculation."""
        checksum = self.handler._calculate_file_checksum(self.test_file1)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex digest length

        # Same file should have same checksum
        checksum2 = self.handler._calculate_file_checksum(self.test_file1)
        assert checksum == checksum2

        # Different files should have different checksums
        checksum3 = self.handler._calculate_file_checksum(self.test_file2)
        assert checksum != checksum3

    def test_calculate_file_checksum_nonexistent(self):
        """Test checksum calculation for non-existent file."""
        nonexistent = self.temp_path / "nonexistent.yml"
        checksum = self.handler._calculate_file_checksum(nonexistent)
        assert checksum == ""

    def test_should_process_event_directory(self):
        """Test that directory events are ignored."""
        from watchdog.events import DirModifiedEvent

        event = DirModifiedEvent(str(self.temp_path))
        assert not self.handler._should_process_event(event)

    def test_should_process_event_unwatched_file(self):
        """Test that unwatched files are ignored."""
        from watchdog.events import FileModifiedEvent

        unwatched_file = self.temp_path / "unwatched.yml"
        unwatched_file.write_text("unwatched: value\n")

        event = FileModifiedEvent(str(unwatched_file))
        assert not self.handler._should_process_event(event)

    def test_should_process_event_watched_file(self):
        """Test that watched files are processed."""
        from watchdog.events import FileModifiedEvent

        # Modify the file to change its checksum
        self.test_file1.write_text("config1: modified_value\n")

        event = FileModifiedEvent(str(self.test_file1))
        assert self.handler._should_process_event(event)

    def test_should_process_event_no_change(self):
        """Test that files without changes are ignored."""
        from watchdog.events import FileModifiedEvent

        # File content hasn't changed
        event = FileModifiedEvent(str(self.test_file1))
        assert not self.handler._should_process_event(event)

    def test_dispatch_config_event(self):
        """Test configuration event dispatching."""
        events_received = []

        def event_handler(event):
            events_received.append(event)

        get_event_manager().register_handler(event_handler)

        self.handler._dispatch_config_event(self.test_file1)

        assert len(events_received) == 1
        event = events_received[0]
        assert isinstance(event, ConfigChangeEvent)
        assert event.event_type == ConfigEventType.FILE_MODIFIED
        assert event.file_path == self.test_file1
        assert event.checksum is not None

    def test_schedule_event_processing_debouncing(self):
        """Test event processing with debouncing."""
        events_received = []

        def event_handler(event):
            events_received.append(event)

        get_event_manager().register_handler(event_handler)

        # Schedule multiple events quickly
        self.handler._schedule_event_processing(self.test_file1)
        self.handler._schedule_event_processing(self.test_file1)
        self.handler._schedule_event_processing(self.test_file1)

        # Events should be debounced
        time.sleep(0.1)  # Wait for debounce delay

        assert len(events_received) == 1
        assert events_received[0].file_path == self.test_file1

    def test_on_modified(self):
        """Test file modification event handling."""
        from watchdog.events import FileModifiedEvent

        events_received = []

        def event_handler(event):
            events_received.append(event)

        get_event_manager().register_handler(event_handler)

        # Modify the file
        self.test_file1.write_text("config1: new_value\n")

        event = FileModifiedEvent(str(self.test_file1))
        self.handler.on_modified(event)

        # Wait for debouncing
        time.sleep(0.1)

        assert len(events_received) == 1
        assert events_received[0].event_type == ConfigEventType.FILE_MODIFIED

    def test_on_deleted(self):
        """Test file deletion event handling."""
        from watchdog.events import FileDeletedEvent

        events_received = []

        def event_handler(event):
            events_received.append(event)

        get_event_manager().register_handler(event_handler)

        event = FileDeletedEvent(str(self.test_file1))
        self.handler.on_deleted(event)

        assert len(events_received) == 1
        assert events_received[0].event_type == ConfigEventType.FILE_DELETED
        assert events_received[0].file_path == self.test_file1

        # File should be removed from checksums
        assert self.test_file1 not in self.handler._file_checksums

    def test_on_moved(self):
        """Test file move event handling."""
        from watchdog.events import FileMovedEvent

        events_received = []

        def event_handler(event):
            events_received.append(event)

        get_event_manager().register_handler(event_handler)

        new_path = self.temp_path / "moved_config.yml"

        # Add new path to watched files for move handling
        self.handler.watched_files.add(new_path)

        event = FileMovedEvent(str(self.test_file1), str(new_path))
        self.handler.on_moved(event)

        assert len(events_received) == 1
        assert events_received[0].event_type == ConfigEventType.FILE_MOVED
        assert events_received[0].file_path == new_path
        assert events_received[0].old_path == self.test_file1


class TestConfigWatcher:
    """Test cases for ConfigWatcher class."""

    def setup_method(self):
        """Set up test fixtures."""
        reset_event_manager()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir).resolve()

        # Create test files
        self.test_file1 = (self.temp_path / "config1.yml").resolve()
        self.test_file2 = (self.temp_path / "config2.yml").resolve()

        self.test_file1.write_text("config1: value1\n")
        self.test_file2.write_text("config2: value2\n")

        self.watcher = ConfigWatcher(debounce_delay=0.05)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.watcher.is_running():
            self.watcher.stop()
        shutil.rmtree(self.temp_dir)
        reset_event_manager()

    def test_init(self):
        """Test ConfigWatcher initialization."""
        assert self.watcher.debounce_delay == 0.05
        assert not self.watcher.is_running()
        assert len(self.watcher.get_watched_files()) == 0

    def test_add_file(self):
        """Test adding files to watch."""
        self.watcher.add_file(self.test_file1)
        assert self.test_file1 in self.watcher.get_watched_files()

        # Adding same file again should not duplicate
        self.watcher.add_file(self.test_file1)
        assert len(self.watcher.get_watched_files()) == 1

    def test_add_nonexistent_file(self):
        """Test adding non-existent file."""
        nonexistent = self.temp_path / "nonexistent.yml"
        self.watcher.add_file(nonexistent)
        assert nonexistent not in self.watcher.get_watched_files()

    def test_remove_file(self):
        """Test removing files from watch."""
        self.watcher.add_file(self.test_file1)
        self.watcher.add_file(self.test_file2)

        assert len(self.watcher.get_watched_files()) == 2

        self.watcher.remove_file(self.test_file1)
        assert self.test_file1 not in self.watcher.get_watched_files()
        assert self.test_file2 in self.watcher.get_watched_files()

    def test_remove_unwatched_file(self):
        """Test removing unwatched file."""
        self.watcher.add_file(self.test_file1)
        self.watcher.remove_file(self.test_file2)  # Not watched

        assert len(self.watcher.get_watched_files()) == 1
        assert self.test_file1 in self.watcher.get_watched_files()

    def test_start_stop(self):
        """Test starting and stopping the watcher."""
        self.watcher.add_file(self.test_file1)

        assert not self.watcher.is_running()

        self.watcher.start()
        assert self.watcher.is_running()

        self.watcher.stop()
        assert not self.watcher.is_running()

    def test_start_with_no_files(self):
        """Test starting watcher with no files."""
        assert not self.watcher.is_running()

        self.watcher.start()
        assert not self.watcher.is_running()  # Should not start

    def test_start_already_running(self):
        """Test starting watcher when already running."""
        self.watcher.add_file(self.test_file1)

        self.watcher.start()
        assert self.watcher.is_running()

        # Starting again should not cause issues
        self.watcher.start()
        assert self.watcher.is_running()

    def test_stop_not_running(self):
        """Test stopping watcher when not running."""
        assert not self.watcher.is_running()

        # Should not cause issues
        self.watcher.stop()
        assert not self.watcher.is_running()

    def test_context_manager(self):
        """Test watcher as context manager."""
        self.watcher.add_file(self.test_file1)

        with self.watcher:
            assert self.watcher.is_running()

        assert not self.watcher.is_running()

    def test_file_change_detection(self):
        """Test actual file change detection."""
        events_received = []

        def event_handler(event):
            events_received.append(event)

        get_event_manager().register_handler(event_handler)

        self.watcher.add_file(self.test_file1)

        with self.watcher:
            # Modify the file
            self.test_file1.write_text("config1: modified_value\n")

            # Wait for file system event and debouncing
            time.sleep(0.2)

        # Should have received a modification event
        assert len(events_received) > 0
        assert any(event.event_type == ConfigEventType.FILE_MODIFIED for event in events_received)
        assert any(event.file_path == self.test_file1 for event in events_received)

    def test_multiple_files_same_directory(self):
        """Test watching multiple files in same directory."""
        self.watcher.add_file(self.test_file1)
        self.watcher.add_file(self.test_file2)

        with self.watcher:
            assert len(self.watcher._watched_directories) == 1
            assert self.temp_path in self.watcher._watched_directories

    def test_multiple_files_different_directories(self):
        """Test watching files in different directories."""
        subdir = self.temp_path / "subdir"
        subdir.mkdir()

        subfile = (subdir / "subconfig.yml").resolve()
        subfile.write_text("subconfig: value\n")

        self.watcher.add_file(self.test_file1)
        self.watcher.add_file(subfile)

        with self.watcher:
            assert len(self.watcher._watched_directories) == 2
            assert self.temp_path in self.watcher._watched_directories
            assert subdir.resolve() in self.watcher._watched_directories

    def test_rapid_file_changes(self):
        """Test handling of rapid file changes with debouncing."""
        events_received = []

        def event_handler(event):
            events_received.append(event)

        get_event_manager().register_handler(event_handler)

        self.watcher.add_file(self.test_file1)

        with self.watcher:
            # Make rapid changes
            for i in range(5):
                self.test_file1.write_text(f"config1: value{i}\n")
                time.sleep(0.01)

            # Wait for debouncing
            time.sleep(0.2)

        # Should receive fewer events due to debouncing
        assert len(events_received) < 5

        # But should receive at least one event
        assert len(events_received) >= 1
        assert events_received[0].event_type == ConfigEventType.FILE_MODIFIED


class TestConfigWatcherIntegration:
    """Integration tests for ConfigWatcher."""

    def setup_method(self):
        """Set up test fixtures."""
        reset_event_manager()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir).resolve()

        self.config_file = (self.temp_path / "config.yml").resolve()
        self.config_file.write_text("test: config\n")

        self.watcher = ConfigWatcher()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.watcher.is_running():
            self.watcher.stop()
        shutil.rmtree(self.temp_dir)
        reset_event_manager()

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        events_received = []

        def event_handler(event):
            events_received.append(event)

        get_event_manager().register_handler(event_handler)

        # Add file and start watching
        self.watcher.add_file(self.config_file)
        self.watcher.start()

        try:
            # Modify file
            self.config_file.write_text("test: modified\n")
            time.sleep(0.2)

            # Create new file
            new_file = (self.temp_path / "new_config.yml").resolve()
            new_file.write_text("new: config\n")

            self.watcher.add_file(new_file)
            new_file.write_text("new: modified\n")
            time.sleep(0.2)

            # Delete file
            self.config_file.unlink()
            time.sleep(0.2)

        finally:
            self.watcher.stop()

        # Verify events
        assert len(events_received) >= 2

        # Should have modification and deletion events
        event_types = [event.event_type for event in events_received]
        assert ConfigEventType.FILE_MODIFIED in event_types
        assert ConfigEventType.FILE_DELETED in event_types
