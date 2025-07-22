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
Unit tests for configuration events system.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from aiq.runtime.events import ConfigChangeEvent
from aiq.runtime.events import ConfigEventManager
from aiq.runtime.events import ConfigEventType
from aiq.runtime.events import get_event_manager
from aiq.runtime.events import reset_event_manager


class TestConfigEventType:
    """Test cases for ConfigEventType enum."""

    def test_event_types(self):
        """Test that all expected event types are defined."""
        assert ConfigEventType.FILE_MODIFIED == "file_modified"
        assert ConfigEventType.FILE_CREATED == "file_created"
        assert ConfigEventType.FILE_DELETED == "file_deleted"
        assert ConfigEventType.FILE_MOVED == "file_moved"


class TestConfigChangeEvent:
    """Test cases for ConfigChangeEvent model."""

    def test_basic_event_creation(self):
        """Test creating a basic configuration change event."""
        file_path = Path("/test/config.yml")
        event = ConfigChangeEvent(event_type=ConfigEventType.FILE_MODIFIED, file_path=file_path)

        assert event.event_type == ConfigEventType.FILE_MODIFIED
        assert event.file_path == file_path
        assert isinstance(event.timestamp, datetime)
        assert event.old_path is None
        assert event.checksum is None

    def test_event_with_all_fields(self):
        """Test creating an event with all fields."""
        file_path = Path("/test/config.yml")
        old_path = Path("/test/old_config.yml")
        timestamp = datetime.now()
        checksum = "abc123"

        event = ConfigChangeEvent(event_type=ConfigEventType.FILE_MOVED,
                                  file_path=file_path,
                                  timestamp=timestamp,
                                  old_path=old_path,
                                  checksum=checksum)

        assert event.event_type == ConfigEventType.FILE_MOVED
        assert event.file_path == file_path
        assert event.timestamp == timestamp
        assert event.old_path == old_path
        assert event.checksum == checksum

    def test_event_serialization(self):
        """Test event serialization to dict."""
        file_path = Path("/test/config.yml")
        event = ConfigChangeEvent(event_type=ConfigEventType.FILE_MODIFIED, file_path=file_path)

        event_dict = event.model_dump()

        assert event_dict["event_type"] == "file_modified"
        assert event_dict["file_path"] == file_path
        assert "timestamp" in event_dict


class TestConfigEventManager:
    """Test cases for ConfigEventManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ConfigEventManager()
        self.test_file = Path("/test/config.yml")

    def test_init(self):
        """Test ConfigEventManager initialization."""
        assert len(self.manager._handlers) == 4  # Four event types
        assert len(self.manager._global_handlers) == 0
        assert len(self.manager._recent_events) == 0
        assert self.manager._max_recent_events == 100

    def test_register_specific_handler(self):
        """Test registering a handler for specific event type."""
        handler = MagicMock()

        self.manager.register_handler(handler, ConfigEventType.FILE_MODIFIED)

        assert handler in self.manager._handlers[ConfigEventType.FILE_MODIFIED]
        assert handler not in self.manager._global_handlers
        assert self.manager.get_handler_count(ConfigEventType.FILE_MODIFIED) == 1

    def test_register_global_handler(self):
        """Test registering a global handler."""
        handler = MagicMock()

        self.manager.register_handler(handler)

        assert handler in self.manager._global_handlers
        assert self.manager.get_handler_count() == 1

    def test_unregister_specific_handler(self):
        """Test unregistering a specific handler."""
        handler = MagicMock()

        self.manager.register_handler(handler, ConfigEventType.FILE_MODIFIED)
        assert self.manager.get_handler_count(ConfigEventType.FILE_MODIFIED) == 1

        self.manager.unregister_handler(handler, ConfigEventType.FILE_MODIFIED)
        assert self.manager.get_handler_count(ConfigEventType.FILE_MODIFIED) == 0

    def test_unregister_global_handler(self):
        """Test unregistering a global handler."""
        handler = MagicMock()

        self.manager.register_handler(handler)
        assert self.manager.get_handler_count() == 1

        self.manager.unregister_handler(handler)
        assert self.manager.get_handler_count() == 0

    def test_dispatch_event_to_specific_handler(self):
        """Test dispatching event to specific handler."""
        handler = MagicMock()
        self.manager.register_handler(handler, ConfigEventType.FILE_MODIFIED)

        event = ConfigChangeEvent(event_type=ConfigEventType.FILE_MODIFIED, file_path=self.test_file)

        self.manager.dispatch_event(event)

        handler.assert_called_once_with(event)

    def test_dispatch_event_to_global_handler(self):
        """Test dispatching event to global handler."""
        handler = MagicMock()
        self.manager.register_handler(handler)

        event = ConfigChangeEvent(event_type=ConfigEventType.FILE_MODIFIED, file_path=self.test_file)

        self.manager.dispatch_event(event)

        handler.assert_called_once_with(event)

    def test_dispatch_event_to_multiple_handlers(self):
        """Test dispatching event to multiple handlers."""
        specific_handler = MagicMock()
        global_handler = MagicMock()

        self.manager.register_handler(specific_handler, ConfigEventType.FILE_MODIFIED)
        self.manager.register_handler(global_handler)

        event = ConfigChangeEvent(event_type=ConfigEventType.FILE_MODIFIED, file_path=self.test_file)

        self.manager.dispatch_event(event)

        specific_handler.assert_called_once_with(event)
        global_handler.assert_called_once_with(event)

    def test_dispatch_event_wrong_type(self):
        """Test that handlers for wrong event type are not called."""
        handler = MagicMock()
        self.manager.register_handler(handler, ConfigEventType.FILE_DELETED)

        event = ConfigChangeEvent(event_type=ConfigEventType.FILE_MODIFIED, file_path=self.test_file)

        self.manager.dispatch_event(event)

        handler.assert_not_called()

    def test_dispatch_event_handler_exception(self):
        """Test that handler exceptions don't break event dispatch."""
        failing_handler = MagicMock(side_effect=ValueError("Test error"))
        working_handler = MagicMock()

        self.manager.register_handler(failing_handler, ConfigEventType.FILE_MODIFIED)
        self.manager.register_handler(working_handler, ConfigEventType.FILE_MODIFIED)

        event = ConfigChangeEvent(event_type=ConfigEventType.FILE_MODIFIED, file_path=self.test_file)

        # Should not raise exception
        self.manager.dispatch_event(event)

        # Both handlers should have been called
        failing_handler.assert_called_once_with(event)
        working_handler.assert_called_once_with(event)

    def test_recent_events_tracking(self):
        """Test that recent events are tracked."""
        event1 = ConfigChangeEvent(event_type=ConfigEventType.FILE_MODIFIED, file_path=self.test_file)
        event2 = ConfigChangeEvent(event_type=ConfigEventType.FILE_DELETED, file_path=self.test_file)

        self.manager.dispatch_event(event1)
        self.manager.dispatch_event(event2)

        recent_events = self.manager.get_recent_events()

        assert len(recent_events) == 2
        assert recent_events[0] == event2  # Most recent first
        assert recent_events[1] == event1

    def test_recent_events_limit(self):
        """Test that recent events are limited."""
        self.manager._max_recent_events = 3

        events = []
        for i in range(5):
            event = ConfigChangeEvent(event_type=ConfigEventType.FILE_MODIFIED, file_path=Path(f"/test/config{i}.yml"))
            events.append(event)
            self.manager.dispatch_event(event)

        recent_events = self.manager.get_recent_events()

        # Should only keep the last 3 events
        assert len(recent_events) == 3
        assert recent_events[0] == events[4]  # Most recent first
        assert recent_events[1] == events[3]
        assert recent_events[2] == events[2]

    def test_get_recent_events_with_limit(self):
        """Test getting recent events with limit."""
        for i in range(5):
            event = ConfigChangeEvent(event_type=ConfigEventType.FILE_MODIFIED, file_path=Path(f"/test/config{i}.yml"))
            self.manager.dispatch_event(event)

        recent_events = self.manager.get_recent_events(limit=2)

        assert len(recent_events) == 2

    def test_clear_recent_events(self):
        """Test clearing recent events."""
        event = ConfigChangeEvent(event_type=ConfigEventType.FILE_MODIFIED, file_path=self.test_file)

        self.manager.dispatch_event(event)
        assert len(self.manager.get_recent_events()) == 1

        self.manager.clear_recent_events()
        assert len(self.manager.get_recent_events()) == 0

    def test_get_handler_count(self):
        """Test getting handler count."""
        handler1 = MagicMock()
        handler2 = MagicMock()
        global_handler = MagicMock()

        assert self.manager.get_handler_count() == 0
        assert self.manager.get_handler_count(ConfigEventType.FILE_MODIFIED) == 0

        self.manager.register_handler(handler1, ConfigEventType.FILE_MODIFIED)
        self.manager.register_handler(handler2, ConfigEventType.FILE_DELETED)
        self.manager.register_handler(global_handler)

        assert self.manager.get_handler_count() == 3
        assert self.manager.get_handler_count(ConfigEventType.FILE_MODIFIED) == 1
        assert self.manager.get_handler_count(ConfigEventType.FILE_DELETED) == 1
        assert self.manager.get_handler_count(ConfigEventType.FILE_CREATED) == 0


class TestGlobalEventManager:
    """Test cases for global event manager functions."""

    def setup_method(self):
        """Set up test fixtures."""
        reset_event_manager()

    def teardown_method(self):
        """Clean up test fixtures."""
        reset_event_manager()

    def test_get_event_manager_singleton(self):
        """Test that get_event_manager returns singleton."""
        manager1 = get_event_manager()
        manager2 = get_event_manager()

        assert manager1 is manager2
        assert isinstance(manager1, ConfigEventManager)

    def test_reset_event_manager(self):
        """Test resetting event manager."""
        manager1 = get_event_manager()
        handler = MagicMock()
        manager1.register_handler(handler)

        reset_event_manager()

        manager2 = get_event_manager()
        assert manager1 is not manager2
        assert manager2.get_handler_count() == 0

    def test_event_manager_persistence(self):
        """Test that event manager persists across calls."""
        manager = get_event_manager()
        handler = MagicMock()
        manager.register_handler(handler)

        # Get manager again
        manager2 = get_event_manager()
        assert manager2.get_handler_count() == 1

        # Handler should still work
        event = ConfigChangeEvent(event_type=ConfigEventType.FILE_MODIFIED, file_path=Path("/test/config.yml"))
        manager2.dispatch_event(event)

        handler.assert_called_once_with(event)
