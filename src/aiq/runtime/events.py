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
Configuration change event system for AIQ Toolkit.

This module defines the event types and handlers for configuration file changes,
providing the foundation for hot-reloading functionality.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


class ConfigEventType(str, Enum):
    """Types of configuration change events."""

    FILE_MODIFIED = "file_modified"
    FILE_CREATED = "file_created"
    FILE_DELETED = "file_deleted"
    FILE_MOVED = "file_moved"


class ConfigChangeEvent(BaseModel):
    """
    Represents a configuration file change event.

    This event is triggered when a configuration file is modified, created, deleted, or moved.
    """

    event_type: ConfigEventType = Field(description="Type of the configuration change event")
    file_path: Path = Field(description="Path to the changed configuration file")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the event occurred")
    old_path: Path | None = Field(default=None, description="Previous path for move events")
    checksum: str | None = Field(default=None, description="File checksum for duplicate detection")

    class Config:
        arbitrary_types_allowed = True


ConfigEventHandler = Callable[[ConfigChangeEvent], None]
"""Type alias for configuration event handlers."""


class ConfigEventManager:
    """
    Manages configuration change events and handlers.

    This class provides a centralized event system for handling configuration file changes.
    It supports registering event handlers and dispatching events to all registered handlers.
    """

    def __init__(self) -> None:
        self._handlers: dict[ConfigEventType, list[ConfigEventHandler]] = {
            event_type: []
            for event_type in ConfigEventType
        }
        self._global_handlers: list[ConfigEventHandler] = []
        self._recent_events: list[ConfigChangeEvent] = []
        self._max_recent_events: int = 100

    def register_handler(self, handler: ConfigEventHandler, event_type: ConfigEventType | None = None) -> None:
        """
        Register an event handler for configuration changes.

        Parameters
        ----------
        handler : ConfigEventHandler
            The handler function to register
        event_type : Optional[ConfigEventType]
            The specific event type to handle. If None, handles all events.
        """
        if event_type is None:
            self._global_handlers.append(handler)
            handler_name = getattr(handler, '__name__', str(handler))
            logger.debug("Registered global configuration event handler: %s", handler_name)
        else:
            self._handlers[event_type].append(handler)
            handler_name = getattr(handler, '__name__', str(handler))
            logger.debug("Registered configuration event handler for %s: %s", event_type, handler_name)

    def unregister_handler(self, handler: ConfigEventHandler, event_type: ConfigEventType | None = None) -> None:
        """
        Unregister an event handler.

        Parameters
        ----------
        handler : ConfigEventHandler
            The handler function to unregister
        event_type : Optional[ConfigEventType]
            The specific event type to unregister from. If None, removes from all handlers.
        """
        if event_type is None:
            if handler in self._global_handlers:
                self._global_handlers.remove(handler)
                handler_name = getattr(handler, '__name__', str(handler))
                logger.debug("Unregistered global configuration event handler: %s", handler_name)
        else:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
                handler_name = getattr(handler, '__name__', str(handler))
                logger.debug("Unregistered configuration event handler for %s: %s", event_type, handler_name)

    def dispatch_event(self, event: ConfigChangeEvent) -> None:
        """
        Dispatch a configuration change event to all registered handlers.

        Parameters
        ----------
        event : ConfigChangeEvent
            The event to dispatch
        """
        logger.debug("Dispatching configuration event: %s for file %s", event.event_type, event.file_path)

        # Store the event for recent events tracking
        self._recent_events.append(event)
        if len(self._recent_events) > self._max_recent_events:
            self._recent_events.pop(0)

        # Call specific event type handlers
        for handler in self._handlers[event.event_type]:
            try:
                handler(event)
            except Exception as e:
                handler_name = getattr(handler, '__name__', str(handler))
                logger.error("Error in configuration event handler %s: %s", handler_name, e, exc_info=True)

        # Call global handlers
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception as e:
                handler_name = getattr(handler, '__name__', str(handler))
                logger.error("Error in global configuration event handler %s: %s", handler_name, e, exc_info=True)

    def get_recent_events(self, limit: int | None = None) -> list[ConfigChangeEvent]:
        """
        Get recent configuration change events.

        Parameters
        ----------
        limit : Optional[int]
            Maximum number of events to return. If None, returns all recent events.

        Returns
        -------
        List[ConfigChangeEvent]
            List of recent events, most recent first
        """
        events = list(reversed(self._recent_events))
        if limit is not None:
            events = events[:limit]
        return events

    def clear_recent_events(self) -> None:
        """Clear the recent events history."""
        self._recent_events.clear()
        logger.debug("Cleared recent configuration events history")

    def get_handler_count(self, event_type: ConfigEventType | None = None) -> int:
        """
        Get the number of registered handlers.

        Parameters
        ----------
        event_type : Optional[ConfigEventType]
            The event type to count handlers for. If None, counts all handlers.

        Returns
        -------
        int
            Number of registered handlers
        """
        if event_type is None:
            return len(self._global_handlers) + sum(len(handlers) for handlers in self._handlers.values())
        else:
            return len(self._handlers[event_type])


# Global event manager instance
_event_manager: ConfigEventManager | None = None


def get_event_manager() -> ConfigEventManager:
    """
    Get the global configuration event manager instance.

    Returns
    -------
    ConfigEventManager
        The global event manager instance
    """
    global _event_manager
    if _event_manager is None:
        _event_manager = ConfigEventManager()
    return _event_manager


def reset_event_manager() -> None:
    """
    Reset the global event manager instance.

    This is primarily used for testing purposes to ensure a clean state.
    """
    global _event_manager
    _event_manager = None
