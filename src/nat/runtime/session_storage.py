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
Session storage for managing cookies and headers across WebSocket and HTTP contexts.

This module implements a pattern similar to MCP Inspector's session-based header management,
allowing cookies extracted from WebSocket connections to be available during HTTP authentication flows.

Cleanup:
- Sessions are automatically cleaned up when WebSocket connections close
- Expired sessions (older than SESSION_TIMEOUT) are cleaned up by background task
- Manual cleanup is available via clear_session() and cleanup_expired_sessions()

Usage:
    # Start background cleanup (typically in application startup)
    await start_background_cleanup(interval_seconds=300)  # 5 minutes

    # Manual cleanup
    cleanup_expired_sessions()

    # WebSocket disconnect cleanup (called automatically)
    session_manager.cleanup_websocket_session(websocket)
"""

import logging
import threading
import time
from typing import Any
from typing import Dict
from typing import Optional

logger = logging.getLogger(__name__)

# Global session storage (thread-safe)
# Maps session_id -> {cookies: {...}, headers: {...}}
_SESSION_STORAGE: Dict[str, Dict[str, Any]] = {}
_SESSION_TIMESTAMPS: Dict[str, float] = {}  # session_id -> timestamp
_STORAGE_LOCK = threading.RLock()

# Session timeout (1 hour)
SESSION_TIMEOUT = 3600


def store_session_cookies(session_id: str, cookies: Dict[str, str]) -> None:
    """
    Store cookies for a specific session.

    Args:
        session_id: Unique session identifier
        cookies: Dictionary of cookies to store
    """
    if not session_id:
        logger.warning("Cannot store cookies: session_id is empty")
        return

    with _STORAGE_LOCK:
        if session_id not in _SESSION_STORAGE:
            _SESSION_STORAGE[session_id] = {}
        _SESSION_STORAGE[session_id]["cookies"] = cookies.copy()
        _SESSION_TIMESTAMPS[session_id] = time.time()
        logger.debug(f"Stored cookies for session {session_id}: {list(cookies.keys())}")


def get_session_cookies(session_id: str) -> Dict[str, str]:
    """
    Retrieve cookies for a specific session.

    Args:
        session_id: Unique session identifier

    Returns:
        Dictionary of cookies, empty dict if session not found
    """
    if not session_id:
        return {}

    with _STORAGE_LOCK:
        session_data = _SESSION_STORAGE.get(session_id, {})
        cookies = session_data.get("cookies", {})
        logger.debug(f"Retrieved cookies for session {session_id}: {list(cookies.keys())}")
        return cookies.copy()


def store_session_headers(session_id: str, headers: Dict[str, str]) -> None:
    """
    Store headers for a specific session.

    Args:
        session_id: Unique session identifier
        headers: Dictionary of headers to store
    """
    if not session_id:
        logger.warning("Cannot store headers: session_id is empty")
        return

    with _STORAGE_LOCK:
        if session_id not in _SESSION_STORAGE:
            _SESSION_STORAGE[session_id] = {}
        _SESSION_STORAGE[session_id]["headers"] = headers.copy()
        _SESSION_TIMESTAMPS[session_id] = time.time()
        logger.debug(f"Stored headers for session {session_id}: {list(headers.keys())}")


def get_session_headers(session_id: str) -> Dict[str, str]:
    """
    Retrieve headers for a specific session.

    Args:
        session_id: Unique session identifier

    Returns:
        Dictionary of headers, empty dict if session not found
    """
    if not session_id:
        return {}

    with _STORAGE_LOCK:
        session_data = _SESSION_STORAGE.get(session_id, {})
        headers = session_data.get("headers", {})
        logger.debug(f"Retrieved headers for session {session_id}: {list(headers.keys())}")
        return headers.copy()


def clear_session(session_id: str) -> None:
    """
    Clear all data for a specific session.

    Args:
        session_id: Unique session identifier
    """
    if not session_id:
        return

    with _STORAGE_LOCK:
        if session_id in _SESSION_STORAGE:
            del _SESSION_STORAGE[session_id]
        if session_id in _SESSION_TIMESTAMPS:
            del _SESSION_TIMESTAMPS[session_id]
        logger.debug(f"Cleared session {session_id}")


def get_session_count() -> int:
    """
    Get the number of active sessions.

    Returns:
        Number of active sessions
    """
    with _STORAGE_LOCK:
        return len(_SESSION_STORAGE)


def extract_session_id_from_cookies(cookies: Dict[str, str]) -> Optional[str]:
    """
    Extract session ID from cookies dictionary.

    Args:
        cookies: Dictionary of cookies

    Returns:
        Session ID if found, None otherwise
    """
    return cookies.get("nat-session")


def extract_session_id_from_headers(headers: Dict[str, str]) -> Optional[str]:
    """
    Extract session ID from headers dictionary.

    Args:
        headers: Dictionary of headers

    Returns:
        Session ID if found, None otherwise
    """
    # Check for session ID in various header formats
    session_id = headers.get("mcp-session-id")
    if session_id:
        return session_id

    # Check for session ID in cookie header
    cookie_header = headers.get("cookie", "")
    if "nat-session=" in cookie_header:
        # Parse cookie header to extract nat-session value
        for cookie in cookie_header.split(";"):
            cookie = cookie.strip()
            if cookie.startswith("nat-session="):
                return cookie.split("=", 1)[1]

    return None


def cleanup_expired_sessions() -> int:
    """
    Remove sessions older than SESSION_TIMEOUT.

    Returns:
        Number of sessions cleaned up
    """
    current_time = time.time()
    cleaned_count = 0

    with _STORAGE_LOCK:
        expired_sessions = [
            session_id for session_id, timestamp in _SESSION_TIMESTAMPS.items()
            if current_time - timestamp > SESSION_TIMEOUT
        ]

        for session_id in expired_sessions:
            if session_id in _SESSION_STORAGE:
                del _SESSION_STORAGE[session_id]
            if session_id in _SESSION_TIMESTAMPS:
                del _SESSION_TIMESTAMPS[session_id]
            cleaned_count += 1
            logger.debug(f"Cleaned up expired session {session_id}")

    if cleaned_count > 0:
        logger.info(f"Cleaned up {cleaned_count} expired sessions")

    return cleaned_count


def get_expired_session_count() -> int:
    """
    Get the number of expired sessions.

    Returns:
        Number of expired sessions
    """
    current_time = time.time()
    with _STORAGE_LOCK:
        return sum(1 for timestamp in _SESSION_TIMESTAMPS.values() if current_time - timestamp > SESSION_TIMEOUT)


def get_session_info() -> Dict[str, Any]:
    """
    Get information about all sessions.

    Returns:
        Dictionary with session information
    """
    current_time = time.time()
    with _STORAGE_LOCK:
        active_sessions = len(_SESSION_STORAGE)
        expired_sessions = get_expired_session_count()

        session_details = {}
        for session_id, timestamp in _SESSION_TIMESTAMPS.items():
            age_seconds = current_time - timestamp
            session_details[session_id] = {
                "age_seconds": age_seconds,
                "is_expired": age_seconds > SESSION_TIMEOUT,
                "has_cookies": "cookies" in _SESSION_STORAGE.get(session_id, {}),
                "has_headers": "headers" in _SESSION_STORAGE.get(session_id, {})
            }

        return {
            "active_sessions": active_sessions,
            "expired_sessions": expired_sessions,
            "total_sessions": len(_SESSION_TIMESTAMPS),
            "session_timeout": SESSION_TIMEOUT,
            "sessions": session_details
        }


# Background cleanup task
_cleanup_task = None


async def start_background_cleanup(interval_seconds: int = 300) -> None:
    """
    Start background cleanup task to remove expired sessions.

    Args:
        interval_seconds: How often to run cleanup (default: 5 minutes)
    """
    global _cleanup_task

    if _cleanup_task is not None:
        logger.warning("Background cleanup task is already running")
        return

    async def cleanup_loop():
        while True:
            try:
                import asyncio
                await asyncio.sleep(interval_seconds)
                cleaned = cleanup_expired_sessions()
                if cleaned > 0:
                    logger.info(f"Background cleanup removed {cleaned} expired sessions")
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")

    _cleanup_task = asyncio.create_task(cleanup_loop())
    logger.info(f"Started background session cleanup (interval: {interval_seconds}s)")


async def stop_background_cleanup() -> None:
    """Stop the background cleanup task."""
    global _cleanup_task

    if _cleanup_task is not None:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
        _cleanup_task = None
        logger.info("Stopped background session cleanup")
